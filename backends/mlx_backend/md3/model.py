import gc
from functools import partial

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Literal, Optional, Tuple, TypedDict
from PIL import Image
from tokenizers import Tokenizer

from .config import MoondreamConfig
from .vision import VisionEncoder
from .text import TextModel
from .region import RegionModel, SpatialRefs, bin_to_size
from .image_crops import prepare_crops


DEFAULT_MAX_TOKENS = 768
DEFAULT_TEMPERATURE = 0.5
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_OBJECTS = 150
DISABLE_KV_QUANT = True


TextSamplingSettings = TypedDict(
    "TextSamplingSettings",
    {"max_tokens": int, "temperature": float, "top_p": float},
    total=False,
)

ObjectSamplingSettings = TypedDict(
    "ObjectSamplingSettings",
    {"max_objects": int},
    total=False,
)


DEFAULT_MAX_SEQ_LEN = 1024

_compiled_cache: Dict[int, Dict[str, any]] = {}


def _get_compiled(model, name: str, fn):
    model_id = id(model)
    if model_id not in _compiled_cache:
        _compiled_cache[model_id] = {}
    cache = _compiled_cache[model_id]
    if name not in cache:
        cache[name] = mx.compile(fn)
    return cache[name]


def _clear_compiled(model, name: str = None):
    model_id = id(model)
    if model_id in _compiled_cache:
        if name is None:
            del _compiled_cache[model_id]
        elif name in _compiled_cache[model_id]:
            del _compiled_cache[model_id][name]


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def _sample_top_p(logits: mx.array, temp: float, top_p: float) -> mx.array:
    probs = mx.softmax(logits / temp, axis=-1)

    sorted_indices = mx.argsort(-probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
    cumsum = mx.cumsum(sorted_probs, axis=-1)

    mask = (cumsum - sorted_probs) > top_p
    sorted_probs = mx.where(mask, 0.0, sorted_probs)
    sorted_probs = sorted_probs / mx.sum(sorted_probs, axis=-1, keepdims=True)

    inverse_indices = mx.argsort(sorted_indices, axis=-1)
    probs = mx.take_along_axis(sorted_probs, inverse_indices, axis=-1)
    return mx.random.categorical(mx.log(probs + 1e-12), axis=-1)


class Moondream(nn.Module):
    def __init__(self, config: MoondreamConfig, max_seq_len: int = DEFAULT_MAX_SEQ_LEN):
        super().__init__()
        self.config = config
        self.max_seq_len = max_seq_len
        self.tokenizer = Tokenizer.from_pretrained("moondream/starmie-v1")
        self.vision = VisionEncoder(config.vision)
        self.text = TextModel(config.text)
        self.region = RegionModel(config.region)

        max_ctx = config.text.max_context
        attn_mask = mx.tril(mx.ones((1, 1, max_ctx, max_ctx), dtype=mx.bool_))

        patch_w = config.vision.crop_size // config.vision.enc_patch_size
        prefix_attn_len = 1 + patch_w**2

        prefix_mask = mx.ones((1, 1, prefix_attn_len, prefix_attn_len), dtype=mx.bool_)
        prefix_mask_padded = mx.pad(
            prefix_mask,
            [(0, 0), (0, 0), (0, max_ctx - prefix_attn_len), (0, max_ctx - prefix_attn_len)],
        )
        attn_mask = attn_mask | prefix_mask_padded
        self.attn_mask = attn_mask
        self._kv_quant = False

    def _get_vision_compiled(self):
        return _get_compiled(self, "vision", self.vision)

    def _get_decode_step_compiled(self):
        model_id = id(self)
        if model_id in _compiled_cache and "decode_step" in _compiled_cache[model_id]:
            return _compiled_cache[model_id]["decode_step"]

        text = self.text
        def decode_step(embedding, positions, mask, cache, cache_pos, kv_quant):
            x = embedding
            new_caches = []
            for i, block in enumerate(text.blocks):
                block_cache = cache[i] if cache is not None else None
                x, new_cache = block(x, text.freqs_cis, positions, mask, block_cache, cache_pos, kv_quant)
                new_caches.append(new_cache)
            return x, new_caches

        return _get_compiled(self, "decode_step", decode_step)

    def quantize_experts(self) -> None:
        from .moe import QuantizedMoEMLP

        for block in self.text.blocks:
            if hasattr(block, "is_moe") and block.is_moe:
                block.mlp = QuantizedMoEMLP.from_float(block.mlp)

        gc.collect()

        self._kv_quant = True

        _clear_compiled(self, "decode_step")

    def _allocate_kv_cache(self, batch_size: int = 1) -> List:
        n_layers = len(self.text.blocks)
        n_kv_heads = self.config.text.n_kv_heads
        head_dim = self.config.text.dim // self.config.text.n_heads

        cache = []

        kv_quant = self._kv_quant and not DISABLE_KV_QUANT
        if kv_quant:
            packed_dim = head_dim // 8
            scales_dim = head_dim // 64

            for _ in range(n_layers):
                k_q = mx.zeros((batch_size, n_kv_heads, self.max_seq_len, packed_dim), dtype=mx.uint32)
                k_scales = mx.zeros((batch_size, n_kv_heads, self.max_seq_len, scales_dim))
                k_biases = mx.zeros((batch_size, n_kv_heads, self.max_seq_len, scales_dim))
                v_q = mx.zeros((batch_size, n_kv_heads, self.max_seq_len, packed_dim), dtype=mx.uint32)
                v_scales = mx.zeros((batch_size, n_kv_heads, self.max_seq_len, scales_dim))
                v_biases = mx.zeros((batch_size, n_kv_heads, self.max_seq_len, scales_dim))

                cache.append((k_q, k_scales, k_biases, v_q, v_scales, v_biases))
        else:
            for _ in range(n_layers):
                k = mx.zeros((batch_size, n_kv_heads, self.max_seq_len, head_dim))
                v = mx.zeros((batch_size, n_kv_heads, self.max_seq_len, head_dim))
                cache.append((k, v))

        return cache

    def _run_vision_encoder(self, image: Image.Image) -> mx.array:
        crops, tiling = prepare_crops(
            image,
            crop_size=self.config.vision.crop_size,
            max_crops=self.config.vision.max_crops,
            overlap_margin=self.config.vision.overlap_margin,
        )
        return self._get_vision_compiled()(crops, tiling)

    def _sample_token(self, logits: mx.array, temperature: float, top_p: float) -> mx.array:
        if temperature == 0:
            return mx.argmax(logits, axis=-1, keepdims=True)
        return _sample_top_p(logits, temperature, top_p)[:, None]

    def _prefill(
        self,
        embeddings: mx.array,
        cache_pos: int,
        cache: List,
    ) -> Tuple[mx.array, List]:
        seq_len = embeddings.shape[1]
        positions = mx.arange(cache_pos, cache_pos + seq_len)
        mask = self.attn_mask[:, :, cache_pos:cache_pos + seq_len, :]
        kv_quant = self._kv_quant and not DISABLE_KV_QUANT
        hidden, new_cache = self.text(embeddings, positions, mask, cache, cache_pos, kv_quant)
        return hidden, new_cache

    def _decode_one(
        self,
        embedding: mx.array,
        cache_pos: int,
        cache: List,
    ) -> Tuple[mx.array, mx.array, List]:
        positions = mx.array([cache_pos])
        mask = None
        kv_quant = self._kv_quant and not DISABLE_KV_QUANT
        hidden, new_cache = self._get_decode_step_compiled()(
            embedding, positions, mask, cache, cache_pos, kv_quant
        )
        logits = self.text.generate_logits(hidden)
        return logits, hidden, new_cache

    def query(
        self,
        image: Image.Image,
        question: str,
        reasoning: bool = True,
        stream: bool = False,
        settings: Optional[TextSamplingSettings] = None,
    ) -> Dict:
        max_tokens = settings.get("max_tokens", DEFAULT_MAX_TOKENS) if settings else DEFAULT_MAX_TOKENS
        temperature = settings.get("temperature", DEFAULT_TEMPERATURE) if settings else DEFAULT_TEMPERATURE
        top_p = settings.get("top_p", DEFAULT_TOP_P) if settings else DEFAULT_TOP_P

        if self.config.tokenizer.templates["query"] is None:
            raise NotImplementedError("Model does not support querying.")

        img_emb = self._run_vision_encoder(image)
        bos_emb = self.text.embed(mx.array([[self.config.tokenizer.bos_id]]))
        inputs_embeds = mx.concatenate([bos_emb, img_emb[None]], axis=1)

        cache = self._allocate_kv_cache()
        _, cache = self._prefill(inputs_embeds, cache_pos=0, cache=cache)
        pos = inputs_embeds.shape[1]

        prompt_toks = self.config.tokenizer.templates["query"]["prefix"]
        prompt_toks = prompt_toks + self.tokenizer.encode(question).ids

        result = {}

        if reasoning:
            prompt_toks = prompt_toks + [self.config.tokenizer.thinking_id]
            prompt_tokens = mx.array([prompt_toks])
            prompt_emb = self.text.embed(prompt_tokens)

            hidden, cache = self._prefill(prompt_emb, pos, cache)
            pos += prompt_emb.shape[1]

            reasoning_text, pos, cache = self._generate_reasoning(hidden, pos, cache, settings)
            result["reasoning"] = {"text": reasoning_text, "grounding": []}

            suffix_toks = self.config.tokenizer.templates["query"]["suffix"]
            prompt_tokens = mx.array([suffix_toks])
            prompt_emb = self.text.embed(prompt_tokens)
            hidden, cache = self._prefill(prompt_emb, pos, cache)
            logits = self.text.generate_logits(hidden)
            next_token = self._sample_token(logits, temperature, top_p)
            pos += prompt_emb.shape[1]
        else:
            prompt_toks = prompt_toks + self.config.tokenizer.templates["query"]["suffix"]
            prompt_tokens = mx.array([prompt_toks])
            prompt_emb = self.text.embed(prompt_tokens)
            hidden, cache = self._prefill(prompt_emb, pos, cache)
            logits = self.text.generate_logits(hidden)
            next_token = self._sample_token(logits, temperature, top_p)
            pos += prompt_emb.shape[1]

        def generator():
            nonlocal next_token, pos, cache
            eos_id = self.config.tokenizer.eos_id
            tokens = []
            generated = 0

            while int(next_token[0, 0]) != eos_id and generated < max_tokens:
                token_id = int(next_token[0, 0])
                tokens.append(token_id)

                next_emb = self.text.embed(next_token)
                logits, _, cache = self._decode_one(next_emb, pos, cache)
                next_token = self._sample_token(logits, temperature, top_p)
                mx.async_eval(next_token)
                pos += 1
                generated += 1

                text = self.tokenizer.decode(tokens)
                if len(text) > 0:
                    yield text
                    tokens = []

            if tokens:
                yield self.tokenizer.decode(tokens)

        if stream:
            result["answer"] = generator()
        else:
            result["answer"] = "".join(list(generator()))

        return result

    def _generate_reasoning(
        self,
        hidden: mx.array,
        pos: int,
        cache: List,
        settings: Optional[TextSamplingSettings] = None,
    ) -> Tuple[str, int, List]:
        max_tokens = settings.get("max_tokens", DEFAULT_MAX_TOKENS) if settings else DEFAULT_MAX_TOKENS
        temperature = settings.get("temperature", DEFAULT_TEMPERATURE) if settings else DEFAULT_TEMPERATURE
        top_p = settings.get("top_p", DEFAULT_TOP_P) if settings else DEFAULT_TOP_P

        logits = self.text.generate_logits(hidden)
        next_token = self._sample_token(logits, temperature, top_p)

        tokens = []
        eos_id = self.config.tokenizer.answer_id
        generated = 0

        while int(next_token[0, 0]) != eos_id and generated < max_tokens:
            token_id = int(next_token[0, 0])
            tokens.append(token_id)

            if token_id == self.config.tokenizer.coord_id:
                coord_logits = self.region.decode_coordinate(hidden[:, -1:, :])
                coord = mx.argmax(coord_logits, axis=-1) / coord_logits.shape[-1]
                next_emb = self.region.encode_coordinate(coord.reshape(-1, 1)).reshape(1, 1, -1)
            else:
                next_emb = self.text.embed(next_token)

            logits, hidden, cache = self._decode_one(next_emb, pos, cache)
            next_token = self._sample_token(logits, temperature, top_p)
            mx.async_eval(next_token)
            pos += 1
            generated += 1

        return self.tokenizer.decode(tokens), pos, cache

    def caption(
        self,
        image: Image.Image,
        length: Literal["normal", "short", "long"] = "normal",
        stream: bool = False,
        settings: Optional[TextSamplingSettings] = None,
    ) -> Dict:
        max_tokens = settings.get("max_tokens", DEFAULT_MAX_TOKENS) if settings else DEFAULT_MAX_TOKENS
        temperature = settings.get("temperature", DEFAULT_TEMPERATURE) if settings else DEFAULT_TEMPERATURE
        top_p = settings.get("top_p", DEFAULT_TOP_P) if settings else DEFAULT_TOP_P

        if self.config.tokenizer.templates["caption"] is None:
            raise NotImplementedError("Model does not support captioning.")
        if length not in self.config.tokenizer.templates["caption"]:
            raise ValueError(f"Model does not support caption length '{length}'.")

        img_emb = self._run_vision_encoder(image)
        bos_emb = self.text.embed(mx.array([[self.config.tokenizer.bos_id]]))
        inputs_embeds = mx.concatenate([bos_emb, img_emb[None]], axis=1)

        cache = self._allocate_kv_cache()
        _, cache = self._prefill(inputs_embeds, cache_pos=0, cache=cache)
        pos = inputs_embeds.shape[1]

        prompt_toks = self.config.tokenizer.templates["caption"][length]
        prompt_tokens = mx.array([prompt_toks])
        prompt_emb = self.text.embed(prompt_tokens)
        hidden, cache = self._prefill(prompt_emb, pos, cache)
        logits = self.text.generate_logits(hidden)
        next_token = self._sample_token(logits, temperature, top_p)
        pos += prompt_emb.shape[1]

        def generator():
            nonlocal next_token, pos, cache
            eos_id = self.config.tokenizer.eos_id
            tokens = []
            generated = 0

            while int(next_token[0, 0]) != eos_id and generated < max_tokens:
                token_id = int(next_token[0, 0])
                tokens.append(token_id)

                next_emb = self.text.embed(next_token)
                logits, _, cache = self._decode_one(next_emb, pos, cache)
                next_token = self._sample_token(logits, temperature, top_p)
                mx.async_eval(next_token)
                pos += 1
                generated += 1

                text = self.tokenizer.decode(tokens)
                if len(text) > 0:
                    yield text
                    tokens = []

            if tokens:
                yield self.tokenizer.decode(tokens)

        if stream:
            return {"caption": generator()}
        else:
            return {"caption": "".join(list(generator()))}

    def detect(
        self,
        image: Image.Image,
        object: str,
        settings: Optional[ObjectSamplingSettings] = None,
    ) -> Dict:
        max_objects = settings.get("max_objects", DEFAULT_MAX_OBJECTS) if settings else DEFAULT_MAX_OBJECTS

        if self.config.tokenizer.templates["detect"] is None:
            raise NotImplementedError("Model does not support detection.")

        img_emb = self._run_vision_encoder(image)
        bos_emb = self.text.embed(mx.array([[self.config.tokenizer.bos_id]]))
        inputs_embeds = mx.concatenate([bos_emb, img_emb[None]], axis=1)

        cache = self._allocate_kv_cache()
        _, cache = self._prefill(inputs_embeds, cache_pos=0, cache=cache)
        pos = inputs_embeds.shape[1]

        prompt_toks = (
            self.config.tokenizer.templates["detect"]["prefix"]
            + self.tokenizer.encode(" " + object).ids
            + self.config.tokenizer.templates["detect"]["suffix"]
        )
        prompt_tokens = mx.array([prompt_toks])
        prompt_emb = self.text.embed(prompt_tokens)
        hidden, cache = self._prefill(prompt_emb, pos, cache)
        logits = self.text.generate_logits(hidden)
        next_token = self._sample_token(logits, 0, 0)
        pos += prompt_emb.shape[1]

        objects = self._generate_points(
            hidden[:, -1:, :], next_token, pos, cache, include_size=True, max_objects=max_objects
        )

        return {"objects": objects}

    def point(
        self,
        image: Image.Image,
        object: str,
        settings: Optional[ObjectSamplingSettings] = None,
    ) -> Dict:
        max_objects = settings.get("max_objects", DEFAULT_MAX_OBJECTS) if settings else DEFAULT_MAX_OBJECTS

        if self.config.tokenizer.templates["point"] is None:
            raise NotImplementedError("Model does not support pointing.")

        img_emb = self._run_vision_encoder(image)
        bos_emb = self.text.embed(mx.array([[self.config.tokenizer.bos_id]]))
        inputs_embeds = mx.concatenate([bos_emb, img_emb[None]], axis=1)

        cache = self._allocate_kv_cache()
        _, cache = self._prefill(inputs_embeds, cache_pos=0, cache=cache)
        pos = inputs_embeds.shape[1]

        prompt_toks = (
            self.config.tokenizer.templates["point"]["prefix"]
            + self.tokenizer.encode(" " + object).ids
            + self.config.tokenizer.templates["point"]["suffix"]
        )
        prompt_tokens = mx.array([prompt_toks])
        prompt_emb = self.text.embed(prompt_tokens)
        hidden, cache = self._prefill(prompt_emb, pos, cache)
        logits = self.text.generate_logits(hidden)
        next_token = self._sample_token(logits, 0, 0)
        pos += prompt_emb.shape[1]

        points = self._generate_points(
            hidden[:, -1:, :], next_token, pos, cache, include_size=False, max_objects=max_objects
        )

        return {"points": points}

    def _generate_points(
        self,
        hidden: mx.array,
        next_token: mx.array,
        pos: int,
        cache: List,
        include_size: bool,
        max_objects: int,
    ) -> List[Dict]:
        out = []
        eos_id = self.config.tokenizer.eos_id
        coord_id = self.config.tokenizer.coord_id

        while int(next_token[0, 0]) != eos_id and len(out) < max_objects:
            x_logits = self.region.decode_coordinate(hidden)
            x_center = float(mx.argmax(x_logits, axis=-1) / x_logits.shape[-1])
            x_coord = mx.array([[x_center]], dtype=hidden.dtype)
            next_emb = self.region.encode_coordinate(x_coord).reshape(1, 1, -1)

            logits, hidden, cache = self._decode_one(next_emb, pos, cache)
            pos += 1
            y_logits = self.region.decode_coordinate(hidden)
            y_center = float(mx.argmax(y_logits, axis=-1) / y_logits.shape[-1])
            y_coord = mx.array([[y_center]], dtype=hidden.dtype)
            next_emb = self.region.encode_coordinate(y_coord).reshape(1, 1, -1)

            if include_size:
                logits, hidden, cache = self._decode_one(next_emb, pos, cache)
                pos += 1
                size_logits = self.region.decode_size(hidden[:, -1, :])

                w_bin = int(mx.argmax(size_logits[0], axis=-1))
                h_bin = int(mx.argmax(size_logits[1], axis=-1))
                w = float(bin_to_size(mx.array([w_bin]))[0])
                h = float(bin_to_size(mx.array([h_bin]))[0])

                size_arr = mx.array([[w, h]], dtype=hidden.dtype)
                next_emb = self.region.encode_size(size_arr).reshape(1, 1, -1)

                out.append({
                    "x_min": x_center - w / 2,
                    "y_min": y_center - h / 2,
                    "x_max": x_center + w / 2,
                    "y_max": y_center + h / 2,
                })
            else:
                out.append({"x": x_center, "y": y_center})

            logits, hidden, cache = self._decode_one(next_emb, pos, cache)
            pos += 1

            coord_eos_logits = mx.stack([logits[:, coord_id], logits[:, eos_id]], axis=-1)
            next_idx = int(mx.argmax(coord_eos_logits, axis=-1)[0])
            next_token = mx.array([[coord_id if next_idx == 0 else eos_id]])

        return out
