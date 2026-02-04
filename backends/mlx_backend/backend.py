"""
MLX backend for Moondream Station.

Provides Moondream 3 inference via MLX on Apple Silicon.
Requires macOS with Apple Silicon (M1/M2/M3/M4).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import sys
import os
from pathlib import Path
from typing import Any

_backend_dir = Path(__file__).parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

import mlx.core as mx
from PIL import Image

from adapter_provider import MoondreamAdapterProvider

logger = logging.getLogger(__name__)

_model = None
_model_commit_hash = None
_loaded_model_id = None
_initialized = False
_quantize_mode = None
_adapter_provider = None
_lora_max_rank = 16
_lora_cache_size = 32

MODEL_ID = "moondream/moondream3-preview"
MODEL_ID_INT4 = "moondream/md3p-int4"


def _extract_commit_hash(snapshot_path: Path) -> str:
    return snapshot_path.name


def _snapshot_ok(snapshot_path: Path) -> bool:
    return (snapshot_path / "config.json").is_file() and (
        snapshot_path / "model.safetensors.index.json"
    ).is_file()


def init_backend(**kwargs: Any) -> None:
    global _initialized, _quantize_mode, _lora_max_rank, _lora_cache_size
    _quantize_mode = kwargs.get("quantize", None)
    if "lora_max_rank" in kwargs:
        _lora_max_rank = int(kwargs["lora_max_rank"])
    if "lora_cache_size" in kwargs:
        _lora_cache_size = int(kwargs["lora_cache_size"])
    _initialized = True
    if _quantize_mode:
        logger.info(f"MLX backend initialized with quantization mode: {_quantize_mode}")
    else:
        logger.info("MLX backend initialized (model will load on first request)")


def _load_config(weights_path: Path):
    """Load config from HuggingFace config.json."""
    from md3.config import MoondreamConfig

    config_path = weights_path / "config.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    # HF configs may nest actual config under "config"
    if isinstance(config_dict, dict) and isinstance(config_dict.get("config"), dict):
        config_dict = config_dict["config"]
    return MoondreamConfig.from_dict(config_dict)


def _remap_weight_name(name: str) -> str:
    """Remap HuggingFace weight names to MLX model structure."""
    if name.startswith("model."):
        name = name[6:]

    name = name.replace(".tau.alpha", ".tau_alpha")
    name = name.replace(".tau.wq", ".tau_wq")
    name = name.replace(".tau.wv", ".tau_wv")

    if name == "text.wte":
        name = "text.wte.weight"

    name = name.replace("vision.proj_mlp.fc1", "vision.proj_fc1")
    name = name.replace("vision.proj_mlp.fc2", "vision.proj_fc2")

    moe_match = re.match(r"text\.blocks\.(\d+)\.mlp\.(fc1|fc2)\.weight", name)
    if moe_match:
        block_idx = int(moe_match.group(1))
        fc_name = moe_match.group(2)
        if block_idx >= 4:
            name = f"text.blocks.{block_idx}.mlp.{fc_name}"

    return name


def _torch_to_mx(tensor):
    import numpy as np
    import torch

    if not torch.is_tensor(tensor):
        return mx.array(tensor)

    dtype = tensor.dtype
    if dtype == torch.bfloat16:
        arr = tensor.float().cpu().numpy()
        return mx.array(arr, dtype=mx.bfloat16)
    if dtype == torch.float16:
        return mx.array(tensor.cpu().numpy(), dtype=mx.float16)
    if dtype == torch.float32:
        return mx.array(tensor.cpu().numpy(), dtype=mx.float32)
    if dtype == torch.bool:
        return mx.array(tensor.cpu().numpy().astype(np.bool_))
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        return mx.array(tensor.cpu().numpy())
    return mx.array(tensor.cpu().numpy())


def _dequantize_fp8_moe(weights: dict) -> dict:
    import torch

    float8_keys = [
        k for k, v in weights.items() if torch.is_tensor(v) and "float8" in str(v.dtype)
    ]
    if not float8_keys:
        return weights

    out = dict(weights)
    for key in float8_keys:
        if key.endswith(".mlp.fc1.weight"):
            scale_key = key.replace(".mlp.fc1.weight", ".moe_quant.up_scale")
        elif key.endswith(".mlp.fc2.weight"):
            scale_key = key.replace(".mlp.fc2.weight", ".moe_quant.down_scale")
        else:
            raise RuntimeError(f"Unexpected FP8 weight key: {key}")

        scale = out.get(scale_key)
        if scale is None:
            raise RuntimeError(f"Missing FP8 scale tensor for {key} ({scale_key})")

        weight = out[key]
        if scale.ndim != 2 or weight.ndim != 3:
            raise RuntimeError(
                f"Unexpected FP8 shapes for {key}: weight={weight.shape}, scale={scale.shape}"
            )
        if weight.shape[0] != scale.shape[0] or weight.shape[1] != scale.shape[1]:
            raise RuntimeError(
                f"FP8 scale shape mismatch for {key}: weight={weight.shape}, scale={scale.shape}"
            )

        deq = weight.float() * scale[..., None]
        out[key] = deq.to(dtype=torch.bfloat16)

    # Drop scale tensors after dequantization to reduce memory
    for key in list(out.keys()):
        if key.endswith(".moe_quant.up_scale") or key.endswith(".moe_quant.down_scale"):
            out.pop(key, None)

    return out


def _load_weights_with_torch(weights_path: Path, shards: list[str]) -> dict:
    from safetensors.torch import safe_open

    weights = {}
    for shard in shards:
        shard_path = weights_path / shard
        with safe_open(str(shard_path), framework="pt") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)

    weights = _dequantize_fp8_moe(weights)

    mx_weights = {}
    for key, tensor in weights.items():
        if tensor is None:
            continue
        mx_weights[key] = _torch_to_mx(tensor)
    return mx_weights


def _load_weights_with_mx(weights_path: Path, shards: list[str]) -> dict:
    all_weights = {}
    for shard in shards:
        shard_path = weights_path / shard
        weights = mx.load(str(shard_path))
        all_weights.update(weights)
    return all_weights


def _shard_has_fp8(shard_path: Path) -> bool:
    try:
        from safetensors.torch import safe_open
    except Exception:
        return False

    with safe_open(str(shard_path), framework="pt") as f:
        for key in f.keys():
            dtype = f.get_tensor(key).dtype
            if "float8" in str(dtype):
                return True
    return False


def _load_weights(weights_path: Path, remap: bool = True):
    """Load weights from HuggingFace safetensors.

    Args:
        weights_path: Path to the weights directory
        remap: If True, remap HF names to MLX names. If False, assume already remapped.
    """
    index_path = weights_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r") as f:
            index = json.load(f)
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = ["model.safetensors"]

    use_torch = False
    for shard in shards:
        if _shard_has_fp8(weights_path / shard):
            use_torch = True
            break

    if use_torch:
        all_weights = _load_weights_with_torch(weights_path, shards)
    else:
        try:
            all_weights = _load_weights_with_mx(weights_path, shards)
        except Exception as e:
            raise RuntimeError(
                "Failed to load safetensors with MLX. If this is an FP8 checkpoint, "
                "install safetensors and torch to enable FP8 dequantization."
            ) from e

    if not remap:
        return all_weights

    remapped = {}
    for name, weight in all_weights.items():
        new_name = _remap_weight_name(name)
        if new_name is not None:
            remapped[new_name] = weight

    return remapped


def _is_quantized_weights(weights: dict) -> bool:
    """Check if weights contain pre-quantized MoE weights."""
    for name in weights.keys():
        if name.endswith("_q") and "mlp.fc" in name:
            return True
    return False


def _setup_quantized_moe(model, weights: dict):
    """Replace MoEMLP with QuantizedMoEMLP for blocks that have quantized weights.

    Manually assigns quantized weights because MLX's load_weights() cannot load
    into None attributes - it can only replace existing arrays.
    """
    from md3.moe import QuantizedMoEMLP

    for block in model.text.blocks:
        if hasattr(block, "is_moe") and block.is_moe:
            block_prefix = f"text.blocks.{block.layer_idx}.mlp"
            if f"{block_prefix}.fc1_q" in weights:
                old_mlp = block.mlp
                q_mlp = QuantizedMoEMLP(
                    dim=old_mlp.dim,
                    n_experts=old_mlp.n_experts,
                    expert_dim=old_mlp.expert_dim,
                    experts_per_token=old_mlp.experts_per_token,
                )

                # Manually assign quantized weights (load_weights can't load into None)
                q_mlp.fc1_q = weights[f"{block_prefix}.fc1_q"]
                q_mlp.fc1_scales = weights[f"{block_prefix}.fc1_scales"]
                q_mlp.fc1_biases = weights.get(f"{block_prefix}.fc1_biases")
                q_mlp.fc2_q = weights[f"{block_prefix}.fc2_q"]
                q_mlp.fc2_scales = weights[f"{block_prefix}.fc2_scales"]
                q_mlp.fc2_biases = weights.get(f"{block_prefix}.fc2_biases")

                block.mlp = q_mlp

    # Enable quantized KV cache
    model._kv_quant = True


def _get_model():
    """Get the model, loading from HuggingFace if needed."""
    global _model, _model_commit_hash, _loaded_model_id

    model_id = MODEL_ID_INT4 if _quantize_mode else MODEL_ID

    if _model is not None and _loaded_model_id == model_id:
        return _model

    if _model is not None and _loaded_model_id != model_id:
        _model = None
        _model_commit_hash = None

    from huggingface_hub import snapshot_download
    from md3 import Moondream

    logger.info(f"Loading model: {model_id}")
    weights_path = None

    try:
        weights_path = Path(snapshot_download(model_id, token=True))
    except Exception:
        pass

    if weights_path is None or not _snapshot_ok(weights_path):
        try:
            weights_path = Path(snapshot_download(model_id, token=True, force_download=True))
        except Exception:
            pass

    if weights_path is None or not _snapshot_ok(weights_path):
        try:
            weights_path = Path(snapshot_download(model_id, local_files_only=True))
            logger.info("Using cached model (could not check for updates)")
        except Exception as e:
            raise RuntimeError(
                f"Could not load model {model_id}. "
                f"Either login to HuggingFace with 'huggingface-cli login' "
                f"or ensure the model is already cached. Error: {e}"
            )

    if not _snapshot_ok(weights_path):
        raise RuntimeError(
            f"Model snapshot is incomplete for {model_id}. "
            "Delete the cached snapshot and re-download."
        )

    new_commit_hash = _extract_commit_hash(weights_path)
    logger.info(f"Model commit: {new_commit_hash[:8]}...")

    config = _load_config(weights_path)
    model = Moondream(config)

    # Load weights - int4 weights are already remapped, base weights need remapping
    is_int4_repo = model_id == MODEL_ID_INT4
    weights = _load_weights(weights_path, remap=not is_int4_repo)

    # Check if weights are pre-quantized
    if _is_quantized_weights(weights):
        logger.info("Loading pre-quantized int4 weights")
        _setup_quantized_moe(model, weights)

    model.load_weights(list(weights.items()), strict=False)

    # Delete weights dict to drop extra reference to arrays
    del weights

    mx.eval(model.parameters())
    mx.synchronize()

    _model = model
    _model_commit_hash = new_commit_hash
    _loaded_model_id = model_id
    logger.info(f"Model loaded successfully (commit: {new_commit_hash[:8]}...)")
    return _model


def _load_image(image_url: str) -> Image.Image:
    """Convert a base64 data URL to a PIL Image."""
    if image_url.startswith("data:image"):
        _, encoded = image_url.split(",", 1)
    else:
        encoded = image_url

    raw_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


def _extract_text_settings(kwargs: dict) -> dict | None:
    """Extract text sampling settings from kwargs (handles both nested and flat)."""
    settings = {}
    # Check for nested settings dict (from moondream client)
    if "settings" in kwargs and isinstance(kwargs["settings"], dict):
        nested = kwargs["settings"]
        if "temperature" in nested:
            settings["temperature"] = nested["temperature"]
        if "max_tokens" in nested:
            settings["max_tokens"] = nested["max_tokens"]
        if "top_p" in nested:
            settings["top_p"] = nested["top_p"]
    # Also check top-level kwargs (for direct API calls)
    if "temperature" in kwargs:
        settings["temperature"] = kwargs["temperature"]
    if "max_tokens" in kwargs:
        settings["max_tokens"] = kwargs["max_tokens"]
    if "top_p" in kwargs:
        settings["top_p"] = kwargs["top_p"]
    return settings if settings else None


def _extract_adapter(kwargs: dict) -> str | None:
    """Extract adapter id from kwargs (nested settings or top-level)."""
    adapter = None
    if "settings" in kwargs and isinstance(kwargs["settings"], dict):
        adapter = kwargs["settings"].get("adapter")
    if adapter is None:
        adapter = kwargs.get("adapter")
    if adapter is None:
        return None
    if not isinstance(adapter, str):
        raise TypeError("adapter must be a string or None")
    adapter = adapter.strip()
    if not adapter:
        raise ValueError("adapter must be a non-empty string")
    return adapter


def _get_adapter_provider(model) -> MoondreamAdapterProvider:
    global _adapter_provider
    if _adapter_provider is not None:
        return _adapter_provider

    api_key = os.environ.get("MOONDREAM_API_KEY")
    if not api_key:
        raise RuntimeError("MOONDREAM_API_KEY is required to load LoRA adapters")

    # Force LoRA adapters to bf16 to match the base model precision.
    dtype = mx.bfloat16
    _adapter_provider = MoondreamAdapterProvider(
        text_config=model.config.text,
        api_key=api_key,
        max_lora_rank=_lora_max_rank,
        cache_size=_lora_cache_size,
        dtype=dtype,
    )
    return _adapter_provider


def _resolve_lora(model, adapter_id: str | None):
    if adapter_id is None:
        return None
    provider = _get_adapter_provider(model)
    return provider.get(adapter_id)


def _extract_object_settings(kwargs: dict) -> dict | None:
    """Extract object sampling settings from kwargs (handles both nested and flat)."""
    settings = {}
    if "settings" in kwargs and isinstance(kwargs["settings"], dict):
        nested = kwargs["settings"]
        if "max_objects" in nested:
            settings["max_objects"] = nested["max_objects"]
    if "max_objects" in kwargs:
        settings["max_objects"] = kwargs["max_objects"]
    return settings if settings else None


def caption(
    image_url: str | None = None,
    length: str = "normal",
    stream: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Generate a caption for an image."""
    if not image_url:
        return {"error": "image_url is required"}

    try:
        model = _get_model()
        image = _load_image(image_url)
        settings = _extract_text_settings(kwargs)
        adapter_id = _extract_adapter(kwargs)
        lora = _resolve_lora(model, adapter_id)
        result = model.caption(
            image, length=length, stream=stream, settings=settings, lora=lora
        )
        mx.clear_cache()  # Release unused metal buffers
        return result
    except Exception as e:
        logger.exception("Caption failed")
        return {"error": str(e)}


def query(
    image_url: str | None = None,
    question: str | None = None,
    stream: bool = False,
    reasoning: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Answer a question about an image."""
    if not image_url or not question:
        return {"error": "image_url and question are required"}

    try:
        model = _get_model()
        image = _load_image(image_url)
        settings = _extract_text_settings(kwargs)
        adapter_id = _extract_adapter(kwargs)
        lora = _resolve_lora(model, adapter_id)
        result = model.query(
            image, question, reasoning=reasoning, stream=stream, settings=settings, lora=lora
        )
        mx.clear_cache()  # Release unused metal buffers
        return result
    except Exception as e:
        logger.exception("Query failed")
        return {"error": str(e)}


def detect(
    image_url: str | None = None,
    object: str | None = None,
    obj: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Detect objects in an image with bounding boxes."""
    target_obj = object or obj
    if not image_url or not target_obj:
        return {"error": "image_url and object are required"}

    try:
        model = _get_model()
        image = _load_image(image_url)
        settings = _extract_object_settings(kwargs)
        adapter_id = _extract_adapter(kwargs)
        lora = _resolve_lora(model, adapter_id)
        result = model.detect(image, target_obj, settings=settings, lora=lora)
        mx.clear_cache()  # Release unused metal buffers
        return {"objects": result.get("objects", [])}
    except Exception as e:
        logger.exception("Detect failed")
        return {"error": str(e)}


def point(
    image_url: str | None = None,
    object: str | None = None,
    obj: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Find points where an object appears in an image."""
    target_obj = object or obj
    if not image_url or not target_obj:
        return {"error": "image_url and object are required"}

    try:
        model = _get_model()
        image = _load_image(image_url)
        settings = _extract_object_settings(kwargs)
        adapter_id = _extract_adapter(kwargs)
        lora = _resolve_lora(model, adapter_id)
        result = model.point(image, target_obj, settings=settings, lora=lora)
        mx.clear_cache()  # Release unused metal buffers
        points = result.get("points", [])
        return {"points": points, "count": len(points)}
    except Exception as e:
        logger.exception("Point failed")
        return {"error": str(e)}
