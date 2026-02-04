import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, List, Any

from .config import TextConfig
from .attention import TextAttention, precompute_freqs_cis
from .moe import MoEMLP


def _apply_dense_lora(x: mx.array, lora_a: mx.array, lora_b: mx.array) -> mx.array:
    """Apply dense LoRA: x @ A.T @ B.T."""
    b, t, c = x.shape
    x_flat = x.reshape(-1, c)
    # [BT, rank]
    lora_mid = mx.matmul(x_flat, lora_a.T)
    # [BT, out_dim]
    lora_out = mx.matmul(lora_mid, lora_b.T)
    return lora_out.reshape(b, t, -1)


class DenseMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array, lora: Any = None) -> mx.array:
        h = self.fc1(x)
        if lora is not None:
            h = h + _apply_dense_lora(x, lora.up_a, lora.up_b)
        h = nn.gelu_approx(h)
        out = self.fc2(h)
        if lora is not None:
            out = out + _apply_dense_lora(h, lora.down_a, lora.down_b)
        return out


class TextBlock(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln = nn.LayerNorm(config.dim)
        self.attn = TextAttention(config.dim, config.n_heads, config.n_kv_heads)

        if config.moe is not None and layer_idx >= config.moe.start_layer:
            self.mlp = MoEMLP(
                dim=config.dim,
                n_experts=config.moe.num_experts,
                expert_dim=config.moe.expert_inner_dim,
                experts_per_token=config.moe.experts_per_token,
            )
            self.is_moe = True
        else:
            self.mlp = DenseMLP(config.dim, config.ff_dim)
            self.is_moe = False

    def __call__(
        self,
        x: mx.array,
        freqs_cis: mx.array,
        positions: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple] = None,
        cache_pos: int = 0,
        kv_quant: bool = False,
        lora: Any = None,
    ) -> Tuple[mx.array, Tuple]:
        h = self.ln(x)
        attn_out, new_cache = self.attn(h, freqs_cis, positions, mask, cache, cache_pos, kv_quant)
        mlp_out = self.mlp(h, lora=lora) if lora is not None else self.mlp(h)
        out = x + attn_out + mlp_out
        return out, new_cache


class TextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = [TextBlock(config, i) for i in range(config.n_layers)]
        self.post_ln = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size)
        rot_dim = config.dim // (2 * config.n_heads)
        self.freqs_cis = precompute_freqs_cis(rot_dim, config.max_context)

    def embed(self, tokens: mx.array) -> mx.array:
        return self.wte(tokens)

    def __call__(
        self,
        x: mx.array,
        positions: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Tuple]] = None,
        cache_pos: int = 0,
        kv_quant: bool = False,
        lora: Any = None,
    ) -> Tuple[mx.array, List[Tuple]]:
        new_caches = []
        is_prefill = positions.shape[0] > 1

        for i, block in enumerate(self.blocks):
            block_cache = cache[i] if cache is not None else None
            layer_lora = None
            if lora is not None:
                if getattr(block, "is_moe", False):
                    layer_lora = lora.moe_layer(i)
                else:
                    layer_lora = lora.dense_layer(i)
            x, new_cache = block(
                x,
                self.freqs_cis,
                positions,
                mask,
                block_cache,
                cache_pos,
                kv_quant,
                lora=layer_lora,
            )
            new_caches.append(new_cache)

            if is_prefill and (i + 1) % 4 == 0:
                mx.eval(x)

        return x, new_caches

    def generate_logits(self, hidden: mx.array, indices: Optional[mx.array] = None) -> mx.array:
        hidden = hidden[:, -1, :]
        hidden = self.post_ln(hidden)

        if indices is not None:
            logits = hidden @ self.lm_head.weight[indices].T
            if hasattr(self.lm_head, 'bias') and self.lm_head.bias is not None:
                logits = logits + self.lm_head.bias[indices]
        else:
            logits = self.lm_head(hidden)

        return logits
