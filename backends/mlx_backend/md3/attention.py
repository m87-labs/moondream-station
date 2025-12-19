from functools import partial

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple


def precompute_freqs_cis(dim: int, max_len: int, theta: float = 1500000.0) -> mx.array:
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
    t = mx.arange(max_len).astype(mx.float32)
    freqs = t[:, None] * freqs[None, :]
    cos_freqs = mx.cos(freqs)
    sin_freqs = mx.sin(freqs)
    return mx.stack([cos_freqs, sin_freqs], axis=-1)


def quantize_kv(
    k: mx.array,
    v: mx.array,
    bits: int = 4,
    group_size: int = 64,
    mode: str = "affine",
) -> Tuple:
    B, n_kv_heads, T, head_dim = k.shape
    has_biases = mode == "affine"

    k_flat = k.reshape(-1, head_dim)
    v_flat = v.reshape(-1, head_dim)

    k_result = mx.quantize(k_flat, bits=bits, group_size=group_size, mode=mode)
    v_result = mx.quantize(v_flat, bits=bits, group_size=group_size, mode=mode)

    if has_biases:
        k_q, k_scales, k_biases = k_result
        v_q, v_scales, v_biases = v_result
    else:
        k_q, k_scales = k_result
        v_q, v_scales = v_result

    packed_dim = k_q.shape[-1]
    scales_dim = k_scales.shape[-1]

    k_q = k_q.reshape(B, n_kv_heads, T, packed_dim)
    k_scales = k_scales.reshape(B, n_kv_heads, T, scales_dim)
    v_q = v_q.reshape(B, n_kv_heads, T, packed_dim)
    v_scales = v_scales.reshape(B, n_kv_heads, T, scales_dim)

    if has_biases:
        k_biases = k_biases.reshape(B, n_kv_heads, T, scales_dim)
        v_biases = v_biases.reshape(B, n_kv_heads, T, scales_dim)
    else:
        k_biases = mx.zeros((B, n_kv_heads, T, scales_dim))
        v_biases = mx.zeros((B, n_kv_heads, T, scales_dim))

    return (k_q, k_scales, k_biases, v_q, v_scales, v_biases)


def dequantize_kv(
    k_q: mx.array,
    k_scales: mx.array,
    k_biases: Optional[mx.array],
    v_q: mx.array,
    v_scales: mx.array,
    v_biases: Optional[mx.array],
    bits: int = 4,
    group_size: int = 64,
    mode: str = "affine",
) -> Tuple[mx.array, mx.array]:
    B, n_kv_heads, T, packed_dim = k_q.shape

    k_q_flat = k_q.reshape(-1, packed_dim)
    k_scales_flat = k_scales.reshape(-1, k_scales.shape[-1])
    v_q_flat = v_q.reshape(-1, packed_dim)
    v_scales_flat = v_scales.reshape(-1, v_scales.shape[-1])

    if k_biases is not None:
        k_biases_flat = k_biases.reshape(-1, k_biases.shape[-1])
        v_biases_flat = v_biases.reshape(-1, v_biases.shape[-1])
    else:
        k_biases_flat, v_biases_flat = None, None

    k = mx.dequantize(k_q_flat, k_scales_flat, k_biases_flat, bits=bits, group_size=group_size, mode=mode)
    v = mx.dequantize(v_q_flat, v_scales_flat, v_biases_flat, bits=bits, group_size=group_size, mode=mode)

    head_dim = k.shape[-1]

    k = k.reshape(B, n_kv_heads, T, head_dim)
    v = v.reshape(B, n_kv_heads, T, head_dim)

    return k, v


@partial(mx.compile, shapeless=True)
def _quantize_kv_affine(k: mx.array, v: mx.array) -> Tuple:
    B, n_kv_heads, T, head_dim = k.shape

    k_flat = k.reshape(-1, head_dim)
    v_flat = v.reshape(-1, head_dim)

    k_q, k_scales, k_biases = mx.quantize(k_flat, bits=4, group_size=64, mode="affine")
    v_q, v_scales, v_biases = mx.quantize(v_flat, bits=4, group_size=64, mode="affine")

    packed_dim = k_q.shape[-1]
    scales_dim = k_scales.shape[-1]

    return (
        k_q.reshape(B, n_kv_heads, -1, packed_dim),
        k_scales.reshape(B, n_kv_heads, -1, scales_dim),
        k_biases.reshape(B, n_kv_heads, -1, scales_dim),
        v_q.reshape(B, n_kv_heads, -1, packed_dim),
        v_scales.reshape(B, n_kv_heads, -1, scales_dim),
        v_biases.reshape(B, n_kv_heads, -1, scales_dim),
    )


@partial(mx.compile, shapeless=True)
def _dequantize_kv_affine(
    k_q: mx.array,
    k_scales: mx.array,
    k_biases: mx.array,
    v_q: mx.array,
    v_scales: mx.array,
    v_biases: mx.array,
) -> Tuple[mx.array, mx.array]:
    B, n_kv_heads, T, packed_dim = k_q.shape

    k_q_flat = k_q.reshape(-1, packed_dim)
    k_scales_flat = k_scales.reshape(-1, k_scales.shape[-1])
    k_biases_flat = k_biases.reshape(-1, k_biases.shape[-1])
    v_q_flat = v_q.reshape(-1, packed_dim)
    v_scales_flat = v_scales.reshape(-1, v_scales.shape[-1])
    v_biases_flat = v_biases.reshape(-1, v_biases.shape[-1])

    k = mx.dequantize(k_q_flat, k_scales_flat, k_biases_flat, bits=4, group_size=64, mode="affine")
    v = mx.dequantize(v_q_flat, v_scales_flat, v_biases_flat, bits=4, group_size=64, mode="affine")

    head_dim = k.shape[-1]
    return k.reshape(B, n_kv_heads, -1, head_dim), v.reshape(B, n_kv_heads, -1, head_dim)


def _quantized_scaled_dot_product_attention(
    queries: mx.array,
    q_keys: Tuple[mx.array, mx.array, mx.array],
    q_values: Tuple[mx.array, mx.array, mx.array],
    scale: float,
    mask: Optional[mx.array],
    group_size: int = 64,
    bits: int = 4,
) -> mx.array:
    B, n_q_heads, q_len, head_dim = queries.shape
    if len(q_keys) == 2:
        k_q, k_scales = q_keys
        k_biases = None
    else:
        k_q, k_scales, k_biases = q_keys

    if len(q_values) == 2:
        v_q, v_scales = q_values
        v_biases = None
    else:
        v_q, v_scales, v_biases = q_values

    n_kv_heads = k_q.shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    queries = queries * scale

    if n_repeats > 1:
        queries = queries.reshape(B, n_kv_heads, n_repeats, q_len, head_dim)
        k_q = mx.expand_dims(k_q, axis=-3)
        k_scales = mx.expand_dims(k_scales, axis=-3)
        k_biases = mx.expand_dims(k_biases, axis=-3)
        v_q = mx.expand_dims(v_q, axis=-3)
        v_scales = mx.expand_dims(v_scales, axis=-3)
        v_biases = mx.expand_dims(v_biases, axis=-3)

    if k_biases is None:
        scores = mx.quantized_matmul(
            queries,
            k_q,
            k_scales,
            transpose=True,
            group_size=group_size,
            bits=bits,
        )
    else:
        scores = mx.quantized_matmul(
            queries,
            k_q,
            k_scales,
            k_biases,
            transpose=True,
            group_size=group_size,
            bits=bits,
        )

    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores = scores + mask

    scores = mx.softmax(scores, axis=-1, precise=True)
    if v_biases is None:
        out = mx.quantized_matmul(
            scores,
            v_q,
            v_scales,
            transpose=False,
            group_size=group_size,
            bits=bits,
        )
    else:
        out = mx.quantized_matmul(
            scores,
            v_q,
            v_scales,
            v_biases,
            transpose=False,
            group_size=group_size,
            bits=bits,
        )

    if n_repeats > 1:
        out = out.reshape(B, n_q_heads, q_len, head_dim)

    return out


def apply_rotary_emb(x: mx.array, freqs_cis: mx.array, positions: mx.array) -> mx.array:
    rot_dim = freqs_cis.shape[1] * 2
    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]

    d_q = x_rot.shape[-1] // 2
    xq_r = x_rot[..., :d_q]
    xq_i = x_rot[..., d_q:]

    freqs_cos = freqs_cis[positions, :, 0][None, None, :, :]
    freqs_sin = freqs_cis[positions, :, 1][None, None, :, :]

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos

    x_rot_out = mx.stack([xq_out_r, xq_out_i], axis=-1)
    x_rot_out = x_rot_out.reshape(*x_rot_out.shape[:-2], -1)

    return mx.concatenate([x_rot_out, x_pass], axis=-1)


class VisionAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.transpose(0, 2, 3, 1, 4)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(out)


class TextAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        rotary_dim = self.head_dim // 2
        qkv_dim = dim + 2 * (dim * n_kv_heads // n_heads)
        self.qkv = nn.Linear(dim, qkv_dim)
        self.proj = nn.Linear(dim, dim)
        self.tau_wq = mx.zeros((n_heads, qkv_dim))
        self.tau_wv = mx.zeros((n_heads, qkv_dim))
        self.tau_alpha = mx.zeros((n_heads,))

    def __call__(
        self,
        x: mx.array,
        freqs_cis: mx.array,
        positions: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple] = None,
        cache_pos: int = 0,
        kv_quant: bool = False,
    ) -> Tuple[mx.array, Tuple]:
        B, T, C = x.shape
        qkv_out = self.qkv(x)

        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        q, k, v = mx.split(qkv_out, [q_dim, q_dim + kv_dim], axis=-1)

        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        tok_feat = nn.gelu(qkv_out)
        tok_q = mx.tanh(tok_feat @ self.tau_wq.T).transpose(0, 2, 1)
        tok_v = mx.tanh(tok_feat @ self.tau_wv.T).transpose(0, 2, 1)

        pos = positions.astype(mx.float32) + 1
        tau_pos = 1 + (mx.sigmoid(self.tau_alpha[:, None] * mx.log(pos)) - 0.5)

        tau_q = (tok_q + tau_pos[None])[..., None]
        tau_v = (tok_v + tau_pos[None])[..., None]

        q = q * tau_q
        v = v * tau_v

        q = apply_rotary_emb(q, freqs_cis, positions)
        k = apply_rotary_emb(k, freqs_cis, positions)

        if cache is not None:
            if kv_quant:
                k_q_cache, k_scales_cache, k_biases_cache, v_q_cache, v_scales_cache, v_biases_cache = cache

                new_k_q, new_k_scales, new_k_biases, new_v_q, new_v_scales, new_v_biases = _quantize_kv_affine(k, v)

                k_q_cache = k_q_cache.at[:, :, cache_pos : cache_pos + T, :].add(new_k_q)
                k_scales_cache = k_scales_cache.at[:, :, cache_pos : cache_pos + T, :].add(new_k_scales)
                k_biases_cache = k_biases_cache.at[:, :, cache_pos : cache_pos + T, :].add(new_k_biases)
                v_q_cache = v_q_cache.at[:, :, cache_pos : cache_pos + T, :].add(new_v_q)
                v_scales_cache = v_scales_cache.at[:, :, cache_pos : cache_pos + T, :].add(new_v_scales)
                v_biases_cache = v_biases_cache.at[:, :, cache_pos : cache_pos + T, :].add(new_v_biases)

                kv_len = cache_pos + T
                q_keys = (
                    k_q_cache[:, :, :kv_len, :],
                    k_scales_cache[:, :, :kv_len, :],
                    k_biases_cache[:, :, :kv_len, :],
                )
                q_values = (
                    v_q_cache[:, :, :kv_len, :],
                    v_scales_cache[:, :, :kv_len, :],
                    v_biases_cache[:, :, :kv_len, :],
                )

                new_cache = (k_q_cache, k_scales_cache, k_biases_cache, v_q_cache, v_scales_cache, v_biases_cache)
            else:
                k_cache, v_cache = cache
                k_cache = k_cache.at[:, :, cache_pos : cache_pos + T, :].add(k)
                v_cache = v_cache.at[:, :, cache_pos : cache_pos + T, :].add(v)
                k = k_cache[:, :, : cache_pos + T, :]
                v = v_cache[:, :, : cache_pos + T, :]
                new_cache = (k_cache, v_cache)
        else:
            new_cache = (k, v)

        if kv_quant:
            if cache is None:
                k_q, k_scales, k_biases, v_q, v_scales, v_biases = _quantize_kv_affine(k, v)
                q_keys = (k_q, k_scales, k_biases)
                q_values = (v_q, v_scales, v_biases)
                kv_len = k_q.shape[2]

            if mask is not None:
                mask_slice = mask[:, :, :, :kv_len]
            else:
                mask_slice = None

            out = _quantized_scaled_dot_product_attention(
                q,
                q_keys,
                q_values,
                scale=self.scale,
                mask=mask_slice,
                group_size=64,
                bits=4,
            )
        else:
            if self.n_kv_heads != self.n_heads:
                n_rep = self.n_heads // self.n_kv_heads
                k = mx.repeat(k, n_rep, axis=1)
                v = mx.repeat(v, n_rep, axis=1)

            if mask is not None:
                kv_len = k.shape[2]
                mask_slice = mask[:, :, :, :kv_len]
                out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask_slice)
            else:
                out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)

        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(out), new_cache
