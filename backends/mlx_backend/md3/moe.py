import mlx.core as mx
import mlx.nn as nn
from typing import Optional


class MoEMLP(nn.Module):
    def __init__(self, dim: int, n_experts: int, expert_dim: int, experts_per_token: int):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.expert_dim = expert_dim
        self.experts_per_token = experts_per_token

        self.router = nn.Linear(dim, n_experts)
        self.fc1 = mx.zeros((n_experts, 2 * expert_dim, dim))
        self.fc2 = mx.zeros((n_experts, dim, expert_dim))

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)

        router_logits = self.router(x_flat)
        topk_idxs = mx.argpartition(-router_logits, self.experts_per_token, axis=-1)
        topk_idxs = topk_idxs[:, :self.experts_per_token]
        topk_logits = mx.take_along_axis(router_logits, topk_idxs, axis=-1)
        topk_weights = mx.softmax(topk_logits, axis=-1)

        if T == 1:
            return self._forward_single_token(x_flat, topk_idxs, topk_weights, B, T, C)
        else:
            return self._forward_multi_token(x_flat, topk_idxs, topk_weights, B, T, C)

    def _forward_single_token(self, x_flat, topk_idxs, topk_weights, B, T, C):
        num_tokens = x_flat.shape[0]
        top_k = self.experts_per_token

        flat_idxs = topk_idxs.reshape(-1)
        flat_weights = topk_weights.reshape(-1)

        w1_selected = self.fc1[flat_idxs]
        w2_selected = self.fc2[flat_idxs]

        x_expanded = mx.broadcast_to(x_flat[:, None, :], (num_tokens, top_k, C))
        x_expanded = x_expanded.reshape(-1, C, 1)

        x1_full = mx.matmul(w1_selected, x_expanded).squeeze(-1)
        h, g = mx.split(x1_full, 2, axis=-1)
        h = nn.gelu(h) * (g + 1)

        expert_outs = mx.matmul(w2_selected, h[:, :, None]).squeeze(-1)
        weighted_outs = expert_outs * flat_weights[:, None]
        weighted_outs = weighted_outs.reshape(num_tokens, top_k, C)
        mlp_out = weighted_outs.sum(axis=1)

        return mlp_out.reshape(B, T, C)

    def _forward_multi_token(self, x_flat, topk_idxs, topk_weights, B, T, C):
        num_tokens = x_flat.shape[0]
        out = mx.zeros((num_tokens, C))

        for expert_idx in range(self.n_experts):
            expert_mask = topk_idxs == expert_idx
            expert_weights = mx.where(expert_mask, topk_weights, mx.zeros_like(topk_weights))
            token_weights = expert_weights.sum(axis=-1)

            w1 = self.fc1[expert_idx]
            w2 = self.fc2[expert_idx]

            h_full = x_flat @ w1.T
            h, g = mx.split(h_full, 2, axis=-1)
            h = nn.gelu(h) * (g + 1)
            expert_out = h @ w2.T

            out = out + expert_out * token_weights[:, None]

        return out.reshape(B, T, C)


class QuantizedMoEMLP(nn.Module):
    """MoE with quantized expert weights."""

    def __init__(
        self,
        dim: int,
        n_experts: int,
        expert_dim: int,
        experts_per_token: int,
        bits: int = 4,
        group_size: int = 64,
        mode: str = "affine",
    ):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.expert_dim = expert_dim
        self.experts_per_token = experts_per_token
        self.bits = bits
        self.group_size = group_size
        self.mode = mode

        self.router = nn.Linear(dim, n_experts)
        # Quantized weights (set by from_float)
        self.fc1_q: Optional[mx.array] = None
        self.fc1_scales: Optional[mx.array] = None
        self.fc1_biases: Optional[mx.array] = None
        self.fc2_q: Optional[mx.array] = None
        self.fc2_scales: Optional[mx.array] = None
        self.fc2_biases: Optional[mx.array] = None

    @classmethod
    def from_float(
        cls,
        moe: MoEMLP,
        bits: int = 4,
        group_size: int = 64,
        mode: str = "affine",
    ) -> "QuantizedMoEMLP":
        """Convert a float MoE to quantized."""
        q = cls(
            moe.dim,
            moe.n_experts,
            moe.expert_dim,
            moe.experts_per_token,
            bits,
            group_size,
            mode,
        )
        q.router = moe.router

        # Quantize each expert's fc1 and fc2
        # Note: affine mode returns (w_q, scales, biases)
        # mxfp4/mxfp8/nvfp4 modes return (w_q, scales) only
        has_biases = mode == "affine"

        fc1_qs, fc1_scales, fc1_biases = [], [], []
        fc2_qs, fc2_scales, fc2_biases = [], [], []

        for i in range(moe.n_experts):
            result = mx.quantize(moe.fc1[i], bits=bits, group_size=group_size, mode=mode)
            if has_biases:
                w_q, s, b = result
            else:
                w_q, s = result
                b = None
            fc1_qs.append(w_q)
            fc1_scales.append(s)
            if b is not None:
                fc1_biases.append(b)

            result = mx.quantize(moe.fc2[i], bits=bits, group_size=group_size, mode=mode)
            if has_biases:
                w_q, s, b = result
            else:
                w_q, s = result
                b = None
            fc2_qs.append(w_q)
            fc2_scales.append(s)
            if b is not None:
                fc2_biases.append(b)

        q.fc1_q = mx.stack(fc1_qs)
        q.fc1_scales = mx.stack(fc1_scales)
        q.fc1_biases = mx.stack(fc1_biases) if fc1_biases else None
        q.fc2_q = mx.stack(fc2_qs)
        q.fc2_scales = mx.stack(fc2_scales)
        q.fc2_biases = mx.stack(fc2_biases) if fc2_biases else None

        # CRITICAL: Evaluate quantized weights immediately to break reference to bf16 weights
        # Without this, the lazy computation graph keeps bf16 weights alive
        arrays_to_eval = [q.fc1_q, q.fc1_scales, q.fc2_q, q.fc2_scales]
        if q.fc1_biases is not None:
            arrays_to_eval.extend([q.fc1_biases, q.fc2_biases])
        mx.eval(*arrays_to_eval)

        return q

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)

        router_logits = self.router(x_flat)
        topk_idxs = mx.argpartition(-router_logits, self.experts_per_token, axis=-1)
        topk_idxs = topk_idxs[:, : self.experts_per_token]
        topk_logits = mx.take_along_axis(router_logits, topk_idxs, axis=-1)
        topk_weights = mx.softmax(topk_logits, axis=-1)

        if T == 1:
            return self._forward_single_token(x_flat, topk_idxs, topk_weights, B, T, C)
        else:
            return self._forward_multi_token(x_flat, topk_idxs, topk_weights, B, T, C)

    def _forward_single_token(self, x_flat, topk_idxs, topk_weights, B, T, C):
        num_tokens = x_flat.shape[0]
        top_k = self.experts_per_token

        flat_idxs = topk_idxs.reshape(-1)
        flat_weights = topk_weights.reshape(-1)

        # Select quantized weights for chosen experts
        fc1_q_sel = self.fc1_q[flat_idxs]
        fc1_scales_sel = self.fc1_scales[flat_idxs]
        fc1_biases_sel = self.fc1_biases[flat_idxs] if self.fc1_biases is not None else None
        fc2_q_sel = self.fc2_q[flat_idxs]
        fc2_scales_sel = self.fc2_scales[flat_idxs]
        fc2_biases_sel = self.fc2_biases[flat_idxs] if self.fc2_biases is not None else None

        # Expand input: (num_tokens, C) -> (num_tokens * top_k, C)
        x_expanded = mx.broadcast_to(x_flat[:, None, :], (num_tokens, top_k, C))
        x_expanded = x_expanded.reshape(-1, C)

        # fc1: (num_tokens * top_k, C) @ (2 * expert_dim, C).T -> (num_tokens * top_k, 2 * expert_dim)
        # Use loop since quantized_matmul doesn't support batched weights
        x1_results = []
        for i in range(num_tokens * top_k):
            x1 = mx.quantized_matmul(
                x_expanded[i : i + 1],
                fc1_q_sel[i],
                fc1_scales_sel[i],
                fc1_biases_sel[i] if fc1_biases_sel is not None else None,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )
            x1_results.append(x1)
        x1_full = mx.concatenate(x1_results, axis=0)

        h, g = mx.split(x1_full, 2, axis=-1)
        h = nn.gelu(h) * (g + 1)

        # fc2: (num_tokens * top_k, expert_dim) @ (C, expert_dim).T -> (num_tokens * top_k, C)
        expert_results = []
        for i in range(num_tokens * top_k):
            out = mx.quantized_matmul(
                h[i : i + 1],
                fc2_q_sel[i],
                fc2_scales_sel[i],
                fc2_biases_sel[i] if fc2_biases_sel is not None else None,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )
            expert_results.append(out)
        expert_outs = mx.concatenate(expert_results, axis=0)

        weighted_outs = expert_outs * flat_weights[:, None]
        weighted_outs = weighted_outs.reshape(num_tokens, top_k, C)
        mlp_out = weighted_outs.sum(axis=1)

        return mlp_out.reshape(B, T, C)

    def _forward_multi_token(self, x_flat, topk_idxs, topk_weights, B, T, C):
        num_tokens = x_flat.shape[0]
        out = mx.zeros((num_tokens, C))

        for expert_idx in range(self.n_experts):
            expert_mask = topk_idxs == expert_idx
            expert_weights = mx.where(expert_mask, topk_weights, mx.zeros_like(topk_weights))
            token_weights = expert_weights.sum(axis=-1)

            # fc1: x @ w1.T using quantized_matmul
            fc1_bias = self.fc1_biases[expert_idx] if self.fc1_biases is not None else None
            h_full = mx.quantized_matmul(
                x_flat,
                self.fc1_q[expert_idx],
                self.fc1_scales[expert_idx],
                fc1_bias,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )
            h, g = mx.split(h_full, 2, axis=-1)
            h = nn.gelu(h) * (g + 1)

            # fc2: h @ w2.T using quantized_matmul
            fc2_bias = self.fc2_biases[expert_idx] if self.fc2_biases is not None else None
            expert_out = mx.quantized_matmul(
                h,
                self.fc2_q[expert_idx],
                self.fc2_scales[expert_idx],
                fc2_bias,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )

            out = out + expert_out * token_weights[:, None]

        return out.reshape(B, T, C)
