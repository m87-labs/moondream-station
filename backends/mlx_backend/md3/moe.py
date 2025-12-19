import mlx.core as mx
import mlx.nn as nn

from functools import partial
from typing import Optional, Tuple


def _gather_sort(x: mx.array, indices: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
    *_, M = indices.shape
    indices_flat = indices.flatten()
    order = mx.argsort(indices_flat)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices_flat[order], inv_order


def _scatter_unsort(x: mx.array, inv_order: mx.array, shape: Tuple) -> mx.array:
    x = x[inv_order]
    return mx.unflatten(x, 0, shape)


@partial(mx.compile, shapeless=True)
def _gated_gelu_compiled(h: mx.array, g: mx.array) -> mx.array:
    return nn.gelu(h) * (g + 1)


class MoEMLP(nn.Module):

    def __init__(
        self, dim: int, n_experts: int, expert_dim: int, experts_per_token: int
    ):
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
        topk_idxs = topk_idxs[:, : self.experts_per_token]
        topk_logits = mx.take_along_axis(router_logits, topk_idxs, axis=-1)
        topk_weights = mx.softmax(topk_logits, axis=-1)

        x_expanded = mx.expand_dims(x_flat, (-2, -3))

        do_sort = topk_idxs.size >= 64
        idx = topk_idxs
        inv_order = None
        if do_sort:
            x_expanded, idx, inv_order = _gather_sort(x_expanded, topk_idxs)

        h_full = mx.gather_mm(
            x_expanded,
            self.fc1.swapaxes(-1, -2),
            rhs_indices=idx,
            sorted_indices=do_sort,
        )

        h, g = mx.split(h_full, 2, axis=-1)
        h = _gated_gelu_compiled(h, g)

        expert_outs = mx.gather_mm(
            h,
            self.fc2.swapaxes(-1, -2),
            rhs_indices=idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            expert_outs = _scatter_unsort(expert_outs, inv_order, topk_idxs.shape)

        expert_outs = expert_outs.squeeze(-2)
        weighted_outs = expert_outs * topk_weights[:, :, None]
        mlp_out = weighted_outs.sum(axis=1)

        return mlp_out.reshape(B, T, C)


class QuantizedMoEMLP(nn.Module):
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

        has_biases = mode == "affine"

        fc1_qs, fc1_scales, fc1_biases = [], [], []
        fc2_qs, fc2_scales, fc2_biases = [], [], []

        for i in range(moe.n_experts):
            result = mx.quantize(
                moe.fc1[i], bits=bits, group_size=group_size, mode=mode
            )
            if has_biases:
                w_q, s, b = result
            else:
                w_q, s = result
                b = None
            fc1_qs.append(w_q)
            fc1_scales.append(s)
            if b is not None:
                fc1_biases.append(b)

            result = mx.quantize(
                moe.fc2[i], bits=bits, group_size=group_size, mode=mode
            )
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

        x_expanded = mx.expand_dims(x_flat, (-2, -3))

        do_sort = topk_idxs.size >= 64
        idx = topk_idxs
        inv_order = None
        if do_sort:
            x_expanded, idx, inv_order = _gather_sort(x_expanded, topk_idxs)

        h_full = mx.gather_qmm(
            x_expanded,
            self.fc1_q,
            self.fc1_scales,
            self.fc1_biases,
            rhs_indices=idx,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=do_sort,
        )

        h, g = mx.split(h_full, 2, axis=-1)
        h = _gated_gelu_compiled(h, g)

        expert_outs = mx.gather_qmm(
            h,
            self.fc2_q,
            self.fc2_scales,
            self.fc2_biases,
            rhs_indices=idx,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=do_sort,
        )

        if do_sort:
            expert_outs = _scatter_unsort(expert_outs, inv_order, topk_idxs.shape)

        expert_outs = expert_outs.squeeze(-2)
        weighted_outs = expert_outs * topk_weights[:, :, None]
        mlp_out = weighted_outs.sum(axis=1)

        return mlp_out.reshape(B, T, C)
