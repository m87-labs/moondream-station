"""LoRA utilities for the MLX Moondream backend.

LoRA is applied only to text MLPs (dense + MoE). This mirrors Kestrel's
behavior and intentionally does NOT touch attention, embeddings, or lm_head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import re

import mlx.core as mx
import numpy as np

from .config import TextConfig


@dataclass
class DenseLoRALayer:
    up_a: mx.array
    up_b: mx.array
    down_a: mx.array
    down_b: mx.array


@dataclass
class MoELoRALayer:
    up_a: mx.array
    up_b: mx.array
    down_a: mx.array
    down_b: mx.array


class TextLoRA:
    """Container for text LoRA weights (dense + MoE)."""

    def __init__(
        self,
        text_config: TextConfig,
        *,
        rank: int,
        max_rank: int,
        dtype: mx.Dtype,
        adapter_id: Optional[str] = None,
    ) -> None:
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if max_rank < rank:
            raise ValueError("max_rank must be >= rank")

        self.text_config = text_config
        self.rank = rank
        self.max_rank = max_rank
        self.adapter_id = adapter_id

        moe_cfg = text_config.moe
        self.start_layer = moe_cfg.start_layer if moe_cfg else text_config.n_layers

        if moe_cfg is not None:
            self.rank_per_expert = rank // moe_cfg.experts_per_token
            if self.rank_per_expert < 1:
                raise ValueError(
                    f"rank ({rank}) must be >= experts_per_token ({moe_cfg.experts_per_token})"
                )
            self.max_rank_per_expert = max_rank // moe_cfg.experts_per_token
            if self.max_rank_per_expert < 1:
                raise ValueError(
                    f"max_rank ({max_rank}) must be >= experts_per_token ({moe_cfg.experts_per_token})"
                )
        else:
            self.rank_per_expert = 0
            self.max_rank_per_expert = 0

        d_model = text_config.dim
        d_ffn = text_config.ff_dim

        # Dense layers: [0, start_layer)
        self.dense: list[DenseLoRALayer] = []
        for _ in range(self.start_layer):
            self.dense.append(
                DenseLoRALayer(
                    up_a=mx.zeros((max_rank, d_model), dtype=dtype),
                    up_b=mx.zeros((d_ffn, max_rank), dtype=dtype),
                    down_a=mx.zeros((max_rank, d_ffn), dtype=dtype),
                    down_b=mx.zeros((d_model, max_rank), dtype=dtype),
                )
            )

        # MoE layers: [start_layer, n_layers)
        self.moe: list[MoELoRALayer] = []
        if moe_cfg is not None:
            num_experts = moe_cfg.num_experts
            d_expert = moe_cfg.expert_inner_dim
            for _ in range(text_config.n_layers - self.start_layer):
                self.moe.append(
                    MoELoRALayer(
                        up_a=mx.zeros((num_experts, self.max_rank_per_expert, d_model), dtype=dtype),
                        up_b=mx.zeros((num_experts, d_expert * 2, self.max_rank_per_expert), dtype=dtype),
                        down_a=mx.zeros((num_experts, self.max_rank_per_expert, d_expert), dtype=dtype),
                        down_b=mx.zeros((num_experts, d_model, self.max_rank_per_expert), dtype=dtype),
                    )
                )

    def dense_layer(self, layer_idx: int) -> Optional[DenseLoRALayer]:
        if layer_idx < len(self.dense):
            return self.dense[layer_idx]
        return None

    def moe_layer(self, layer_idx: int) -> Optional[MoELoRALayer]:
        moe_idx = layer_idx - self.start_layer
        if 0 <= moe_idx < len(self.moe):
            return self.moe[moe_idx]
        return None

    @staticmethod
    def _to_mx(value: Any, *, dtype: Optional[mx.Dtype] = None) -> mx.array:
        if isinstance(value, mx.array):
            arr = value
        elif hasattr(value, "detach"):
            # torch.Tensor
            tensor = value.detach().cpu()
            # numpy doesn't support bfloat16; upcast to float32
            if str(getattr(tensor, "dtype", "")) == "torch.bfloat16":
                tensor = tensor.float()
            arr = tensor.numpy()
        elif isinstance(value, np.ndarray):
            arr = value
        elif hasattr(value, "numpy"):
            arr = value.numpy()
        else:
            arr = np.array(value)
        return mx.array(arr, dtype=dtype) if dtype is not None else mx.array(arr)

    @staticmethod
    def _pad_axis(arr: mx.array, target: int, axis: int) -> mx.array:
        if arr.shape[axis] == target:
            return arr
        if arr.shape[axis] > target:
            raise ValueError(f"LoRA tensor rank {arr.shape[axis]} exceeds max {target}")
        pad_shape = list(arr.shape)
        pad_shape[axis] = target - arr.shape[axis]
        pad = mx.zeros(pad_shape, dtype=arr.dtype)
        return mx.concatenate([arr, pad], axis=axis)

    @classmethod
    def detect_rank(cls, state_dict: Dict[str, Any], text_config: TextConfig) -> int:
        # Dense keys
        for key, tensor in state_dict.items():
            if "dense" in key and "up_a" in key:
                shape = tensor.shape
                return int(shape[0])
        # MoE keys
        for key, tensor in state_dict.items():
            if "moe" in key and "up_a" in key:
                shape = tensor.shape
                rank_per_expert = int(shape[1])
                moe_cfg = text_config.moe
                if moe_cfg:
                    return rank_per_expert * moe_cfg.experts_per_token
                return rank_per_expert
        raise ValueError("Could not detect LoRA rank from state dict")

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, Any],
        *,
        text_config: TextConfig,
        max_rank: int,
        dtype: mx.Dtype,
        adapter_id: Optional[str] = None,
    ) -> "TextLoRA":
        rank = cls.detect_rank(state_dict, text_config)
        if rank > max_rank:
            raise ValueError(f"Adapter rank ({rank}) exceeds max_rank ({max_rank})")

        lora = cls(text_config, rank=rank, max_rank=max_rank, dtype=dtype, adapter_id=adapter_id)

        dense_seen = set()
        moe_seen = set()

        for key, tensor in state_dict.items():
            # Normalize prefix: take last occurrence of dense./moe.
            match = re.search(r"(dense|moe)\.(\d+)\.(up_a|up_b|down_a|down_b)$", key)
            if not match:
                continue
            kind, idx_str, name = match.group(1), match.group(2), match.group(3)
            idx = int(idx_str)
            arr = cls._to_mx(tensor, dtype=dtype)

            if kind == "dense":
                if idx >= len(lora.dense):
                    raise ValueError(f"Dense LoRA layer index {idx} out of range")
                layer = lora.dense[idx]
                if name in ("up_a", "down_a"):
                    arr = cls._pad_axis(arr, lora.max_rank, axis=0)
                else:
                    arr = cls._pad_axis(arr, lora.max_rank, axis=1)
                setattr(layer, name, arr)
                dense_seen.add((idx, name))
            else:
                if idx >= len(lora.moe):
                    raise ValueError(f"MoE LoRA layer index {idx} out of range")
                layer = lora.moe[idx]
                if name in ("up_a", "down_a"):
                    arr = cls._pad_axis(arr, lora.max_rank_per_expert, axis=1)
                else:
                    arr = cls._pad_axis(arr, lora.max_rank_per_expert, axis=2)
                setattr(layer, name, arr)
                moe_seen.add((idx, name))

        # Basic completeness checks
        for layer_idx in range(len(lora.dense)):
            for name in ("up_a", "up_b", "down_a", "down_b"):
                if (layer_idx, name) not in dense_seen:
                    raise ValueError(f"Adapter missing dense LoRA for layer {layer_idx} ({name})")
        for layer_idx in range(len(lora.moe)):
            for name in ("up_a", "up_b", "down_a", "down_b"):
                if (layer_idx, name) not in moe_seen:
                    raise ValueError(f"Adapter missing MoE LoRA for layer {layer_idx} ({name})")

        # Materialize arrays on device
        arrays = []
        for layer in lora.dense:
            arrays.extend([layer.up_a, layer.up_b, layer.down_a, layer.down_b])
        for layer in lora.moe:
            arrays.extend([layer.up_a, layer.up_b, layer.down_a, layer.down_b])
        if arrays:
            mx.eval(*arrays)

        return lora
