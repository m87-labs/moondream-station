import mlx.core as mx
import mlx.nn as nn
from typing import Tuple

from .config import VisionConfig
from .attention import VisionAttention
from .image_crops import reconstruct_from_crops


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu_approx(self.fc1(x)))


class VisionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = VisionAttention(dim, n_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, ff_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class VisionEncoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config

        patch_dim = config.enc_patch_size * config.enc_patch_size * config.in_channels
        self.patch_emb = nn.Linear(patch_dim, config.enc_dim)

        grid_size = config.crop_size // config.enc_patch_size
        num_patches = grid_size * grid_size
        self.pos_emb = mx.zeros((1, num_patches, config.enc_dim))

        self.blocks = [
            VisionBlock(config.enc_dim, config.enc_n_heads, config.enc_ff_dim)
            for _ in range(config.enc_n_layers)
        ]
        self.post_ln = nn.LayerNorm(config.enc_dim)

        self.proj_fc1 = nn.Linear(config.enc_dim * 2, config.proj_inner_dim)
        self.proj_fc2 = nn.Linear(config.proj_inner_dim, config.proj_out_dim)

    def create_patches(self, x: mx.array) -> mx.array:
        B, C, H, W = x.shape
        P = self.config.enc_patch_size
        x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.transpose(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, (H // P) * (W // P), C * P * P)
        return x

    def encode_crops(self, crops: mx.array) -> mx.array:
        x = self.create_patches(crops)
        x = self.patch_emb(x)
        x = x + self.pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.post_ln(x)
        return x

    def project(self, global_features: mx.array, reconstructed: mx.array) -> mx.array:
        final_features = mx.concatenate([global_features, reconstructed], axis=-1)
        x = nn.gelu_approx(self.proj_fc1(final_features))
        x = self.proj_fc2(x)
        return x

    def __call__(self, crops: mx.array, tiling: Tuple[int, int]) -> mx.array:
        encoded = self.encode_crops(crops)
        global_features = encoded[0]
        local_features = encoded[1:]

        grid_size = self.config.crop_size // self.config.enc_patch_size
        local_features = local_features.reshape(-1, grid_size, grid_size, self.config.enc_dim)

        reconstructed = reconstruct_from_crops(
            local_features, tiling, patch_size=1, overlap_margin=self.config.overlap_margin
        )

        reconstructed = reconstructed.transpose(2, 0, 1)
        target_h = target_w = grid_size
        H, W = reconstructed.shape[1], reconstructed.shape[2]

        pool_h = H // target_h
        pool_w = W // target_w
        reconstructed = reconstructed[:, :pool_h * target_h, :pool_w * target_w]
        reconstructed = reconstructed.reshape(self.config.enc_dim, target_h, pool_h, target_w, pool_w)
        reconstructed = reconstructed.mean(axis=(2, 4))
        reconstructed = reconstructed.transpose(1, 2, 0).reshape(grid_size * grid_size, self.config.enc_dim)

        return self.project(global_features, reconstructed)
