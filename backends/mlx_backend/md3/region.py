import mlx.core as mx
import mlx.nn as nn
import math
from typing import List, Tuple, Union, Dict, Optional

from .config import RegionConfig

SpatialRefs = List[Union[Tuple[float, float], Tuple[float, float, float, float]]]


def fourier_features(x: mx.array, w: mx.array) -> mx.array:
    f = 2 * math.pi * (x @ w)
    return mx.concatenate([mx.cos(f), mx.sin(f)], axis=-1)


class RegionModel(nn.Module):
    def __init__(self, config: RegionConfig):
        super().__init__()
        self.config = config

        self.coord_features = mx.zeros((1, config.coord_feat_dim // 2))
        self.coord_encoder = nn.Linear(config.coord_feat_dim, config.dim)
        self.coord_decoder = nn.Linear(config.dim, config.coord_out_dim)

        self.size_features = mx.zeros((2, config.size_feat_dim // 2))
        self.size_encoder = nn.Linear(config.size_feat_dim, config.dim)
        self.size_decoder = nn.Linear(config.dim, config.size_out_dim)

        self.ln = nn.LayerNorm(config.dim)

    def encode_coordinate(self, coord: mx.array) -> mx.array:
        features = fourier_features(coord, self.coord_features)
        return self.coord_encoder(features)

    def decode_coordinate(self, hidden_state: mx.array) -> mx.array:
        hidden_state = self.ln(hidden_state)
        return self.coord_decoder(hidden_state)

    def encode_size(self, size: mx.array) -> mx.array:
        features = fourier_features(size, self.size_features)
        return self.size_encoder(features)

    def decode_size(self, hidden_state: mx.array) -> mx.array:
        hidden_state = self.ln(hidden_state)
        logits = self.size_decoder(hidden_state)
        return logits.reshape(2, -1)

    def encode_spatial_refs(self, spatial_refs: SpatialRefs) -> Dict[str, Optional[mx.array]]:
        coords = []
        sizes = []

        for ref in spatial_refs:
            if len(ref) == 2:
                coords.append(ref[0])
                coords.append(ref[1])
            else:
                x_c = (ref[0] + ref[2]) / 2
                y_c = (ref[1] + ref[3]) / 2
                width = ref[2] - ref[0]
                height = ref[3] - ref[1]
                coords.append(x_c)
                coords.append(y_c)
                sizes.append([width, height])

        coords_arr = mx.array(coords, dtype=self.coord_features.dtype).reshape(-1, 1)
        encoded_coords = self.encode_coordinate(coords_arr)

        if sizes:
            sizes_arr = mx.array(sizes, dtype=self.size_features.dtype)
            encoded_sizes = self.encode_size(sizes_arr)
        else:
            encoded_sizes = None

        return {"coords": encoded_coords, "sizes": encoded_sizes}


def bin_to_size(bin_idx: mx.array) -> mx.array:
    return mx.power(2.0, (bin_idx.astype(mx.float32) / 1023.0) * 10.0 - 10.0)


def size_to_bin(size: mx.array) -> mx.array:
    size = mx.maximum(size, 1.0 / 1024.0)
    return ((mx.log2(size) + 10.0) / 10.0 * 1023.0).astype(mx.int32)
