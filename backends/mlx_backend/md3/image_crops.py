import math
import numpy as np
import mlx.core as mx
from typing import TypedDict
from PIL import Image


def select_tiling(height: int, width: int, crop_size: int, max_crops: int) -> tuple[int, int]:
    if height <= crop_size or width <= crop_size:
        return (1, 1)

    min_h = math.ceil(height / crop_size)
    min_w = math.ceil(width / crop_size)

    if min_h * min_w > max_crops:
        ratio = math.sqrt(max_crops / (min_h * min_w))
        return (max(1, math.floor(min_h * ratio)), max(1, math.floor(min_w * ratio)))

    h_tiles = math.floor(math.sqrt(max_crops * height / width))
    w_tiles = math.floor(math.sqrt(max_crops * width / height))

    h_tiles = max(h_tiles, min_h)
    w_tiles = max(w_tiles, min_w)

    if h_tiles * w_tiles > max_crops:
        if w_tiles > h_tiles:
            w_tiles = math.floor(max_crops / h_tiles)
        else:
            h_tiles = math.floor(max_crops / w_tiles)

    return (max(1, h_tiles), max(1, w_tiles))


class OverlapCropOutput(TypedDict):
    crops: np.ndarray
    tiling: tuple[int, int]


def overlap_crop_image(
    image: np.ndarray,
    overlap_margin: int,
    max_crops: int,
    base_size: tuple[int, int] = (378, 378),
    patch_size: int = 14,
) -> OverlapCropOutput:
    original_h, original_w = image.shape[:2]
    margin_pixels = patch_size * overlap_margin
    total_margin_pixels = margin_pixels * 2

    crop_patches = base_size[0] // patch_size
    crop_window_patches = crop_patches - (2 * overlap_margin)
    crop_window_size = crop_window_patches * patch_size

    tiling = select_tiling(
        original_h - total_margin_pixels,
        original_w - total_margin_pixels,
        crop_window_size,
        max_crops,
    )

    n_crops = tiling[0] * tiling[1] + 1
    crops = np.zeros((n_crops, base_size[0], base_size[1], image.shape[2]), dtype=np.uint8)

    target_size = (
        tiling[0] * crop_window_size + total_margin_pixels,
        tiling[1] * crop_window_size + total_margin_pixels,
    )

    pil_img = Image.fromarray(image)
    resized = pil_img.resize(
        (int(target_size[1]), int(target_size[0])),
        resample=Image.Resampling.LANCZOS,
    )
    image = np.asarray(resized)

    global_pil = pil_img.resize(
        (int(base_size[1]), int(base_size[0])), resample=Image.Resampling.LANCZOS
    )
    crops[0] = np.asarray(global_pil)

    for i in range(tiling[0]):
        for j in range(tiling[1]):
            y0 = i * crop_window_size
            x0 = j * crop_window_size

            y_end = min(y0 + base_size[0], image.shape[0])
            x_end = min(x0 + base_size[1], image.shape[1])

            crop_region = image[y0:y_end, x0:x_end]
            crops[1 + i * tiling[1] + j, : crop_region.shape[0], : crop_region.shape[1]] = crop_region

    return {"crops": crops, "tiling": tiling}


def reconstruct_from_crops(
    crops: mx.array,
    tiling: tuple[int, int],
    overlap_margin: int,
    patch_size: int = 1,
) -> mx.array:
    tiling_h, tiling_w = tiling
    crop_height, crop_width = crops.shape[1], crops.shape[2]
    margin_pixels = overlap_margin * patch_size

    rows = []
    for tile_y in range(tiling_h):
        row_pieces = []
        for tile_x in range(tiling_w):
            i = tile_y * tiling_w + tile_x
            crop = crops[i]

            x_start = 0 if tile_x == 0 else margin_pixels
            x_end = crop_width if tile_x == tiling_w - 1 else crop_width - margin_pixels
            y_start = 0 if tile_y == 0 else margin_pixels
            y_end = crop_height if tile_y == tiling_h - 1 else crop_height - margin_pixels

            piece = crop[y_start:y_end, x_start:x_end]
            row_pieces.append(piece)

        row = mx.concatenate(row_pieces, axis=1)
        rows.append(row)

    return mx.concatenate(rows, axis=0)


def prepare_crops(
    image: Image.Image,
    crop_size: int = 378,
    max_crops: int = 12,
    overlap_margin: int = 4,
) -> tuple[mx.array, tuple[int, int]]:
    np_image = np.array(image.convert("RGB"))
    overlap_crops = overlap_crop_image(np_image, max_crops=max_crops, overlap_margin=overlap_margin)
    all_crops = overlap_crops["crops"]

    all_crops = np.transpose(all_crops, (0, 3, 1, 2))
    all_crops = all_crops.astype(np.float32) / 255.0
    all_crops = (all_crops - 0.5) / 0.5

    return mx.array(all_crops, dtype=mx.bfloat16), overlap_crops["tiling"]
