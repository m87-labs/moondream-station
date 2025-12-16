"""
MLX backend for Moondream Station.

Provides Moondream 3 inference via MLX on Apple Silicon.
Requires macOS with Apple Silicon (M1/M2/M3/M4).
"""

from __future__ import annotations

import base64
import gc
import io
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
from PIL import Image

_backend_dir = Path(__file__).parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

logger = logging.getLogger(__name__)

_model = None
_model_commit_hash = None
_initialized = False
_quantize_mode = None

MODEL_ID = "moondream/moondream3-preview"


def _extract_commit_hash(snapshot_path: Path) -> str:
    return snapshot_path.name


def init_backend(**kwargs: Any) -> None:
    global _initialized, _quantize_mode
    _quantize_mode = kwargs.get("quantize", None)
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


def _load_weights(weights_path: Path):
    """Load weights from HuggingFace safetensors and remap to MLX model structure."""
    index_path = weights_path / "model.safetensors.index.json"
    with open(index_path, "r") as f:
        index = json.load(f)

    all_weights = {}
    shards = set(index["weight_map"].values())
    for shard in shards:
        shard_path = weights_path / shard
        weights = mx.load(str(shard_path))
        all_weights.update(weights)

    remapped = {}
    for name, weight in all_weights.items():
        new_name = _remap_weight_name(name)
        if new_name is not None:
            remapped[new_name] = weight

    return remapped


def _get_model():
    """Get the model, checking for updates from HuggingFace."""
    global _model, _model_commit_hash

    from huggingface_hub import snapshot_download
    from md3 import Moondream

    logger.debug(f"Checking for model updates: {MODEL_ID}")
    weights_path = None

    try:
        weights_path = Path(snapshot_download(MODEL_ID, token=True))
    except Exception:
        pass

    if weights_path is None:
        try:
            weights_path = Path(snapshot_download(MODEL_ID, local_files_only=True))
            logger.info("Using cached model (could not check for updates)")
        except Exception as e:
            raise RuntimeError(
                f"Could not load model {MODEL_ID}. "
                f"Either login to HuggingFace with 'huggingface-cli login' "
                f"or ensure the model is already cached. Error: {e}"
            )

    new_commit_hash = _extract_commit_hash(weights_path)

    if _model is None:
        logger.info(f"Loading model (commit: {new_commit_hash[:8]}...)")
    elif new_commit_hash != _model_commit_hash:
        logger.info(f"New model version detected: {new_commit_hash[:8]}...")
        _model = None
        mx.clear_cache()
    else:
        return _model

    config = _load_config(weights_path)
    model = Moondream(config)
    weights = _load_weights(weights_path)
    model.load_weights(list(weights.items()), strict=False)

    # Delete weights dict to drop extra reference to bf16 arrays
    # The model's parameters now hold the only references
    del weights

    mx.eval(model.parameters())
    mx.synchronize()

    # Apply quantization if specified
    if _quantize_mode:
        logger.info(f"Quantizing MoE experts with mode: {_quantize_mode}")
        model.quantize_experts(mode=_quantize_mode)
        mx.eval(model.parameters())
        mx.clear_cache()  # Clear MLX internal caches
        gc.collect()  # Force Python GC to free old MoEMLP objects
        logger.info("Quantization complete")

    _model = model
    _model_commit_hash = new_commit_hash
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
        return model.caption(image, length=length, stream=stream, settings=settings)
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
        return model.query(image, question, reasoning=reasoning, stream=stream, settings=settings)
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
        result = model.detect(image, target_obj, settings=settings)
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
        result = model.point(image, target_obj, settings=settings)
        points = result.get("points", [])
        return {"points": points, "count": len(points)}
    except Exception as e:
        logger.exception("Point failed")
        return {"error": str(e)}
