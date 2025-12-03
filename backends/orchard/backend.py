"""
Orchard backend for Moondream Station.

Provides Moondream 3 inference via Orchard's optimized Apple Silicon runtime.
Requires macOS with Apple Silicon (M1/M2/M3/M4).

https://github.com/TheProxyCompany/orchard-py
"""

from __future__ import annotations

import base64
import io
import logging
from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:
    from orchard.clients.moondream import MoondreamClient
    from orchard.engine.inference_engine import InferenceEngine

logger = logging.getLogger(__name__)

_engine: InferenceEngine | None = None
_client: MoondreamClient | None = None


def init_backend(**kwargs: Any) -> None:
    """
    Initialize the Orchard inference engine and Moondream client.

    This launches the PIE (Proxy Inference Engine) binary if not already running,
    establishes IPC connections, and prepares the Moondream 3 model for inference.

    Args:
        **kwargs: Ignored. Orchard handles model configuration internally.
    """
    global _engine, _client

    from orchard.clients.moondream import MoondreamClient
    from orchard.engine.inference_engine import InferenceEngine

    if _engine is not None and _client is not None:
        logger.debug("Orchard backend already initialized")
        return

    logger.info("Initializing Orchard backend")
    _engine = InferenceEngine()
    _client = _engine.client(MoondreamClient.model_id)

    if not isinstance(_client, MoondreamClient):
        raise RuntimeError(
            f"Expected MoondreamClient, got {type(_client).__name__}"
        )

    logger.info("Orchard backend initialized successfully")


def _get_client() -> MoondreamClient:
    """Get the initialized client, raising if not ready."""
    if _client is None:
        raise RuntimeError("Backend not initialized. Call init_backend() first.")
    return _client


def _load_image(image_url: str) -> Image.Image:
    """
    Convert a base64 data URL to a PIL Image.

    Args:
        image_url: Base64-encoded image as data URL (data:image/...;base64,...)
                   or raw base64 string.

    Returns:
        PIL Image in RGB mode.
    """
    if image_url.startswith("data:image"):
        _, encoded = image_url.split(",", 1)
    else:
        encoded = image_url

    raw_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


def caption(
    image_url: str | None = None,
    length: str = "normal",
    stream: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Generate a caption for an image.

    Args:
        image_url: Base64-encoded image data URL.
        length: Caption length - "normal", "short", or "long".
        stream: If True, returns a generator yielding caption chunks.
        **kwargs: Additional parameters (temperature, etc.)

    Returns:
        {"caption": str} or {"caption": generator} if streaming.
        {"error": str} on failure.
    """
    if not image_url:
        return {"error": "image_url is required"}

    try:
        client = _get_client()
        image = _load_image(image_url)
        return client.caption(image, length=length, stream=stream, **kwargs)
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
    """
    Answer a question about an image.

    Args:
        image_url: Base64-encoded image data URL.
        question: The question to answer.
        stream: Ignored. Orchard streams internally and aggregates.
        reasoning: If True, includes reasoning trace with grounding data.
        **kwargs: Additional parameters (temperature, etc.)

    Returns:
        {"answer": str} or {"answer": str, "reasoning": {...}} if reasoning=True.
        {"error": str} on failure.
    """
    if not image_url or not question:
        return {"error": "image_url and question are required"}

    try:
        client = _get_client()
        image = _load_image(image_url)
        return client.query(prompt=question, image=image, reasoning=reasoning, **kwargs)
    except Exception as e:
        logger.exception("Query failed")
        return {"error": str(e)}


def detect(
    image_url: str | None = None,
    object: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Detect objects in an image with bounding boxes.

    Args:
        image_url: Base64-encoded image data URL.
        object: Object class to detect (e.g., "dog", "car").
        **kwargs: Additional parameters (temperature, etc.)

    Returns:
        {"objects": [{"x_min", "y_min", "x_max", "y_max"}, ...]}
        {"error": str} on failure.
    """
    if not image_url or not object:
        return {"error": "image_url and object are required"}

    try:
        client = _get_client()
        image = _load_image(image_url)
        return client.detect(image, object=object, **kwargs)
    except Exception as e:
        logger.exception("Detect failed")
        return {"error": str(e)}


def point(
    image_url: str | None = None,
    object: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Find points where an object appears in an image.

    Args:
        image_url: Base64-encoded image data URL.
        object: Object to locate (e.g., "eye", "face").
        **kwargs: Additional parameters (temperature, etc.)

    Returns:
        {"points": [{"x": float, "y": float}, ...], "count": int}
        {"error": str} on failure.
    """
    if not image_url or not object:
        return {"error": "image_url and object are required"}

    try:
        client = _get_client()
        image = _load_image(image_url)
        result = client.point(image, object=object, **kwargs)
        points = result.get("points", [])
        return {"points": points, "count": len(points)}
    except Exception as e:
        logger.exception("Point failed")
        return {"error": str(e)}
