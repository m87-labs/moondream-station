import io
import time
import warnings
import logging
import json
import sys
import os
import base64

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from model_service import ModelService
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.background import BackgroundTask

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger("moondream2")
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format="%(message)s")

logging.getLogger("uvicorn").setLevel(logging.ERROR)
logging.getLogger("pyvips").setLevel(logging.ERROR)


VERSION = "v0.0.2"


async def lifespan(app: FastAPI):
    model_name = getattr(app.state, "model_id", "vikhyatk/moondream2")
    revision = getattr(app.state, "revision", None)
    app.state.model_service = ModelService(model_name, revision)
    logger.info("Model initialized successfully.")
    logger.info("Moondream Server startup complete.")
    yield


app = FastAPI(
    title="Moondream Inference Server",
    version=VERSION,
    lifespan=lifespan,
)


# Middleware to log total time of requests
class MiddlewareLogging(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        logger.info(f"New request: {request.url.path}")
        response = await call_next(request)

        if isinstance(response, StreamingResponse):
            # Attach a background task to log the total streaming time after the response is finished.
            def log_stream_time():
                elapsed_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"Completed streaming {request.url.path} in {elapsed_ms:.2f} ms"
                )

            response.background = BackgroundTask(log_stream_time)
            return response
        else:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"Completed {request.url.path} in {elapsed_ms:.2f} ms")
            return response


app.add_middleware(MiddlewareLogging)


def get_model_service(request: Request) -> ModelService:
    """Retrieve the model service instance stored in app.state."""
    model_service = request.app.state.model_service
    if not model_service:
        raise HTTPException(status_code=500, detail="Model service not initialized")
    return model_service


def load_image(file: UploadFile) -> Image.Image:
    """Reads an uploaded file and converts it into a PIL Image."""
    try:
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return image
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=f"Error reading image: {str(e)}")
    finally:
        file.file.close()


def load_base64_image(image_url: str) -> Image.Image:
    """Decodes a base64-encoded image and returns a PIL image."""
    if image_url.startswith("data:image"):
        _, encoded = image_url.split(",", 1)
    else:
        encoded = image_url
    try:
        raw_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return image
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")


def sse_event_generator(raw_generator):
    for token in raw_generator:
        yield f"data: {json.dumps({'chunk': token})}\n\n"
    yield f"data: {json.dumps({'completed': True})}\n\n"


def process_inference(image: Image.Image, inference_func, **kwargs) -> dict:
    """Applies the given inference function to a PIL image."""
    try:
        return inference_func(image, **kwargs)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


def process_inference_stream(key: str, image: Image.Image, inference_func, **kwargs):
    """Applies the given inference function to a PIL image in streaming mode."""
    try:
        result = inference_func(image, **kwargs)
        raw_generator = result[key]
        return sse_event_generator(raw_generator)
    except Exception as e:
        logger.error(f"Inference error (streaming): {e}")
        raise HTTPException(
            status_code=500, detail=f"Inference error (streaming): {str(e)}"
        )


@app.post("/v1/caption", summary="Generate a caption for an image")
async def caption_endpoint(
    request: Request,
    length: str = Form(None, description="Caption length: 'short' or 'normal'"),
    init_image: UploadFile = File(None, description="Input image file"),
    model_service: ModelService = Depends(get_model_service),
):
    content_type = request.headers.get("content-type", "")
    variant = None
    if "application/json" in content_type:
        body = await request.json()
        image_url = body.get("image_url")
        length = body.get("length", "normal")
        stream = body.get("stream", False)
        settings = body.get("settings", {})
        variant = body.get("variant")
        if not image_url:
            raise HTTPException(status_code=400, detail="Missing 'image_url' in JSON.")
        image = load_base64_image(image_url)
    else:
        if not init_image:
            raise HTTPException(
                status_code=400,
                detail="For multipart form-data, 'init_image' must be provided.",
            )
        if not length:
            raise HTTPException(
                status_code=400,
                detail="For multipart form-data, 'length' must be provided.",
            )
        image = load_image(init_image)
        stream = False
        settings = {}

    if stream:
        event_generator = process_inference_stream(
            "caption",
            image,
            model_service.caption,
            length=length,
            stream=True,
            settings=settings,
            variant=variant,
        )
        return StreamingResponse(event_generator, media_type="text/event-stream")
    else:
        result = process_inference(
            image,
            model_service.caption,
            length=length,
            settings=settings,
            variant=variant,
        )
        return JSONResponse({"caption": result["caption"], "request_id": 0})


@app.post("/v1/query", summary="Answer a visual query about an image")
async def query_endpoint(
    request: Request,
    question: str = Form(None, description="The visual query question"),
    init_image: UploadFile = File(None, description="Input image file"),
    model_service: ModelService = Depends(get_model_service),
):
    content_type = request.headers.get("content-type", "")
    variant = None
    if "application/json" in content_type:
        body = await request.json()
        image_url = body.get("image_url")
        question = body.get("question", "")
        stream = body.get("stream", False)
        settings = body.get("settings", {})
        variant = body.get("variant")
        reasoning = body.get("reasoning", False)
        if not image_url or not question:
            raise HTTPException(
                status_code=400,
                detail="Both 'image_url' and 'question' must be present in JSON.",
            )
        image = load_base64_image(image_url)
    else:
        if not init_image:
            raise HTTPException(
                status_code=400,
                detail="For multipart/form-data, 'init_image' must be provided.",
            )
        if not question:
            raise HTTPException(
                status_code=400,
                detail="For multipart/form-data, 'question' must be provided.",
            )
        image = load_image(init_image)
    if stream:
        event_generator = process_inference_stream(
            "answer",
            image,
            model_service.query,
            question=question,
            stream=True,
            settings=settings,
            variant=variant,
        )
        return StreamingResponse(event_generator, media_type="text/event-stream")
    else:
        result = process_inference(
            image,
            model_service.query,
            question=question,
            variant=variant,
            reasoning=reasoning,
        )
        resp = {"answer": result["answer"], "request_id": 0}
        if result.get("reasoning"):
            resp["reasoning"] = result.get("reasoning")
        return JSONResponse(resp)


@app.post("/v1/detect", summary="Detect objects in an image")
async def detect_endpoint(
    request: Request,
    obj: str = Form(None, description="The object to detect (e.g., 'face')"),
    init_image: UploadFile = File(None, description="Input image file"),
    model_service: ModelService = Depends(get_model_service),
):
    content_type = request.headers.get("content-type", "")
    variant = None

    if "application/json" in content_type:
        body = await request.json()
        image_url = body.get("image_url")
        obj = body.get("object", "")
        variant = body.get("variant")
        if not image_url or not obj:
            raise HTTPException(
                status_code=400,
                detail="Both 'image_url' and 'object' must be present in JSON.",
            )
        image = load_base64_image(image_url)
        result = process_inference(
            image, model_service.detect, obj=obj, variant=variant
        )
        obj = result.get("objects", [])
        return JSONResponse({"objects": obj, "request_id": 0})
    else:
        if not init_image:
            raise HTTPException(
                status_code=400,
                detail="For multipart/form-data, 'init_image' must be provided.",
            )
        if not obj:
            raise HTTPException(
                status_code=400,
                detail="For multipart/form-data, 'object' must be provided.",
            )
        image = load_image(init_image)

    result = process_inference(image, model_service.detect, obj=obj)
    obj = result.get("objects", [])
    return JSONResponse({"objects": obj, "request_id": 0})


@app.post("/v1/point", summary="Find points corresponding to an object in an image")
async def point_endpoint(
    request: Request,
    obj: str = Form(None, description="The object to point at (e.g., 'person')"),
    init_image: UploadFile = File(None, description="Input image file"),
    model_service: ModelService = Depends(get_model_service),
):
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        body = await request.json()
        image_url = body.get("image_url")
        obj = body.get("object", "")
        variant = body.get("variant")
        if not image_url or not obj:
            raise HTTPException(
                status_code=400,
                detail="Both 'image_url' and 'object' must be present in JSON.",
            )
        image = load_base64_image(image_url)
        result = process_inference(image, model_service.point, obj=obj, variant=variant)
        points = result.get("points", [])
        return JSONResponse({"points": points, "count": len(points)})
    else:
        if not init_image:
            raise HTTPException(
                status_code=400,
                detail="For multipart/form-data, 'init_image' must be provided.",
            )
        if not obj:
            raise HTTPException(
                status_code=400,
                detail="For multipart/form-data, 'object' must be provided.",
            )
        image = load_image(init_image)

    result = process_inference(image, model_service.point, obj=obj)
    points = result.get("points", [])
    return JSONResponse({"points": points, "count": len(points)})


@app.get("/v1/health", summary="Health check endpoint")
def health():
    return {"status": "ok"}


@app.get("/v1/version", summary="Health check endpoint")
def health(model_service: ModelService = Depends(get_model_service)):
    return {
        "inference_server_version": VERSION,
        "model_revision": model_service.revision,
    }


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="Run Moondream Server")
    parser.add_argument(
        "--port", type=int, default=20200, help="Port to run the server on"
    )
    parser.add_argument(
        "--revision", type=str, default=None, help="Moondream revision to use"
    )
    parser.add_argument(
        "--model-id", type=str, default=None, help="Moondream model ID to use"
    )
    args, _ = parser.parse_known_args()

    if args.model_id:
        app.state.model_id = args.model_id

    if args.revision:
        app.state.revision = args.revision

    logger.info(f"Starting server on port: {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="error")
