import time
import warnings
import logging
import json
import asyncio
import uvicorn
import argparse
import os
import sys

from typing import Generator
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.background import BackgroundTask

from hypervisor import Hypervisor
from display_utils import RUNNING
from misc import get_app_dir


def configure_logging() -> logging.Logger:
    app_dir = get_app_dir()

    logger = logging.getLogger("hypervisor")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Debug logs
    log_file = os.path.join(app_dir, "hypervisor.log")
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Only show warnings and above to console
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = configure_logging()

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


async def lifespan(app: FastAPI):
    # Initialize the hypervisor
    app.state.hypervisor = Hypervisor(manifest_url=app.state.manifest_url)
    app.state.hypervisor.boot()
    logger.info("Moondream Hypervisor server initialized and ready")
    yield
    # Cleanup on shutdown
    logger.info("Moondream Hypervisor server shutting down")
    try:
        app.state.hypervisor.shutdown()
    except Exception as e:
        logger.error(f"Error during hypervisor shutdown: {e}")
        # Continue with shutdown process even if there was an error


app = FastAPI(
    title="Moondream Hypervisor",
    description="Main server for Moondream. Manages Inference, GUI, and Logging",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware to log request timing
class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        logger.info(f"Request received: {request.method} {request.url.path}")
        response = await call_next(request)

        if isinstance(response, StreamingResponse):

            def log_stream_time():
                elapsed_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"Streaming response completed in {elapsed_ms:.2f} ms: {request.method} {request.url.path}"
                )

            response.background = BackgroundTask(log_stream_time)
        else:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Response completed in {elapsed_ms:.2f} ms: {request.method} {request.url.path}"
            )

        return response


app.add_middleware(TimingMiddleware)


def get_hypervisor(request: Request) -> Hypervisor:
    """Get the InferenceVisor instance from app state."""
    inferencevisor = request.app.state.hypervisor
    if not inferencevisor:
        raise HTTPException(status_code=500, detail="Hypervisor not initialized")
    return inferencevisor


def sse_format_generator(generator):
    """Format a generator as Server-Sent Events."""
    for item in generator:
        try:
            # If the item is already JSON, just add the data: prefix
            json_obj = json.loads(item)
            yield f"data: {json.dumps(json_obj)}\n\n"
        except json.JSONDecodeError:
            # If it's not valid JSON, wrap it in a chunk object
            yield f"data: {json.dumps({'chunk': item})}\n\n"
    yield f"data: {json.dumps({'completed': True})}\n\n"


async def proxy_inference_request(
    request: Request, endpoint: str, hypervisor: Hypervisor
):
    """Generic handler that proxies requests to the inference server."""
    request_data = await request.json()
    stream = request_data.get("stream", False)

    if stream:
        generator = hypervisor.inferencevisor.proxy_request(
            endpoint, request_data, stream=True
        )
        return StreamingResponse(
            sse_format_generator(generator), media_type="text/event-stream"
        )
    else:
        result = hypervisor.inferencevisor.proxy_request(
            endpoint, request_data, stream=False
        )

        if isinstance(result, Generator):
            # If we accidentally got a generator for a non-streaming request,
            # consume it and get the last item which should be the complete response
            last_item = None
            for item in result:
                last_item = item
            if last_item:
                try:
                    result = json.loads(last_item)
                except json.JSONDecodeError:
                    result = {"error": "Failed to parse response from inference server"}
            else:
                result = {"error": "No response from inference server"}

        if "error" in result:
            raise HTTPException(
                status_code=result.get("status_code", 500), detail=result["error"]
            )

        return JSONResponse(result)


# -------------------- Inference --------------------
@app.post("/v1/caption", summary="Generate a caption for an image")
async def caption_endpoint(
    request: Request,
    hypervisor: Hypervisor = Depends(get_hypervisor),
):
    hypervisor.posthog_capture("caption")
    return await proxy_inference_request(request, "caption", hypervisor)


@app.post("/v1/query", summary="Answer a visual query about an image")
async def query_endpoint(
    request: Request,
    hypervisor: Hypervisor = Depends(get_hypervisor),
):
    hypervisor.posthog_capture("query")
    return await proxy_inference_request(request, "query", hypervisor)


@app.post("/v1/detect", summary="Detect objects in an image")
async def detect_endpoint(
    request: Request,
    hypervisor: Hypervisor = Depends(get_hypervisor),
):
    hypervisor.posthog_capture("detect")
    return await proxy_inference_request(request, "detect", hypervisor)


@app.post("/v1/point", summary="Find points corresponding to an object in an image")
async def point_endpoint(
    request: Request,
    hypervisor: Hypervisor = Depends(get_hypervisor),
):
    hypervisor.posthog_capture("point")
    return await proxy_inference_request(request, "point", hypervisor)


@app.get("/v1/health", summary="Health check for the hypervisor server")
async def health_check(
    hypervisor: Hypervisor = Depends(get_hypervisor),
):
    """Check the health of all components."""
    health_result = hypervisor.check_health()
    return health_result


# -------------------- Admin --------------------
@app.post("/admin/set_model", summary="Set inference server's active model")
async def set_model(
    request: Request,
    hypervisor: Hypervisor = Depends(get_hypervisor),
):
    body = await request.json()
    model = body.get("model", None)
    confirm = body.get("confirm", False)

    if not confirm:
        raise HTTPException(
            status_code=428,
            detail=(
                "Changing active model requires confirmation. "
                "Resubmit the same request with 'confirm': true."
            ),
        )

    result = hypervisor.inferencevisor.set_model(model)
    hypervisor.posthog_capture("set_model", {"model": model})
    if result["status"] != 200:
        raise HTTPException(
            status_code=result["status"],
            detail=result["message"],
        )
    return result["message"]


@app.post("/admin/update_manifest", summary="Set inference server's active model")
async def update_manifest(
    request: Request,
    hypervisor: Hypervisor = Depends(get_hypervisor),
):
    print("in hypervisor server update_manifest")
    hypervisor.manifest.update()
    return hypervisor.manifest.notes


@app.get("/admin/get_models", summary="Set inference server's active model")
async def get_models(
    hypervisor: Hypervisor = Depends(get_hypervisor),
):
    return hypervisor.manifest.models


@app.get("/admin/get_inference_client", summary="Set inference server's active model")
async def get_inference_client(
    hypervisor: Hypervisor = Depends(get_hypervisor),
):
    return hypervisor.manifest.inference_clients


@app.get("/admin/status", summary="Return Hypervisor status")
async def get_status(hypervisor: Hypervisor = Depends(get_hypervisor)):
    return {
        "hypervisor": hypervisor.status,
        "inference": hypervisor.inferencevisor.status,
    }


@app.post("/admin/update_hypervisor", summary="Update Hypervisor to latest version")
async def update_hypervisor(
    request: Request,
    hypervisor: Hypervisor = Depends(get_hypervisor),
):
    body = await request.json()
    confirm = body.get("confirm", False)
    if not confirm:
        raise HTTPException(
            status_code=428,
            detail=(
                "Updating Hypervisor requires confirmation. "
                "Resubmit the same request with 'confirm': true."
            ),
        )

    hypervisor.posthog_capture("update_hypervisor")
    hypervisor.update_hypervisor()


@app.post("/admin/update_bootstrap", summary="Update Bootstrap to latest version")
async def update_bootstrap(
    request: Request,
    hypervisor: Hypervisor = Depends(get_hypervisor),
):
    body = await request.json()
    confirm = body.get("confirm", False)
    if not confirm:
        raise HTTPException(
            status_code=428,
            detail=(
                "Updating Bootstrap requires confirmation. "
                "Resubmit the same request with 'confirm': true."
            ),
        )

    hypervisor.posthog_capture("update_bootstrap")
    return hypervisor.update_bootstrap()


@app.post("/admin/update_cli", summary="Update CLI to latest version")
async def update_cli(
    request: Request,
    hypervisor: Hypervisor = Depends(get_hypervisor),
):
    body = await request.json()
    confirm = body.get("confirm", False)
    if not confirm:
        raise HTTPException(
            status_code=428,
            detail=(
                "Updating CLI requires confirmation. "
                "Resubmit the same request with 'confirm': true."
            ),
        )

    hypervisor.posthog_capture("update_cli")
    hypervisor.clivisor.update()


@app.get(
    "/admin/check_updates",
    summary="Check for new versions of the Bootstrap, Hypervisor, Inference, Models, and CLI",
)
async def update_bootstrap(
    hypervisor: Hypervisor = Depends(get_hypervisor),
):
    return hypervisor.check_all_for_updates()


# TODO: FINISH
@app.post(
    "/admin/reset",
    summary="Delete all app data and update Bootstrap to the latest version.",
)
async def reset(request: Request, hypervisor: Hypervisor = Depends(get_hypervisor)):
    body = await request.json()
    confirm = body.get("confirm", False)
    if not confirm:
        raise HTTPException(
            status_code=428,
            detail=(
                "Reset requires confirmation. "
                "Resubmit the same request with 'confirm': true."
            ),
        )

    return hypervisor.reset()


@app.post(
    "/admin/toggle_metric_reports", summary="Enable/Disable Posthog metic reporting"
)
async def toggle_metric_reports(
    request: Request, hypervisor: Hypervisor = Depends(get_hypervisor)
):
    body = await request.json()
    confirm = body.get("confirm", False)
    if not confirm:
        raise HTTPException(
            status_code=428,
            detail=(
                "Toggle metrics requires confirmation."
                "Resubmit the same request with 'confirm': true."
            ),
        )

    return hypervisor.toggle_posthog_capture()


@app.get("/config", summary="Get server configuration")
async def get_config(
    hypervisor: Hypervisor = Depends(get_hypervisor),
):
    hypervisor.config.load()
    return hypervisor.config.data


@app.post("/config/inference_url", summary="Set the inference server URL")
async def set_inference_url(
    request: Request,
    hypervisor: Hypervisor = Depends(get_hypervisor),
):
    """Set the URL for the inference server."""
    body = await request.json()
    url = body.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="Missing 'url' in request body")
    return hypervisor.inferencevisor.set_inference_url(url)


@app.post("/shutdown", summary="Shutdown the hypervisor server")
async def shutdown_server(
    request: Request,
    hypervisor: Hypervisor = Depends(get_hypervisor),
):
    """Initiate a clean shutdown of the hypervisor server and all its components."""
    logger.info("Shutdown requested via API")

    # Start shutdown in a background task so we can return a response first
    async def shutdown_background():
        # Give time for the response to be sent
        await asyncio.sleep(1)
        # Perform hypervisor shutdown which no longer calls sys.exit()
        hypervisor.shutdown()
        # Get the server that's running this app
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()
        # Signal to uvicorn to stop gracefully
        os._exit(0)

    # Schedule the background task
    asyncio.create_task(shutdown_background())

    return {"status": "ok", "message": "Shutdown initiated"}


def main():
    parser = argparse.ArgumentParser(description="Run Hypervisor Server")
    parser.add_argument(
        "--port", type=int, default=2020, help="Port to run the hypervisor server on"
    )
    parser.add_argument(
        "--inference-url",
        type=str,
        default="http://localhost:20200/v1",
        help="URL of the inference server",
    )
    
    parser.add_argument(
    "--manifest-url",
    type=str,
    default="https://depot.moondream.ai/station/md_station_manifest_ubuntu.json",
    help="Custom manifest URL (defaults to Moondream's official manifest)"
    )

    args = parser.parse_args()

    app.state.inference_url = args.inference_url
    app.state.manifest_url = args.manifest_url 

    logger.info(f"Starting hypervisor server on port: {args.port}")
    logger.info(f"Using inference server at: {args.inference_url}")
    logger.info(f"Parsed manifest URL from args: {args.manifest_url}")

    print(f"Moondream Station is running on http://localhost:{args.port}/v1")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
