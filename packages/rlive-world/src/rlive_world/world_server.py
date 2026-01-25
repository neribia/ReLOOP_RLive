from types import SimpleNamespace

from fastapi import FastAPI, Response, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from rlive_common.core.response import ResetResponse, StepResponseJSON, StepResponseMultipart, AttachHardwareResponse, DetachHardwareResponse
from rlive_common.core.request import ResetRequest, StepRequest, AttachHardwareRequest, DetachHardwareRequest
from rlive_world.world import World
from rlive_common.utils import get_logger

logger = get_logger(__name__)


resources = SimpleNamespace()


@asynccontextmanager
async def lifespan(app: FastAPI):
    resources.world = World()

    yield

    # Graceful cleanup
    try:
        if resources.world._hardware_attached:
            logger.info("Shutting down: detaching hardware...")
            resources.world.detach_hardware(DetachHardwareRequest())
    except Exception:
        logger.exception("Error during shutdown")
    finally:
        resources.world.close()


app: FastAPI = FastAPI(title="World API", version="1.0.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server exception")
    return JSONResponse(
        status_code=500,
        content={"error": "InternalServerError",
                 "message": str(exc),
                 "endpoint": request.url.path,
                 "method": request.method
                 }
    )


@app.get("/health")
def health_check():
    """Health check endpoint to verify server is running."""
    return {
        "status": "healthy",
        "hardware_attached": resources.world._hardware_attached if hasattr(resources, "world") else False
    }


@app.get("/status")
def get_status():
    """Get current world status including hardware state."""
    if not hasattr(resources, "world"):
        return {"error": "World not initialized"}

    return {
        "hardware_attached": resources.world._hardware_attached,
        "camera_active": resources.world.camera is not None and resources.world.camera.is_running,
        "robot_connected": resources.world.robot is not None and resources.world.robot.api is not None
    }


@app.post("/attach_hardware", response_model=AttachHardwareResponse)
def attach_hardware(req: AttachHardwareRequest) -> AttachHardwareResponse:
    response = resources.world.attach_hardware(req)
    return response


@app.post("/detach_hardware", response_model=DetachHardwareResponse)
def detach_hardware(req: DetachHardwareRequest) -> DetachHardwareResponse:
    response = resources.world.detach_hardware(req)
    return response


@app.post("/reset", response_model=ResetResponse)
def reset_endpoint(req: ResetRequest) -> ResetResponse:
    result = resources.world.reset(req)
    return result


@app.post("/step_json", response_model=StepResponseJSON)
def step_endpoint_json(req: StepRequest) -> StepResponseJSON:
    result = resources.world.step(req)
    result_json = StepResponseJSON(
        observation=result.observation,
        truncated=result.truncated,
        info=result.info,
    )
    return result_json



@app.post("/step_multipart")
def step_endpoint_binary(req: StepRequest) -> Response:
    result = resources.world.step(req)
    model = StepResponseMultipart(
        observation=result.observation,
        truncated=result.truncated,
        info=result.info,
    )
    body, content_type = model.encode()

    return Response(content=body, media_type=content_type)
