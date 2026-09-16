from types import SimpleNamespace

from fastapi import FastAPI, Response, Request, status
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from rlive_common.core.response import ResetResponse, StepResponseJSON, StepResponseMultipart, AttachHardwareResponse, DetachHardwareResponse
from rlive_common.core.request import ResetRequest, StepRequest, AttachHardwareRequest, DetachHardwareRequest
from rlive_world.errors import PermanentHardwareError, TransientHardwareError
from rlive_world.world import World
from rlive_common.utils import get_logger

logger = get_logger(__name__)


resources = SimpleNamespace()


@asynccontextmanager
async def lifespan(app: FastAPI):
    resources.world = World()

    try:
        yield

    # Graceful cleanup
    finally:
        logger.info("Shutting down: releasing hardware...")
        try:
            resources.world.close()
        except Exception:
            logger.exception("Error during shutdown")


app: FastAPI = FastAPI(title="World API", version="1.0.0", lifespan=lifespan)


# Failures that mean "the hardware did not cooperate" rather than "the server is
# broken". They answer with a structured error and a single warning line; anything
# not listed here falls through to the 500 handler, which logs a full traceback and
# is then re-raised by Starlette and logged again by uvicorn.
#
# The status code carries the retry policy, because the client's
# is_retryable_status() only retries 502/503/504:
#
#   503 - retrying may work (a sleeping Bolt the BLE scan missed).
#   422 - retrying is pointless; the request named hardware that is not there.
#
# TimeoutError subclasses OSError, so registering OSError also covers BLE timeouts
# that were never classified. BleakError only exists with the optional `bolt` extra.
# Starlette resolves handlers along the exception's MRO, so the specific
# Transient/Permanent classes win over the RuntimeError fallback they inherit from.
RETRYABLE_ERRORS: list[type[Exception]] = [TransientHardwareError, RuntimeError, OSError]
NON_RETRYABLE_ERRORS: list[type[Exception]] = [PermanentHardwareError]

try:
    from bleak.exc import BleakError
except ImportError:
    pass
else:
    RETRYABLE_ERRORS.append(BleakError)


def _hardware_error_response(
    request: Request,
    exc: Exception,
    status_code: int,
    retryable: bool,
    suggestion: str,
) -> JSONResponse:
    """Build the structured body shared by both hardware handlers."""
    logger.warning(f"{type(exc).__name__} at {request.url.path}: {exc}")

    message = str(exc)
    if "attach_hardware" in message.lower() or "not attached" in message.lower():
        suggestion = "Call attach_hardware() first"

    return JSONResponse(
        status_code=status_code,
        content={
            "error": "HardwareError",
            "message": message,
            "recoverable": retryable,
            "retryable": retryable,
            "suggestion": suggestion,
            "endpoint": request.url.path,
            "method": request.method
        }
    )


async def retryable_hardware_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Answer a transient hardware failure with a retryable 503."""
    return _hardware_error_response(
        request, exc, status.HTTP_503_SERVICE_UNAVAILABLE, True, "Check the hardware and retry"
    )


async def permanent_hardware_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Answer an unsatisfiable hardware request with a 422 the client will not retry."""
    return _hardware_error_response(
        request, exc, status.HTTP_422_UNPROCESSABLE_CONTENT, False, "Fix the hardware or the request, then try again"
    )


for _exc_type in RETRYABLE_ERRORS:
    app.add_exception_handler(_exc_type, retryable_hardware_error_handler)

for _exc_type in NON_RETRYABLE_ERRORS:
    app.add_exception_handler(_exc_type, permanent_hardware_error_handler)


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
