from types import SimpleNamespace

from fastapi import FastAPI, HTTPException, Response
from contextlib import asynccontextmanager

from rlive_common.core.response import ResetResponse, StepResponseJSON, StepResponseMultipart
from rlive_common.core.request import ResetRequest, StepRequest
from rlive_world.world import World
from rlive_common.utils import get_logger

logger = get_logger(__name__)

resources = SimpleNamespace()


@asynccontextmanager
async def lifespan(app: FastAPI):
    resources.world = World()

    yield

    resources.world.close()


app: FastAPI = FastAPI(title="World API", version="1.0.0", lifespan=lifespan)


@app.post("/reset", response_model=ResetResponse)
def reset_endpoint(req: ResetRequest) -> ResetResponse:
    try:
        result = resources.world.reset(req)
        return result
    except Exception as e:  # pragma: no cover (defensive)
        logger.exception("Reset JSON endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step_json", response_model=StepResponseJSON)
def step_endpoint_json(req: StepRequest) -> StepResponseJSON:
    try:
        result, image = resources.world.step(req)
        result_json = StepResponseJSON(
            image=image,
            observation=result.observation,
            truncated=result.truncated,
            info=result.info,
        )
        return result_json
    except Exception as e:
        logger.exception("Step JSON endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step_multipart")
def step_endpoint_binary(req: StepRequest) -> Response:
    try:
        result, image = resources.world.step(req)
        model = StepResponseMultipart(
            image=image,
            observation=result.observation,
            truncated=result.truncated,
            info=result.info,
        )
        body, content_type = model.encode()

        return Response(content=body, media_type=content_type)
    except Exception as e:  # pragma: no cover (defensive)
        logger.exception("Step multipart endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))
