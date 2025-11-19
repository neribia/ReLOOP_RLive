from fastapi import FastAPI, HTTPException, Response

from rlive_common.core.response import ResetResponse, StepResponseJSON, StepResponseMultipart
from rlive_common.core.request import ResetRequest, StepRequest
from rlive_world.world import World
from rlive_common.utils import get_logger

logger = get_logger(__name__)


world: World = World()
app: FastAPI = FastAPI(title="World API", version="1.0.0")


@app.post("/reset", response_model=ResetResponse)
async def reset_endpoint(req: ResetRequest) -> ResetResponse:
    try:
        result = await world.reset(req)
        return result
    except Exception as e:  # pragma: no cover (defensive)
        logger.exception("Reset JSON endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step_json", response_model=StepResponseJSON)
async def step_endpoint_json(req: StepRequest) -> StepResponseJSON:
    try:
        result, image = await world.step(req)
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
async def step_endpoint_binary(req: StepRequest) -> Response:
    try:
        result, image = await world.step(req)
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