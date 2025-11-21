from typing import Dict, Any, Tuple

import asyncio
import numpy as np

from rlive_common.core.response import Response, ResetResponse
from rlive_common.core.request import ResetRequest, StepRequest
from rlive_common.utils import get_logger

logger = get_logger(__name__)


class World:
    """
    A minimal World class.
    """

    def __init__(self) -> None:
        self.camera = None
        self.bolt = None
        self._setup_world()

    def _setup_world(self):
        pass
        # camera_cfg = CameraConfig()
        # self.camera = CameraService(camera_cfg)
        # self.bolt = Bolt()

    def _make_observation(self) -> np.ndarray:
        """
        Make observation vector.
        """
        logger.debug(f"Making observation")
        return np.zeros(3, dtype=np.float32)  # (480, 640))

    def reset(self, req: ResetRequest) -> ResetResponse:
        """
        Reset the world and return an initial observation.
        """
        logger.info(f"Resetting the world.")
        async with self._lock:
            obs = await self._make_observation()
            info: Dict[str, Any] = {"msg": "reset", "status": "ok"}
            return ResetResponse(observation=obs, info=info)

        obs = self._make_observation()
        info: Dict[str, Any] = {"msg": "reset", "status": "ok"}
        return ResetResponse(observation=obs, info=info)

    def step(self, req: StepRequest) -> Tuple[Response, np.ndarray]:
        """
        Make a step in the world with a given action.
        """
        action = req.action
        logger.info(f"Make a step in the world with action {action}")

        obs = self._make_observation()

        info: Dict[str, Any] = {"status": "ok"}
        image = np.random.randint(0, 255, size=(2, 4, 3), dtype=np.uint8)

        return Response(
            observation=obs,
            truncated=False,
            info=info,
        ), image

    def close(self) -> None:
        """
        Closes the world and disconnects still open connections
        """
        pass