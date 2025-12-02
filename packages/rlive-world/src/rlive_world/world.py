from typing import Any

import asyncio
import numpy as np

from rlive_world.camera import CameraService, CameraConfig
from rlive_world.bolt.sphero_bolt_plus import SpheroBoltPlus  # FIXME: make daccessible over bolt(__init__)
from rlive_world.bolt.boltdummys import DummySpheroEduAPI, DummyFinder
from rlive_common.core.response import BaseResponse, ResetResponse, AttachHardwareResponse, DetachHardwareResponse
from rlive_common.core.request import ResetRequest, StepRequest, AttachHardwareRequest, DetachHardwareRequest
from rlive_common.utils import get_logger

logger = get_logger(__name__)


async def fake_scan(timeout: int = 5):
    await asyncio.sleep(timeout)  # simulate a 5-second BLE scan


class World:
    """A minimal World class.
    """

    def __init__(self) -> None:
        self.camera = None
        self.robot = None
        self._setup_world()

    def attach_hardware(self, request: AttachHardwareRequest) -> AttachHardwareResponse:
        logger.debug("Attaching hardware to the world.")
        logger.info(f"request: {request}")

        # Simulate asynchronous hardware scanning
        asyncio.run(fake_scan())

        # Setup Camera
        self.camera = CameraService(CameraConfig())
        self.camera.setup()

        # Setup Robot/Bolt
        self.robot = SpheroBoltPlus(api_class=DummySpheroEduAPI, scanner=DummyFinder())
        self.robot.connect("DummyBolt")

        return AttachHardwareResponse(success=True, info={"status": "ok", "msg": ""})

    def detach_hardware(self, request: DetachHardwareRequest) -> DetachHardwareResponse:
        logger.debug("Detaching hardware from the world.")
        logger.info(f"request: {request}")
        if self.camera:
            self.camera.release()
            self.camera = None

        if self.robot:
            self.robot.disconnect()
            self.robot = None

        return DetachHardwareResponse(success=True, info={"status": "ok", "msg": ""})

    def _setup_world(self):
        # TODO: implement world setup from reset()
        pass


    def _make_observation(self) -> np.ndarray:
        """Make observation vector."""
        logger.debug("Making observation")

        # Simulate asynchronous observation gathering
        asyncio.run(fake_scan(timeout=1))
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image = self.camera.get_image()  # FIXME: refactor CameraService class to get_image()
        return image

    def reset(self, req: ResetRequest) -> ResetResponse:
        """Reset the world and return an initial observation."""
        logger.info("Resetting the world.")

        obs = self._make_observation()
        info: dict[str, Any] = {"msg": "reset", "status": "ok"}
        return ResetResponse(observation=obs, info=info)

    def step(self, req: StepRequest) -> BaseResponse:
        """Make a step in the world with a given action."""
        action = req.action
        logger.info(f"Make a step in the world with action {action}")

        self.robot.move(action)

        obs = self._make_observation()

        info: dict[str, Any] = {"status": "ok"}

        return BaseResponse(
            observation=obs,
            truncated=False,
            info=info,
        )

    def close(self) -> None:
        """Close the world and disconnect any open connections."""
        logger.debug("Closing the world.")
