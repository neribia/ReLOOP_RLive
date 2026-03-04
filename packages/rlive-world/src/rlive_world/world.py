from typing import Any

import numpy as np

from rlive_world.camera import CameraService, CameraConfig as WorldCameraConfig
from rlive_world.bolt import SpheroBoltPlus
from rlive_world.bolt.boltdummys import DummySpheroEduAPI, DummyFinder
from rlive_world.config import config as cfg
from rlive_common.core.response import BaseResponse, ResetResponse, AttachHardwareResponse, DetachHardwareResponse
from rlive_common.core.request import ResetRequest, StepRequest, AttachHardwareRequest, DetachHardwareRequest
from rlive_common.core import WorldConfig
from rlive_common.utils import get_logger

logger = get_logger(__name__)


class World:
    """
    A minimal World class.
    """

    def __init__(self) -> None:
        self.camera = None
        self.robot = None
        self._hardware_attached = False
        self._setup_world()

    def attach_hardware(self, request: AttachHardwareRequest) -> AttachHardwareResponse:
        logger.debug("Attaching hardware to the world.")
        logger.info(f"request: {request}")

        # Check if already attached
        if self._hardware_attached:
            logger.warning("Hardware already attached, skipping.")
            return AttachHardwareResponse(
                success=True,
                info={"status": "already_attached", "msg": "Hardware was already attached"}
            )

        # Extract config
        world_cfg = request.world_config or WorldConfig()

        # Apply defaults for camera config
        camera_defaults = {
            'type': cfg.CAMERA_TYPE,
            'id': cfg.CAMERA_ID,
            'resolution': cfg.CAMERA_RESOLUTION,  # Now a tuple (width, height)
            'exposure_time_ms': cfg.CAMERA_EXPOSURE_TIME_MS,
        }
        camera_resolved = world_cfg._apply_camera_defaults(camera_defaults)

        # Apply defaults for bolt config
        bolt_defaults = {
            'name': cfg.SPHEROBOLTPLUS_NAME,
            'scanning_time': cfg.SPHEROBOLTPLUS_SCANNING_TIME,
            'color_r': cfg.SPHEROBOLTPLUS_DISPLAY_COLOR_R,
            'color_g': cfg.SPHEROBOLTPLUS_DISPLAY_COLOR_G,
            'color_b': cfg.SPHEROBOLTPLUS_DISPLAY_COLOR_B,
            'use_dummy': cfg.USE_DUMMY_HARDWARE,
        }
        bolt_resolved = world_cfg._apply_bolt_defaults(bolt_defaults)

        logger.debug(f"Resolved camera config: type={camera_resolved['type']}, id={camera_resolved['id']}, resolution={camera_resolved['width']}x{camera_resolved['height']}, exposure={camera_resolved['exposure_time_ms']}ms")
        logger.debug(f"Resolved bolt config: name={bolt_resolved['name']}, scanning_time={bolt_resolved['scanning_time']}s, color=({bolt_resolved['color_r']},{bolt_resolved['color_g']},{bolt_resolved['color_b']}), use_dummy={bolt_resolved['use_dummy']}")

        # Setup Robot/Bolt
        if bolt_resolved['use_dummy']:
            logger.info("Setting up dummy hardware...")
            self.robot = SpheroBoltPlus(scanner_class=DummyFinder, api_class=DummySpheroEduAPI)
            self.robot.connect(bolt_name="DummyBolt", timeout=0.01)
            self.camera = CameraService(WorldCameraConfig(
                type="webcam",
                id=camera_resolved['id'],
                width=camera_resolved['width'],
                height=camera_resolved['height'],
                exposure_time_ms=camera_resolved['exposure_time_ms']
            ))
            self.camera.setup()
        else:
            logger.info(f"Setting up real hardware: bolt_name={bolt_resolved['name']}...")
            self.robot = SpheroBoltPlus()
            self.robot.connect(bolt_name=bolt_resolved['name'], timeout=bolt_resolved['scanning_time'])
            self.camera = CameraService(
                WorldCameraConfig(
                    type=camera_resolved['type'],
                    id=camera_resolved['id'],
                    width=camera_resolved['width'],
                    height=camera_resolved['height'],
                    exposure_time_ms=camera_resolved['exposure_time_ms']
                )
            )
            self.camera.setup()

        self._hardware_attached = True
        logger.info("Hardware successfully attached.")

        return AttachHardwareResponse(success=True, info={"status": "ok", "msg": ""})

    def detach_hardware(self, request: DetachHardwareRequest) -> DetachHardwareResponse:
        logger.debug("Detaching hardware from the world.")
        logger.info(f"request: {request}")

        if not self._hardware_attached:
            logger.debug("Hardware not attached, skipping detach.")
            return DetachHardwareResponse(
                success=True,
                info={"status": "not_attached", "msg": "Hardware was not attached"}
            )

        if self.camera:
            self.camera.release()
            self.camera = None

        if self.robot:
            self.robot.disconnect()
            self.robot = None

        self._hardware_attached = False
        logger.info("Hardware successfully detached.")

        return DetachHardwareResponse(success=True, info={"status": "ok", "msg": ""})

    def _setup_world(self):
        # TODO: implement world setup from reset()
        pass

    def _move_robot(self, action: np.ndarray) -> None:
        """Move the robot according to the given action."""
        logger.debug(f"Moving robot with action: {action}")
        heading = int(action[0])
        speed = int(action[1])
        duration = float(action[2])

        if self.robot is None:
            raise RuntimeError("Robot not initialized.")

        self.robot.move(heading=heading, speed=speed, duration=duration)


    def _make_observation(self) -> np.ndarray:
        """Make observation vector."""
        logger.debug("Making observation")

        if not self._hardware_attached:
            raise RuntimeError("Hardware not attached. Call attach_hardware() first.")

        if self.camera is None:
            raise RuntimeError("Camera not initialized.")

        # Simulate asynchronous observation gathering
        image = self.camera.get_image()
        return image

    def reset(self, req: ResetRequest) -> ResetResponse:
        """Reset the world and return an initial observation."""
        logger.info("Resetting the world.")

        if not self._hardware_attached:
            raise RuntimeError("Hardware not attached. Call attach_hardware() first.")

        for action in req.actions:
            self._move_robot(action)

        obs = self._make_observation()
        info: dict[str, Any] = {"msg": "reset", "status": "ok"}
        return ResetResponse(observation=obs, info=info)

    def step(self, req: StepRequest) -> BaseResponse:
        """Make a step in the world with a given action."""
        action = req.action
        logger.info(f"Make a step in the world with action {action}")

        if not self._hardware_attached:
            raise RuntimeError("Hardware not attached. Call attach_hardware() first.")

        self._move_robot(action)

        obs = self._make_observation()

        info: dict[str, Any] = {"status": "ok"}

        return BaseResponse(
            observation=obs,
            truncated=False,
            info=info,
        )

    def close(self) -> None:
        """Placeholder/stub: would close the world and disconnect still open connections."""
        logger.debug("Closing the world.")
        pass
