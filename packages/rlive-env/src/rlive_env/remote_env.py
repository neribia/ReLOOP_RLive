from typing import Any

import numpy as np
import cv2 as cv
import gymnasium as gym

from rlive_env.config import config as cfg
from rlive_env.world_client import WorldInterface
from rlive_common.core.response import ResetResponse, StepResponseJSON, StepResponseMultipart
from rlive_common.utils import get_logger

logger = get_logger(__name__)


class RemoteWorldEnv(gym.Env):
    """Gymnasium-compatible environment that communicates with a remote World server over HTTP.
    """

    metadata = {"render_modes": ["opencv"]}

    def __init__(self, max_episode_steps: int | None = 100, render_mode: str | None = None, auto_attach: bool = True, options: dict[str, Any] | None = None,
                 **kwargs) -> None:
        """Initialize the environment.

        Attributes:
            - max_episodes (int | None): Max epochs (default: 100)
            - render_mode (Optional[str]): Rendering mode ('opencv' or None)
            - auto_attach (bool): Automatically attach hardware on init (default: True)
            - base_url (Optional[str]): Base URL for remote environment.
            - timeout (Optional[float]): Time in seconds to wait for the server to send data
        """
        super().__init__()

        self._max_episode_steps = max_episode_steps
        self._episode = 0  # Start from 0 or 1? Other Env's as reference.
        self.render_mode = render_mode
        self.iface: WorldInterface | None = None
        self.obs = None
        self._hardware_attached = False
        self.options = options or {}

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(360, start=-179)  # placeholder (one valid action)
        # Goal variables
        self.goal_position = (320, 240)

        self._connect(auto_attach=auto_attach, **kwargs)

    def _connect(self, auto_attach: bool = True, **kwargs) -> None:
        """Create iface and optionally attach hardware."""
        logger.info("Setting up interface to RemoteWorld.")
        self.iface = WorldInterface(**kwargs)

        if auto_attach:
            self.attach_hardware()

    def attach_hardware(self) -> Any:
        """Explicitly attach hardware to the world server."""
        if self._hardware_attached:
            logger.warning("Hardware already attached, skipping.")
            return None

        logger.info("Attaching hardware in RemoteWorld.")
        resp = self.iface.attach_hardware(**self.options)
        if not resp.success:
            raise RuntimeError(f"Failed to attach hardware: {resp.info}")

        self._hardware_attached = True
        logger.info("Hardware successfully attached.")
        return resp

    def detach_hardware(self):
        """Explicitly detach hardware from the world server."""
        if not self._hardware_attached:
            logger.debug("Hardware not attached, skipping detach.")
            return

        try:
            resp = self.iface.detach_hardware()
            self._hardware_attached = False
            logger.info("Hardware successfully detached.")
            return resp
        except Exception:
            logger.exception("Failed to detach hardware (ignored).")
            self._hardware_attached = False

    def _disconnect(self):
        """Detach hardware and close HTTP interface (robust gegen Fehler)."""
        iface = getattr(self, "iface", None)
        if iface is None:
            return

        logger.info("Closing connection to RemoteWorld.")

        # Detach hardware if attached
        self.detach_hardware()

        # Close HTTP client
        try:
            iface.close()
        except Exception:
            logger.exception("Failed to close iface (ignored).")

        self.iface = None

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        logger.info("Resetting environment.")

        try:
            data: ResetResponse = self.iface.reset()
            logger.info(f"reset data: {data.model_dump(exclude={'observation'})} | observation shape: {data.observation.shape}")

            self.obs = data.observation
            info = data.info
            return self.obs, info
        except Exception as e:
            logger.exception("Failed to reset environment")
            self._disconnect()
            raise RuntimeError(f"Failed to reset environment: {e}")

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        logger.info(f"Making a step with action: {action}")

        try:
            data: StepResponseJSON | StepResponseMultipart = self.iface.step_json(action) # self.iface.step_multipart(action)
            logger.info(f"step_json data: {data.model_dump(exclude={'observation'})} | observation shape: {data.observation.shape}")

            self.obs = self._draw_goal(data.observation)
            reward = 0.0
            terminated = False
            truncated = data.truncated
            info = data.info
            self._episode += 1
            info["episode"] = f"{self._episode}/{self._max_episode_steps if self._max_episode_steps is not None else '∞'}"
            return self.obs, reward, terminated, truncated, info
        except Exception as e:
            logger.exception("Failed to step environment")
            self._disconnect()
            raise RuntimeError(f"Failed to step environment: {e}")

    def render(self):
        logger.debug(f"OpenCV rendering mode: {self.render_mode}")
        if self.render_mode == "opencv":
            if self.obs is not None:
                show_image = cv.cvtColor(self.obs, cv.COLOR_RGB2BGR)
                cv.imshow("Environment", show_image)
                cv.waitKey(1)

    def close(self) -> None:
        logger.info("Closing environment.")
        self._disconnect()
        cv.destroyAllWindows()

    def _draw_goal(self, image: np.ndarray) -> np.ndarray:
        """Draw transparent goal indicator."""

        annotated_image = image.copy()

        # Create overlay (same size as image)
        overlay = annotated_image.copy()

        thickness = -1  # Filled circle to allow transparency

        cv.circle(overlay, self.goal_position, cfg.GOAL_RADIUS, cfg.GOAL_COLOUR, thickness)

        # Transparency factor (0.0 = invisible, 1.0 = fully visible)
        alpha = cfg.GOLA_ALPHA

        # Blend overlay onto original
        cv.addWeighted(overlay, alpha, annotated_image, 1 - alpha, 0, annotated_image)

        # Draw outline for better visibility (optional)
        cv.circle(annotated_image, self.goal_position, cfg.GOAL_RADIUS, (0, 180, 0), 2)

        return annotated_image

