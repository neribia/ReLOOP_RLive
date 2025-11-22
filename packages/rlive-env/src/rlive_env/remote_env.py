from typing import Tuple, Optional

import numpy as np
import gymnasium as gym

from rlive_env.world_client import WorldInterface
from rlive_common.core.response import ResetResponse, StepResponseJSON
from rlive_common.utils import get_logger

logger = get_logger(__name__)


class RemoteWorldEnv(gym.Env):
    """
    Gymnasium-compatible environment that communicates with a remote World server over HTTP.
    """

    metadata = {"render_modes": []}

    def __init__(self, **kwargs) -> None:
        """
        Initialize the environment.
        Attributes:
            - base_url (Optional[str]): Base URL for remote environment.
            - timeout (Optional[float]): Time in seconds to wait for the server to send data
        """
        super().__init__()

        self.iface: Optional[WorldInterface] = None

        self.action_space = gym.spaces.Discrete(1)  # placeholder (one valid action)

        self._connect(**kwargs)

    def _connect(self, **kwargs):
        """Create iface and attach hardware."""
        self.iface = WorldInterface(**kwargs)

        resp = self.iface.attach_hardware()
        if not resp.success:
            raise RuntimeError("Failed to connect the hardware")

        return resp

    def _disconnect(self):
        """Detach hardware and close HTTP interface (robust gegen Fehler)."""
        iface = getattr(self, "iface", None)
        if iface is None:
            return

        try:
            iface.detach_hardware()
        except Exception:
            logger.exception("Failed to detach hardware (ignored).")

        try:
            iface.close()
        except Exception:
            logger.exception("Failed to close iface (ignored).")

        self.iface = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        logger.info(f"Resetting environment.")

        try:
            data: ResetResponse = self.iface.reset()
            logger.info(f"reset data: {data}")

            obs = data.observation
            info = data.info
            return obs, info
        except Exception as e:
            logger.exception(f"Failed to reset environment")
            self._disconnect()
            raise RuntimeError(f"Failed to reset environment: {e}")

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        logger.info(f"Making a step: {action}")

        try:
            data: StepResponseJSON = self.iface.step_json(action) # self.iface.step_multipart(action)
            logger.info(f"step_json data: {data.model_dump(exclude={'image'})} | image: {data.image.shape}")

            obs = data.observation
            reward = 0.0
            terminated = False
            truncated = data.truncated
            info = data.info
            return obs, reward, terminated, truncated, info
        except Exception as e:
            logger.exception(f"Failed to step environment")
            self._disconnect()
            raise RuntimeError(f"Failed to step environment: {e}")

    def close(self) -> None:
        logger.info(f"Closing environment.")
        self._disconnect()
