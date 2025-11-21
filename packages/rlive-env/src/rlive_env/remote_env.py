from typing import Tuple, Optional

import numpy as np
import gymnasium as gym

from rlive_env.config import config as cfg
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
        self.iface = WorldInterface(**kwargs)

        self.action_space = gym.spaces.Discrete(1)  # placeholder (one valid action)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        logger.info(f"Resetting environment.")
        data: ResetResponse = self.iface.reset()
        logger.info(f"reset data: {data}")

        obs = data.observation
        info = data.info
        return obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        logger.info(f"Making a step: {action}")

        data: StepResponseJSON = self.iface.step_json(action)
        logger.info(f"step_json data: {data}")

        # data = self.iface.step_multipart(action)
        # logger.info(f"step_multipart data: {data}")

        # Handle None gracefully
        obs = data.observation
        reward = 0.0
        terminated = False
        truncated = data.truncated
        info = data.info
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        logger.info(f"Closing environment.")
        self.iface.close()