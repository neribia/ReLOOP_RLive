from typing import Any

import numpy as np
import cv2 as cv
import gymnasium as gym

from rlive_env.config import config as cfg
from rlive_common.utils import get_logger

logger = get_logger(__name__)


class SimulationEnv(gym.Env):
    """Gymnasium-compatible environment that communicates with a remote World server over HTTP.
    """

    metadata = {"render_modes": ["opencv"]}

    def __init__(self, max_episode_steps: int | None = 100, render_mode: str | None = None, options: dict[str, Any] | None = None,
                 **kwargs) -> None:
        """Initialize the environment.

        Attributes:
            - max_episodes (int | None): Max epochs (default: 100)
            - render_mode (Optional[str]): Rendering mode ('opencv' or None)
        """
        super().__init__()

        self._max_episode_steps = max_episode_steps
        self._episode = 0  # Start from 0 or 1? Other Env's as reference.
        self.render_mode = render_mode
        self.obs = None
        self.options = options or {}

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(360, start=-179)  # placeholder (one valid action)
        # Goal variables
        self.goal_position = None


    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """ Resets the Environment.

        Attributes:
            - seed (int): Set the seed for the randomizer.
            - options (dict):

        Returns:
            - Tuple:
                - observation (np.ndarray): Observation of the current state.
                - info (dict): Information about the Environment.
        """
        super().reset(seed=seed)
        logger.info("Resetting environment.")

        self.set_random_goal()
        info: dict[str, Any] = {"status": "ok"}

        return self.obs, info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Make a step in the environment with the given action.

        Attributes:
            - action: The action to take in the environment.

        Returns:
            - obs (np.ndarray): The next observation after taking the action.
            - reward (float): The reward received after taking the action.
            - terminated (bool): Whether the episode has terminated.
            - truncated (bool): Whether the episode has been truncated.
            - info (dict): Additional information about the step.
        """
        logger.info(f"Making a step with action: {action}")

        self.obs = np.random.randint(0, 256, size=self.observation_space.shape, dtype=np.uint8)
        truncated, reward = self.calculate_reward(self.obs)
        terminated = False  # Placeholder logic
        info: dict[str, Any] = {"status": "ok"}

        return self.obs, reward, terminated, truncated, info


    def render(self):
        logger.debug(f"OpenCV rendering mode: {self.render_mode}")
        if self.render_mode == "opencv":
            if self.obs is not None:
                show_image = cv.cvtColor(self.obs, cv.COLOR_RGB2BGR)
                cv.imshow("Environment", show_image)
                cv.waitKey(1)

    def close(self) -> None:
        logger.info("Closing environment.")
        cv.destroyAllWindows()

    def calculate_reward(self, observation: np.ndarray) -> tuple[bool, float]:
        """Calculate reward based on the observation and goal position.

        Attributes:
            - observation (np.ndarray): The current observation from the environment.

        Returns:
            - terminated (bool): Whether the episode has terminated.
            - reward (float): The calculated reward.
        """
        # Placeholder implementation: return a constant reward
        return False, 0.0

    def set_random_goal(self):
        height, width, _ = self.observation_space.shape

        x = np.random.randint(0, width)
        y = np.random.randint(0, height)

        self.goal_position = (x, y)

    def _draw_goal(self, image: np.ndarray) -> np.ndarray:
        """Draw transparent goal indicator."""

        return None
