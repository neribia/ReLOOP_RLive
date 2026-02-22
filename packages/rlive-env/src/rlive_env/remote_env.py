from typing import Any, Literal
import math

import numpy as np
import cv2 as cv
import gymnasium as gym

from rlive_env.config import config as cfg
from rlive_env.world_client import WorldInterface
from rlive_env.localisation import BallLocalisator, BallLocation
from rlive_env.action_space import get_action_transformer, ActionSpaceType
from rlive_common.core.response import ResetResponse, StepResponseJSON, StepResponseMultipart, DetachHardwareResponse, AttachHardwareResponse
from rlive_common.utils import get_logger

logger = get_logger(__name__)


class RemoteWorldEnv(gym.Env):
    """Gymnasium-compatible environment that communicates with a remote World server over HTTP.
    """

    metadata = {"render_modes": ["opencv"]}

    def __init__(self, max_episode_steps: int | None = 100, render_mode: str | None = None, auto_attach: bool = True, options: dict[str, Any] | None = None,
                 reward_mode: Literal["dense", "sparse"] | None = None,
                 action_space_type: ActionSpaceType = ActionSpaceType.POLAR,
                 **kwargs) -> None:
        """Initialize the environment.

        Attributes:
            - max_episodes (int | None): Max epochs (default: 100)
            - render_mode (Optional[str]): Rendering mode ('opencv' or None)
            - auto_attach (bool): Automatically attach hardware on init (default: True)
            - reward_mode (str): Reward calculation mode ('dense' or 'sparse'). Defaults to config value.
            - action_space_type (ActionSpaceType): Action space transformer type. Default: ActionSpaceType.POLAR
            - base_url (Optional[str]): Base URL for remote environment.
            - timeout (Optional[float]): Time in seconds to wait for the server to send data
        """
        super().__init__()

        self._max_episode_steps = max_episode_steps
        self._episode = 0
        self.render_mode = render_mode
        self.reward_mode: Literal["dense", "sparse"] | str = reward_mode or cfg.REWARD_MODE
        self.iface: WorldInterface | None = None
        self.obs = None
        self._hardware_attached = False
        self.options = options or {}

        # Action space transformer
        logger.info(f"Setting up action space transformer: {action_space_type}")
        try:
            self.action_transformer = get_action_transformer(action_space_type)
            logger.debug(f"Action transformer initialized: {self.action_transformer.__class__.__name__}")
        except ValueError as e:
            logger.error(f"Failed to initialize action transformer: {e}")
            raise ValueError(f"Invalid action_space_type '{action_space_type}': {e}") from e

        # Ball localisation
        self.localiser = BallLocalisator()
        self.ball_location: BallLocation | None = None

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)
        self.action_space = self.action_transformer.get_action_space()
        # Goal variables
        self.goal_position = None

        # Calculate max possible distance for reward normalization (diagonal of observation space)
        height, width, _ = self.observation_space.shape
        self._max_distance = math.sqrt(width ** 2 + height ** 2)

        self._connect(auto_attach=auto_attach, **kwargs)

    def _connect(self, auto_attach: bool = True, **kwargs) -> None:
        """Create iface and optionally attach hardware."""
        logger.info("Setting up interface to RemoteWorld.")
        self.iface = WorldInterface(**kwargs)

        # Verify server is healthy before attempting to attach hardware
        try:
            health = self.iface.health_check()
            logger.info(f"Server health check passed: {health}")
        except Exception as e:
            logger.error("Server health check failed, aborting connection.", exc_info=e)
            raise RuntimeError("RemoteWorld server health check failed") from e

        if auto_attach:
            self.attach_hardware()

    def attach_hardware(self) -> AttachHardwareResponse | None:
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

    def detach_hardware(self) -> DetachHardwareResponse | None:
        """Explicitly detach hardware from the world server."""
        if not self._hardware_attached:
            logger.debug("Hardware not attached, skipping detach.")
            return None

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
        self._episode = 0

        # Verify server status before reset
        try:
            status = self.iface.get_status()
            logger.debug(f"Server status: {status}")
        except Exception as e:
            logger.error(f"Cannot reset: server status check failed: {e}")
            self._disconnect()
            raise RuntimeError(f"Server connection error: {e}")

        self.set_random_goal()

        try:
            data: ResetResponse = self.iface.reset()
            logger.info(f"reset data: {data.model_dump(exclude={'observation'})} | observation shape: {data.observation.shape}")

            self.obs = self._draw_goal(data.observation)
            info = data.info
            return self.obs, info
        except Exception as e:
            logger.exception("Failed to reset environment")
            self._disconnect()
            raise RuntimeError(f"Failed to reset environment: {e}")

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Make a step in the environment with the given action.

        Attributes:
            - action: The action to take in the environment (format depends on action_space_type).

        Returns:
            - obs (np.ndarray): The next observation after taking the action.
            - reward (float): The reward received after taking the action.
            - terminated (bool): True if episode ended naturally (success/failure)
            - truncated (bool): True if episode ended due to time/step limit
            - info (dict): Additional information about the step.
        """
        logger.info(f"Making a step with action: {action}")

        # Transform action using the configured action space transformer
        try:
            transformed_action = self.action_transformer.transform_action(action)
            logger.debug(f"Action transformed from {action} to {transformed_action}")
        except ValueError as e:
            logger.error(f"Failed to transform action: {e}")
            return self.obs, 0.0, False, True, {"error": f"Invalid action: {e}"}

        try:
            data: StepResponseJSON | StepResponseMultipart = self.iface.step_json(transformed_action) # self.iface.step_multipart(transformed_action)
            logger.info(f"step_json data: {data.model_dump(exclude={'observation'})} | observation shape: {data.observation.shape}")

            # Ball localisation before draw_goal
            terminated, reward = self.calculate_reward(observation=data.observation)

            # Draw goal after ball localisation
            self.obs = self._draw_goal(data.observation)

            self._episode += 1
            truncated = data.truncated or (self._max_episode_steps is not None and self._episode >= self._max_episode_steps)
            info = data.info
            info["episode"] = f"{self._episode}/{self._max_episode_steps if self._max_episode_steps is not None else '∞'}"
            return self.obs, reward, terminated, truncated, info
        except Exception as e:
            logger.exception("Failed to step environment")
            return self.obs, 0.0, False, True, {"error": str(e)}  # Return truncated=True to end episode on error

    def render(self):
        logger.debug(f"OpenCV rendering mode: {self.render_mode}")
        if self.render_mode == "opencv":
            if self.obs is not None:
                show_image = self.localiser.annotate_image(self.obs,self.ball_location, self.goal_position, cfg.GOAL_RADIUS)
                show_image = cv.cvtColor(show_image, cv.COLOR_RGB2BGR)
                cv.imshow("Environment", show_image)
                cv.waitKey(1)

    def close(self) -> None:
        logger.info("Closing environment.")
        self._disconnect()
        cv.destroyAllWindows()

    def calculate_reward(self, observation: np.ndarray) -> tuple[bool, float]:
        """Calculate reward based on the observation and goal position.

        Uses the BallLocalisator to locate the ball and computes reward based on
        the configured reward_mode:
        - 'dense': Distance-based reward (1.0 - distance/max_distance)
        - 'sparse': Binary reward (+1.0 only when goal is reached)

        Attributes:
            - observation (np.ndarray): The current observation from the environment.

        Returns:
            - goal_reached (bool): Whether the ball has reached the goal.
            - reward (float): The calculated reward.
        """
        if self.goal_position is None:
            logger.warning("Goal position not set, returning zero reward")
            raise RuntimeError("Goal position not set, cannot calculate reward")

        # Locate the ball in the observation
        # Note: observation might be RGB, convert to BGR for OpenCV
        bgr_image = cv.cvtColor(observation, cv.COLOR_RGB2BGR)
        self.ball_location = self.localiser.get_position(bgr_image)

        if self.ball_location is None:
            logger.debug("Ball not detected in observation")
            raise RuntimeError("Ball not detected in observation")

        # Check if goal is reached
        goal_reached = self.ball_location.is_within_radius(self.goal_position, cfg.GOAL_RADIUS) # TODO: radius as Env option.
        logger.debug(f"Ball location: {self.ball_location.as_tuple()}, Goal position: {self.goal_position}, Goal reached: {goal_reached}")

        if self.reward_mode == "sparse":
            # Sparse reward: +1.0 only when goal is reached
            reward = 1.0 if goal_reached else 0.0
        else:
            # Dense reward: higher reward when closer to goal
            distance = self.ball_location.distance_to(self.goal_position)
            reward = 1.0 - (distance / self._max_distance) # FIXME: Normalized or in px?

            # Bonus reward for reaching the goal
            if goal_reached:
                reward += 1.0 # FIXME: Does this make's sense?

        logger.debug(f"Ball at {self.ball_location.as_tuple()}, goal at {self.goal_position}, "
                     f"reward={reward:.3f}, goal_reached={goal_reached}")

        return goal_reached, reward

    def set_random_goal(self):
        """Set a random goal position that is fully visible in the observation.

        The goal is constrained to be at least GOAL_RADIUS away from the borders
        to ensure the entire goal circle is visible.
        """
        height, width, _ = self.observation_space.shape

        # Ensure goal is fully visible by constraining it away from borders
        min_x = cfg.GOAL_RADIUS
        max_x = width - cfg.GOAL_RADIUS
        min_y = cfg.GOAL_RADIUS
        max_y = height - cfg.GOAL_RADIUS

        x = int(self.np_random.integers(min_x, max_x))
        y = int(self.np_random.integers(min_y, max_y))

        self.goal_position = (x, y)
        logger.debug(f"Random goal set at {self.goal_position} (constrained to {min_x}-{max_x}, {min_y}-{max_y})")

    def _draw_goal(self, image: np.ndarray) -> np.ndarray:
        """Draw transparent goal indicator."""

        annotated_image = image.copy()

        # Create overlay (same size as image)
        overlay = annotated_image.copy()

        thickness = -1  # Filled circle to allow transparency

        cv.circle(overlay, self.goal_position, cfg.GOAL_RADIUS, cfg.GOAL_COLOUR, thickness)

        # Transparency factor (0.0 = invisible, 1.0 = fully visible)
        alpha = cfg.GOAL_ALPHA

        # Blend overlay onto original
        cv.addWeighted(overlay, alpha, annotated_image, 1 - alpha, 0, annotated_image)

        # Draw outline for better visibility (optional)
        cv.circle(annotated_image, self.goal_position, cfg.GOAL_RADIUS, (0, 180, 0), 2)

        return annotated_image
