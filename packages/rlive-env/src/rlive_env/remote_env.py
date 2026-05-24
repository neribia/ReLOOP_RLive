from typing import Any, Literal
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2 as cv
import gymnasium as gym

from rlive_common.config import config as common_cfg
from rlive_common.utils.visualisation_utils import annotate_image, draw_goal, compute_goal_radius
from rlive_env.config import config as cfg
from rlive_env.world_client import WorldInterface
from rlive_env.localisation import BallLocalisator
from rlive_common.core.ball_location import BallLocation
from rlive_common.core.action_space import get_action_transformer, ActionSpaceType, BaseActionTransformer
from rlive_common.core.response import ResetResponse, StepResponseJSON, StepResponseMultipart, DetachHardwareResponse, AttachHardwareResponse
from rlive_common.core import WorldConfig
from rlive_common.utils import get_logger

logger = get_logger(__name__)


class RemoteWorldEnv(gym.Env):
    """Gymnasium-compatible environment that communicates with a remote World server over HTTP."""

    metadata = {"render_modes": ["opencv"]}

    def __init__(self, max_episode_steps: int | None = cfg.MAX_STEPS_PER_EPISODE,
                 render_mode: str | None = None,
                 auto_attach: bool = True,
                 world_config: WorldConfig | None = None,
                 reward_mode: Literal["dense", "sparse"] | None = None,
                 action_space_type: ActionSpaceType | str = ActionSpaceType.CARTESIAN,
                 decay: int = cfg.NUMBER_RESET_ACTIONS,
                 fixed_goal: bool = False,
                 save_failed_detections: bool = True,
                 options: dict | None = None,
                 **kwargs,
                 ) -> None:
        """Initialize the environment.

        Attributes:
            - max_episodes (int | None): Max epochs (default: 100)
            - render_mode (Optional[str]): Rendering mode ('opencv' or None)
            - auto_attach (bool): Automatically attach hardware on init (default: True)
            - world_config (WorldConfig | None): World configuration with camera and bolt settings.
              None values use rlive-world defaults.
            - reward_mode (str): Reward calculation mode ('dense' or 'sparse'). Defaults to config value.
            - action_space_type (ActionSpaceType): Action space transformer type. Default: ActionSpaceType.CARTESIAN
            - decay (int | None): Number of random scatter actions sent during reset to decouple episodes.
              Set to 0 to skip scatter moves entirely. Defaults to cfg.NUMBER_RESET_ACTIONS (env var DECAY_STEPS).
            - fixed_goal (bool): If True, the goal is always placed at the centre of the image (default: False).
            - save_failed_detections (bool): If True, saves raw + debug images to disk whenever ball
              detection fails (default: False).
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
        self._raw_image = None
        self._hardware_attached = False
        self.world_config = world_config or WorldConfig()

        # Action space transformer
        logger.info(f"Setting up action space transformer: {action_space_type}")
        try:
            self.action_transformer = get_action_transformer(
                action_space_type,
                speed=int(cfg.SPHEROBOLTPLUS_SPEED),
                duration=float(cfg.SPHEROBOLTPLUS_DURATION),
                speed_factor=float(cfg.SPHEROBOLTPLUS_SPEED_FACTOR)
            )

            # Explicitly validate the returned object
            if not isinstance(self.action_transformer, BaseActionTransformer):
                raise TypeError(f"Expected BaseTransformer, got {type(self.action_transformer).__name__}")

            logger.debug(f"Action transformer initialized: {self.action_transformer.__class__.__name__}")
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to initialize action transformer: {e}")
            raise ValueError(f"Invalid action_space_type '{action_space_type}': {e}") from e

        # Ball localisation
        self.localiser = BallLocalisator(debug=True)
        self.ball_location: BallLocation | None = None

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)
        self.action_space = self.action_transformer.get_action_space()
        # Goal variables
        self.goal_position = None
        self._fixed_goal = fixed_goal

        # Failed-detection saving
        self._save_failed_detections = save_failed_detections
        self._failed_detections_dir = Path("failed_detections")
        self._failed_detection_count = 0

        # Decay for reset
        self._decay = decay

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
        resp = self.iface.attach_hardware(world_config=self.world_config)
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
        """Reset the environment to initial state.

        Args:
            seed: Optional random seed for reproducibility.
            options: Optional reset options. Supported keys:
                - ``"goal_position"`` (tuple[int, int] | None): Manually set the goal
                  position as ``(x, y)`` pixel coordinates. If omitted or ``None``,
                  a random goal is chosen.
        """
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

        # Set goal position – manual override, fixed centre, or random
        manual_goal: tuple[int, int] | None = options.get("goal_position", None) if options is not None else None
        if manual_goal is not None:
            self.goal_position = (int(manual_goal[0]), int(manual_goal[1]))
            logger.debug(f"Goal position manually set to: {self.goal_position}")
        elif self._fixed_goal:
            h, w = self.observation_space.shape[:2]
            self.goal_position = (w // 2, h // 2)
            logger.debug(f"Goal position fixed to centre: {self.goal_position}")
        else:
            self.set_random_goal()

        try:
            actions = []
            for _ in range(self._decay):
                action = self.action_space.sample()
                actions.append(self.action_transformer.transform(action))

            data: ResetResponse = self.iface.reset(actions)
            logger.info(f"reset data: {data.model_dump(exclude={'observation'})} | observation shape: {data.observation.shape}")

            self._raw_image = data.observation
            self.obs = draw_goal(data.observation, self.goal_position)
            info = data.info
            return self.obs, info
        except Exception as e:
            logger.exception("Failed to reset environment")
            self._disconnect()
            raise RuntimeError(f"Failed to reset environment: {e}")

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Make a step in the environment with the given action.

        Attributes:
            - action: The action to take in the environment (format depends on action_space_type).

        Returns:
            - obs (np.ndarray): The next observation after taking the action.
            - reward (float): The reward received after taking the action.
            - terminated (bool): True if episode ended naturally (success/failure)
            - truncated (bool): True if episode ended due to time/step limit
            - info (dict): Additional information about the step.
        """
        logger.debug(f"Making a step with action: {action}")

        # Transform action using the configured action space transformer
        try:
            transformed_action = self.action_transformer.transform(action)
            logger.debug(f"Action transformed from {action} to {transformed_action}")
        except ValueError as e:
            logger.error(f"Failed to transform action: {e}")
            return self.obs, 0.0, False, True, {"error": f"Invalid action: {e}"}

        try:
            data: StepResponseJSON | StepResponseMultipart = self.iface.step_json(transformed_action) # self.iface.step_multipart(transformed_action)
            logger.debug(f"step_json data: {data.model_dump(exclude={'observation'})} | observation shape: {data.observation.shape}")

            self._raw_image = data.observation

            # Ball localisation before draw_goal
            terminated, reward = self.calculate_reward(observation=data.observation)

            # Draw goal after ball localisation
            self.obs = draw_goal(data.observation, self.goal_position)

            self._episode += 1
            truncated = data.truncated or (self._max_episode_steps is not None and self._episode >= self._max_episode_steps)
            info = data.info
            info["episode"] = f"{self._episode}/{self._max_episode_steps if self._max_episode_steps is not None else '∞'}"
            info["is_success"] = terminated
            return self.obs, reward, terminated, truncated, info
        except Exception as e:
            logger.exception("Failed to step environment")
            return self.obs, 0.0, False, True, {"error": str(e)}  # Return truncated=True to end episode on error

    def render(self, scale: float = 1.0, visualize: bool = True) -> None:
        logger.debug(f"OpenCV rendering mode: {self.render_mode}")
        if scale <= 0:
            logger.warning(f"Invalid scale {scale}, must be > 0. Defaulting to 1.0.")
            scale = 1.0
        if self.render_mode == "opencv":
            if self.obs is not None:
                show_image = self.obs.copy()
                if visualize:
                    show_image = annotate_image(show_image, self.ball_location, self.goal_position,
                                               compute_goal_radius(self.observation_space.shape))
                show_image = cv.cvtColor(show_image, cv.COLOR_RGB2BGR)
                res_show_image = cv.resize(show_image, dsize=None, fx=scale, fy=scale)
                cv.imshow("Environment", res_show_image)

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
        # TODO: Create a common reward caluclation for remote_env and simulation_env
        if self.goal_position is None:
            logger.warning("Goal position not set, returning zero reward")
            raise RuntimeError("Goal position not set, cannot calculate reward")

        # Locate the ball in the observation
        self.ball_location = self.localiser.get_position(observation)

        if self.ball_location is None:
            logger.debug("Ball not detected in observation")
            if self._save_failed_detections:
                self._save_failed_detection(observation)
            raise RuntimeError("Ball not detected in observation")

        # Check if goal is reached – radius scales with image diagonal by default
        goal_radius = compute_goal_radius(observation.shape)
        goal_reached = self.ball_location.is_within_radius(self.goal_position, goal_radius)
        logger.debug(f"Ball location: {self.ball_location.as_tuple()}, Goal position: {self.goal_position}, Goal reached: {goal_reached}")

        if self.reward_mode == "sparse":
            # Sparse reward: +1.0 only when goal is reached
            reward = 1.0 if goal_reached else 0.0
        else:
            # Dense reward: based on distance + bonus for reaching goal
            distance = self.ball_location.distance_to(self.goal_position)
            reward =  - distance / self._max_distance
            if goal_reached:
                reward += 1.0

        logger.debug(f"Ball at {self.ball_location.as_tuple()}, goal at {self.goal_position}, "
                     f"reward={reward:.3f}, goal_reached={goal_reached}")

        return goal_reached, reward

    def _save_failed_detection(self, rgb_image: np.ndarray) -> None:
        """Save a collage image to disk when ball detection fails, for later debugging.

        Stacks raw frame, pipeline output and extractor debug side-by-side into a
        single PNG so each failure is a single file.

        Colour legend in the extractor panel:
        - Gray   — contours below ``min_contour_area`` (with A= label)
        - Orange — in area range but failing circularity (with C= / A= labels)
        - Green  — passed all filters
        - Red    — selected detection

        Args:
            rgb_image: The RGB observation received from the server.
        """
        self._failed_detections_dir.mkdir(parents=True, exist_ok=True)
        self._failed_detection_count += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = self._failed_detections_dir / f"{self._failed_detection_count:04d}_{ts}.png"

        target_h, target_w = rgb_image.shape[:2]

        def _to_bgr(img: np.ndarray) -> np.ndarray:
            """Normalise any panel to a 3-channel BGR image at target size."""
            if len(img.shape) == 2:
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            elif img.shape[2] == 3:
                img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            return cv.resize(img, (target_w, target_h))

        def _labeled(img: np.ndarray, label: str) -> np.ndarray:
            bar = np.zeros((22, target_w, 3), dtype=np.uint8)
            cv.putText(bar, label, (4, 16), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            return np.vstack([bar, img])

        panels = [_labeled(_to_bgr(rgb_image), "raw (RGB→BGR)")]

        last = self.localiser.last_result
        if last is not None:
            if last.debug_image is not None:
                panels.append(_labeled(_to_bgr(last.debug_image), "pipeline"))
            if last.extractor_debug_image is not None:
                panels.append(_labeled(_to_bgr(last.extractor_debug_image), "extractor"))

        collage = np.hstack(panels)
        cv.imwrite(str(filename), collage)
        logger.info(f"Saved failed-detection collage ({len(panels)} panels) → {filename}")

    def set_random_goal(self):
        """Set a random goal position that is fully visible in the observation.

        The goal is constrained to be at least GOAL_RADIUS away from the borders
        to ensure the entire goal circle is visible.
        """
        height, width, _ = self.observation_space.shape

        # Radius scales with the image diagonal so the goal is proportional to resolution
        goal_radius = compute_goal_radius(self.observation_space.shape)

        # Ensure goal is fully visible by constraining it away from borders
        min_x = goal_radius
        max_x = width - goal_radius
        min_y = goal_radius
        max_y = height - goal_radius

        x = int(self.np_random.integers(min_x, max_x))
        y = int(self.np_random.integers(min_y, max_y))

        self.goal_position = (x, y)
        logger.debug(f"Random goal set at {self.goal_position} (constrained to {min_x}-{max_x}, {min_y}-{max_y})")

