"""Simulation Environment module.

This module provides a Gymnasium-compatible environment that uses
the SimulationEngine for physics and rendering.

Supports both separate engine configurations and integrated engines,
providing a common RL interface for different simulation architectures.
"""

from typing import Any

import numpy as np
import cv2 as cv
import gymnasium as gym

from rlive_common.utils import get_logger
from rlive_common.core.action_space import get_action_transformer, ActionSpaceType, BaseActionTransformer
from rlive_common.core.ball_location import BallLocation
from rlive_common.config import config as common_cfg
from rlive_common.utils.visualisation_utils import draw_goal, annotate_image
from rlive_sim.config import SimulationConfig, BOLT_DEFAULTS
from rlive_sim.config import config as sim_cfg
import rlive_sim.config.config as cfg
from rlive_sim.engine import SimulationEngine
from rlive_sim.engine.core.base_physics_engine import PhysicsState
from rlive_sim.engine.core.factories import SimulationEngineFactory

logger = get_logger(__name__)


class SimulationEnv(gym.Env):
    """Gymnasium-compatible simulation environment.

    This environment uses SimulationEngine to handle physics and rendering,
    supporting both separate and integrated engine architectures transparently.

    The environment can be configured in three ways:
    1. Use default configuration (basic setup)
    2. Inject pre-configured SimulationEngine directly
    3. Provide custom SimulationConfig with specific engines

    Attributes:
        engine: The SimulationEngine instance handling physics and rendering.
        render_mode: Current render mode ('opencv' or None).
        obs: Current observation (rendered image).
        goal_position: Current goal position for reward calculation.

    Methods:
        reset: Reset the environment to initial state.
        step: Execute one step in the environment.
        render: Render the current environment state.
        close: Clean up environment resources.
        calculate_reward: Calculate reward based on observation and goal.
        set_random_goal: Set a random goal position within the observation space.

    Examples:
        Using default configuration:

            env = SimulationEnv()
            obs, info = env.reset()
            obs, reward, terminated, truncated, info = env.step(action)

        With separate engines (SimulationEngine combines them):

            from rlive_sim.engine import SimplePhysicsEngine, OpenCVRenderEngine

            physics = SimplePhysicsEngine(box_width=640, box_height=480)
            render = OpenCVRenderEngine(width=640, height=480)

            from rlive_sim.engine import SimulationEngine
            sim = SimulationEngine(physics_engine=physics, render_engine=render)
            env = SimulationEnv(engine=sim)

        With integrated engine:

            from rlive_sim.config import IntegratedConfig, IntegratedBackend
            from rlive_sim.engine import IntegratedEngine

            config = IntegratedConfig(backend=IntegratedBackend.MUJOCO)
            integrated = IntegratedEngine(config)
            env = SimulationEnv(engine=SimulationEngine(integrated_engine=integrated))

        With custom config:

            from rlive_sim.config import SimulationConfig, PhysicsConfig, RenderConfig

            config = SimulationConfig(
                physics=PhysicsConfig(dt=0.005),
                render=RenderConfig(width=800, height=600),
            )
            env = SimulationEnv(config=config)
    """

    metadata = {"render_modes": ["opencv"]}

    def __init__(
        self,
        engine: SimulationEngine | None = None,
        config: SimulationConfig | None = None,
        max_episode_steps: int | None = sim_cfg.MAX_STEPS_PER_EPISODE,
        render_mode: str | None = None,
        action_space_type: ActionSpaceType | str = ActionSpaceType.CARTESIAN,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the simulation environment.

        Args:
            engine: Optional pre-configured SimulationEngine. If provided,
                config is ignored for engine creation.
            config: Optional SimulationConfig. Uses default_config if None.
            max_episode_steps: Maximum steps per episode. Overrides config
                value if provided.
            render_mode: Rendering mode ('opencv' or None).
            action_space_type: Action space transformer type. Default: ActionSpaceType.CARTESIAN
            options: Additional environment options.
            **kwargs: Additional arguments passed to engine creation.
        """
        super().__init__()

        # Use provided config or default
        self._config = config if config is not None else SimulationConfig()

        # Set up the simulation engine
        if engine is not None:
            self.engine = engine
        else:
            self.engine = SimulationEngineFactory(self._config, **kwargs)

        # Apply default scene config here if none supplied explicitly
        scene_config = kwargs.get("scene_config", {})
        self.engine.setup_scene(scene_config)

        # Action space transformer
        logger.info(f"Setting up action space transformer: {action_space_type}")
        try:
            self.action_transformer = get_action_transformer(
                action_space_type,
                speed=cfg.SPHEROBOLTPLUS_SPEED,
                duration=cfg.SPHEROBOLTPLUS_DURATION,
                speed_factor=cfg.SPHEROBOLTPLUS_SPEED_FACTOR,
            )

            # Explicitly validate the returned object
            if not isinstance(self.action_transformer, BaseActionTransformer):
                raise TypeError(f"Expected BaseActionTransformer, got {type(self.action_transformer).__name__}")

            logger.debug(f"Action transformer initialized: {self.action_transformer.__class__.__name__}")
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to initialize action transformer: {e}")
            raise ValueError(f"Invalid action_space_type '{action_space_type}': {e}") from e

        # Environment settings
        self._max_episode_steps = max_episode_steps
        self._current_step = 0
        self._episode = 0
        self.render_mode = render_mode
        self.obs: np.ndarray | None = None
        self.options = options or {}

        # Set up spaces based on engine resolution
        obs_shape = self.engine.get_resolution()
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )
        self.action_space = self.action_transformer.get_action_space()

        # Goal variables
        self.goal_position: tuple[int, int] | None = None
        self.ball_location: BallLocation | None = None
        self._max_distance = (obs_shape[0] ** 2 + obs_shape[1] ** 2) ** 0.5


    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Optional random seed for reproducibility.
            options: Optional reset options. Supported keys:
                - ``"initial_state"`` (PhysicsState | None): Place the robot at a
                  specific position/orientation instead of the engine's default.
                - ``"goal_position"`` (tuple[int, int] | None): Manually set the goal
                  position as ``(x, y)`` pixel coordinates. If omitted or ``None``,
                  a random goal is chosen.

        Returns:
            tuple[np.ndarray, dict[str, Any]]: Initial observation and info dict.
        """
        super().reset(seed=seed)
        logger.debug("Resetting environment.")

        # Extract options
        initial_state: PhysicsState | None = None
        manual_goal: tuple[int, int] | None = None
        if options is not None:
            initial_state = options.get("initial_state", None)
            manual_goal = options.get("goal_position", None)

        # Reset the simulation engine
        state, self.obs = self.engine.reset(initial_state)

        self._current_step = 0
        self._episode += 1

        # Set goal position – manual override or random
        if manual_goal is not None:
            self.goal_position = tuple(manual_goal)
            logger.debug(f"Goal position manually set to: {self.goal_position}")
        else:
            self.set_random_goal()

        self.obs = draw_goal(self.obs, self.goal_position)

        info: dict[str, Any] = {
            "status": "ok",
            "episode": self._episode,
            "goal_position": self.goal_position,
        }

        return self.obs, info

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: The action to take. Format depends on action_space.

        Returns:
            tuple containing:
                - obs (np.ndarray): Observation after the action.
                - reward (float): Reward received.
                - terminated (bool): Whether episode ended naturally.
                - truncated (bool): Whether episode was truncated (e.g., max steps).
                - info (dict[str, Any]): Additional information.
        """
        logger.debug(f"Making a step with action: {action}")

        # Transform action using the configured action space transformer
        try:
            transformed_action = self.action_transformer.transform(action)
            logger.debug(f"Action transformed from {action} to {transformed_action}")
        except ValueError as e:
            logger.error(f"Failed to transform action: {e}")
            return self.obs, 0.0, False, True, {"error": f"Invalid action: {e}"}

        # Apply transformed action and step physics and rendering
        physics_state, self.obs = self.engine.update_and_render(transformed_action)
        self.obs = draw_goal(self.obs, self.goal_position)
        self._current_step += 1

        # Calculate reward and check termination
        terminated, reward = self.calculate_reward(self.obs)
        truncated = self._current_step >= self._max_episode_steps

        info: dict[str, Any] = {
            "is_success": terminated,
            "status": "ok",
            "step": self._current_step,
            "physics_state": physics_state.model_dump() if physics_state else None,
        }

        return self.obs, reward, terminated, truncated, info

    def render(self, visualize: bool = False) -> np.ndarray | None:
        """Render the current environment state.

        Returns:
            np.ndarray | None: Rendered image if render_mode is set.
        """
        logger.debug(f"Render mode: {self.render_mode}")

        if self.render_mode == "opencv":
            if self.obs is not None:
                show_image = cv.cvtColor(self.obs, cv.COLOR_RGB2BGR)
                show_image = show_image.copy()

                # First draw transparent goal overlay
                if visualize and self.goal_position is not None :
                    show_image = annotate_image(show_image, self.ball_location, self.goal_position)

                cv.imshow("Simulation Environment", show_image)
                cv.waitKey(1)
            return self.obs

        return None

    def close(self) -> None:
        """Clean up environment resources."""
        logger.info("Closing environment.")
        if self.engine is not None:
            self.engine.close()
        cv.destroyAllWindows()

    def calculate_reward(
        self, observation: np.ndarray
    ) -> tuple[bool, float]:
        """Calculate reward based on observation and goal.

        Args:
            observation: Current observation image.

        Returns:
            tuple[bool, float]: (goal_reached, reward).
        """
        if self.goal_position is None:
            logger.warning("Goal position not set, returning zero reward")
            return False, 0.0

        ball_pos_tuple = self.engine.get_ball_2d_position()

        if ball_pos_tuple is None:
            logger.debug("Ball not detected/visible in observation")
            # Penalize for losing the ball
            return False, -0.1

        self.ball_location = BallLocation(x=ball_pos_tuple[0], y=ball_pos_tuple[1])

        # Check if goal is reached
        goal_radius = self.options.get("goal_radius", common_cfg.GOAL_RADIUS)
        goal_reached = self.ball_location.is_within_radius(self.goal_position, goal_radius)

        reward_mode = self.options.get("reward_mode", "dense")

        if reward_mode == "sparse":
            # Sparse reward: +1.0 only when goal is reached
            reward = 1.0 if goal_reached else 0.0
        else:
            # Dense reward: based on distance
            distance = self.ball_location.distance_to(self.goal_position)
            reward = - distance / self._max_distance

        logger.debug(f"Ball at {self.ball_location.as_tuple()}, goal at {self.goal_position}, "
                     f"reward={reward:.3f}, goal_reached={goal_reached}")

        return goal_reached, float(reward)

    def set_random_goal(self) -> None:
        """Set a random goal position within the observation space."""
        height, width, _ = self.observation_space.shape
        goal_radius = self.options.get("goal_radius", common_cfg.GOAL_RADIUS)

        min_x, min_y, max_x, max_y = self.engine.get_reachable_bounds()
        
        # Try to find a valid goal point projecting to the camera view
        max_tries = self._config.max_goal_tries
        for _ in range(max_tries):
            # Sample physical position
            x3d = self.np_random.uniform(min_x, max_x)
            y3d = self.np_random.uniform(min_y, max_y)
            z3d = BOLT_DEFAULTS.radius_m # Standard radius / ground plane assumption

            proj = self.engine.project_position_to_2d((x3d, y3d, z3d))
            if proj is not None:
                px, py = proj
                # Check if it fits well within image bounds
                if goal_radius <= px <= width - goal_radius and goal_radius <= py <= height - goal_radius:
                    self.goal_position = (px, py)
                    logger.debug(f"Set random goal at: {self.goal_position} from 3d {x3d:.2f},{y3d:.2f}")
                    return

        # Fallback if valid projection wasn't found
        logger.warning("Could not find a valid projected goal position. Using fallback center position.")
        self.goal_position = (width // 2, height // 2)
