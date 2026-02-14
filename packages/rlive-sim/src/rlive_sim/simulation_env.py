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
from rlive_sim.config import SimulationConfig, config as default_config
from rlive_sim.engine import SimulationEngine

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
        max_episode_steps: int | None = None,
        render_mode: str | None = None,
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
            options: Additional environment options.
            **kwargs: Additional arguments passed to engine creation.
        """
        super().__init__()

        # Use provided config or default
        self._config = config or default_config

        # Set up the simulation engine
        if engine is not None:
            self.engine = engine
        else:
            self.engine = self._create_engine_from_config(**kwargs)

        # Environment settings
        self._max_episode_steps = max_episode_steps or self._config.max_episode_steps
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
        self.action_space = gym.spaces.Discrete(360, start=-179)  # Placeholder

        # Goal variables
        self.goal_position: tuple[int, int] | None = None


    def _create_engine_from_config(self, **kwargs: Any) -> SimulationEngine:
        """Create a SimulationEngine from configuration.

        This method creates the appropriate engine based on the config.
        Currently returns a placeholder - implement actual engine creation
        when concrete engines are available.

        Args:
            **kwargs: Additional arguments for engine creation.

        Returns:
            SimulationEngine: Configured simulation engine.

        Raises:
            NotImplementedError: When trying to create engines from config
                before concrete implementations are available.
        """
        # TODO: Implement factory pattern to create engines from config
        # For now, raise informative error
        raise NotImplementedError(
            "Engine creation from config is not yet implemented. "
            "Please provide a pre-configured SimulationEngine via the "
            "'engine' parameter. Example:\n"
            "  from rlive_sim.engine import SimulationEngine, PyMunkPhysicsEngine, MitsubaRenderEngine\n"
            "  engine = SimulationEngine(\n"
            "      physics_engine=PyMunkPhysicsEngine(),\n"
            "      render_engine=MitsubaRenderEngine(),\n"
            "  )\n"
            "  env = SimulationEnv(engine=engine)"
        )

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Optional random seed for reproducibility.
            options: Optional reset options.

        Returns:
            tuple[np.ndarray, dict[str, Any]]: Initial observation and info dict.
        """
        super().reset(seed=seed)
        logger.info("Resetting environment.")

        # Reset the simulation engine
        self.engine.reset()
        self._current_step = 0
        self._episode += 1

        # Set random goal and get initial observation
        self.set_random_goal()
        self.obs = self.engine.get_observation()

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
        logger.info(f"Making a step with action: {action}")

        # Apply action and step physics
        self.engine.apply_action(action)
        physics_state = self.engine.step()

        # Get rendered observation
        self.obs = self.engine.render()
        self._current_step += 1

        # Calculate reward and check termination
        terminated, reward = self.calculate_reward(self.obs)
        truncated = self._current_step >= self._max_episode_steps

        info: dict[str, Any] = {
            "status": "ok",
            "step": self._current_step,
            "physics_state": physics_state.model_dump() if physics_state else None,
        }

        return self.obs, reward, terminated, truncated, info


    def render(self) -> np.ndarray | None:
        """Render the current environment state.

        Returns:
            np.ndarray | None: Rendered image if render_mode is set.
        """
        logger.debug(f"Render mode: {self.render_mode}")

        if self.render_mode == "opencv":
            if self.obs is not None:
                show_image = cv.cvtColor(self.obs, cv.COLOR_RGB2BGR)
                cv.imshow("SimulationEnv", show_image)
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
            tuple[bool, float]: (terminated, reward).
        """
        # Placeholder implementation
        # TODO: Implement actual reward calculation based on goal_position
        return False, 0.0

    def set_random_goal(self) -> None:
        """Set a random goal position within the observation space."""
        height, width, _ = self.observation_space.shape
        x = self.np_random.integers(0, width)
        y = self.np_random.integers(0, height)
        self.goal_position = (x, y)
        logger.debug(f"Set random goal at: {self.goal_position}")

    def _draw_goal(self, image: np.ndarray) -> np.ndarray:
        """Draw goal indicator on the image.

        Args:
            image: Input image to draw on.

        Returns:
            np.ndarray: Image with goal indicator drawn.
        """
        if self.goal_position is None or image is None:
            return image

        # Draw a circle at the goal position
        result = image.copy()
        cv.circle(
            result,
            self.goal_position,
            radius=10,
            color=(0, 255, 0),  # Green
            thickness=2,
        )
        return result

    @property
    def config(self) -> SimulationConfig:
        """Get the current simulation configuration."""
        return self._config
