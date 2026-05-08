from collections.abc import Callable

from stable_baselines3.common.monitor import Monitor

from rlive_common import ActionSpaceType
from rlive_sim import (
    SimulationConfig,
    PhysicsConfig,
    RenderConfig,
    IntegratedConfig,
    PhysicsBackend,
    RenderBackend,
    IntegratedBackend,
    SimulationEnv,
)

def make_env_factory(
    env_fn: Callable,
    **env_kwargs,
) -> Callable[[], object]:
    """Create a factory function that builds environments with fixed parameters.

    Args:
        env_fn: Environment constructor function (e.g., make_sapiens_env)
        **env_kwargs: Keyword arguments to pass to env_fn
    Returns:
        Callable that creates environment instances with the given kwargs
    Example:
        factory = make_env_factory(make_sapiens_env, max_episode_steps=50)
        env = SubprocVecEnv([factory for _ in range(4)])
    """
    def _make_env():
        env = env_fn(**env_kwargs)
        return Monitor(env)
    return _make_env


def make_sapiens_env(
    max_episode_steps: int = 20,
    action_space_type: ActionSpaceType = ActionSpaceType.CARTESIAN,
) -> SimulationEnv:
    """Create a SAPIEN simulation environment with configurable parameters.

    Args:
        max_episode_steps: Maximum steps per episode (default: 20)
        action_space_type: Action space type (default: CARTESIAN)

    Returns:
        SimulationEnv: Configured environment instance
    """
    # Create simulation config using integrated backend
    sim_config = SimulationConfig(
        use_integrated=True,
        integrated=IntegratedConfig(
            backend=IntegratedBackend.SAPIEN,
            # width=640,
            # height=480,
        ),
    )

    # Create environment
    env = SimulationEnv(
        config=sim_config,
        max_episode_steps=max_episode_steps,
        action_space_type=action_space_type,
    )

    return env


def make_simple_env(
    max_episode_steps: int = 20,
    action_space_type: ActionSpaceType = ActionSpaceType.CARTESIAN,
) -> SimulationEnv:
    """Create a Simple (non-SAPIEN) simulation environment with configurable parameters.

    Args:
        max_episode_steps: Maximum steps per episode (default: 20)
        action_space_type: Action space type (default: CARTESIAN)

    Returns:
        SimulationEnv: Configured environment instance
    """
    # Configure physics backend
    physics_config = PhysicsConfig(
        backend=PhysicsBackend.SIMPLE,
        # extra={
        #     "box_width": 0.8,
        #     "box_height": 0.4,
        #     "ball_radius": 0.07,
        # }
    )

    # Configure render backend
    render_config = RenderConfig(
        backend=RenderBackend.OPENCV,
        # width=640,
        # height=480,
        # extra={
        #     "focal_length_mm": 35.0,
        #     "camera_position": [0.0, 0.0, 0.8],
        #     "camera_rotation": [0.0, 0.0, 0.0],
        #     "bg_color": (0, 0, 0),
        # }
    )

    # Create simulation config
    sim_config = SimulationConfig(
        use_integrated=False,
        physics=physics_config,
        render=render_config,
    )

    # Create environment
    env = SimulationEnv(
        config=sim_config,
        max_episode_steps=max_episode_steps,
        action_space_type=action_space_type,
    )
    return env