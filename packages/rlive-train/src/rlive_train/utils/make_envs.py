"""Environment factory helpers for sim and real RL environments."""
from collections.abc import Callable

from stable_baselines3.common.monitor import Monitor

from rlive_common import ActionSpaceType
from rlive_common.core.hardware_config import CameraResolution, CameraType, WorldConfig
from rlive_env import RemoteWorldEnv
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
    fixed_goal: bool = False,
) -> SimulationEnv:
    """Create a SAPIEN simulation environment with configurable parameters.

    Args:
        max_episode_steps: Maximum steps per episode (default: 20)
        action_space_type: Action space type (default: CARTESIAN)
        fixed_goal: If True, the goal is always placed at the centre of the image (default: False)

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
        fixed_goal=fixed_goal,
    )

    return env


def make_simple_env(
    max_episode_steps: int = 20,
    action_space_type: ActionSpaceType = ActionSpaceType.CARTESIAN,
    fixed_goal: bool = False,
) -> SimulationEnv:
    """Create a Simple (non-SAPIEN) simulation environment with configurable parameters.

    Args:
        max_episode_steps: Maximum steps per episode (default: 20)
        action_space_type: Action space type (default: CARTESIAN)
        fixed_goal: If True, the goal is always placed at the centre of the image (default: False)

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
        fixed_goal=fixed_goal,
    )
    return env


def make_real_env(
    base_url: str = "http://127.0.0.1:8000",
    bolt_name: str = "BP-D217",
    camera_id: int = 1,
    max_episode_steps: int = 20,
    action_space_type: ActionSpaceType = ActionSpaceType.CARTESIAN,
) -> RemoteWorldEnv:
    """Create a real RemoteWorldEnv with configurable parameters.

    Resolution is fixed to 640×480 by the server-side world config to match
    the sim observation space, so a sim-trained CnnPolicy transfers directly.

    Args:
        base_url: Base URL of the world server (default: http://127.0.0.1:8000).
        bolt_name: Sphero Bolt device name (default: BP-D217).
        bolt_use_dummy: If True, use a dummy robot — no physical hardware needed (default: False).
        camera_id: OpenCV camera index (default: 1).
        max_episode_steps: Maximum steps per episode (default: 20).
        action_space_type: Action space type (default: CARTESIAN).

    Returns:
        RemoteWorldEnv: Configured environment instance.
    """
    world_config = WorldConfig(
        camera_type=CameraType.WEBCAM,
        camera_id=camera_id,
        camera_resolution=CameraResolution.RES_640x480,
        bolt_name=bolt_name,
        bolt_use_dummy=bolt_use_dummy,
    )

    return RemoteWorldEnv(
        max_episode_steps=max_episode_steps,
        base_url=base_url,
        action_space_type=action_space_type,
        world_config=world_config,
    )

