import cv2
import gymnasium as gym

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
from rlive_sim.engine import SapienIntegratedEngine



def make_sapiens_env():

    # Create simulation config using integrated backend
    sim_config = SimulationConfig(
        use_integrated=True,
        integrated = IntegratedConfig(
            backend = IntegratedBackend.SAPIEN,
            # width=640,
            # height=480,
        ),
    )

    # Create environment


    env = SimulationEnv(
        config=sim_config,
        # render_mode="opencv",
        max_episode_steps=20,
        action_space_type=ActionSpaceType.CARTESIAN,
    )

    return env

def make_simple_env(num_envs):
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
        render_mode="opencv",
        max_episode_steps=10,
        action_space_type=ActionSpaceType.CARTESIAN,
    )
    env = gym.wrappers.frame_stack.FrameStack(env, num_stack=4)
    return env