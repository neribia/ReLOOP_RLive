"""Simple Simulation Example

This example demonstrates how to use the SimulationEnv with the Simple physics
backend and OpenCV render backend.
"""
import cv2 as cv

from rlive_common import ActionSpaceType
from rlive_sim import (
    SimulationConfig,
    PhysicsConfig,
    RenderConfig,
    PhysicsBackend,
    RenderBackend,
    SimulationEnv,
    PhysicsState,
)
from rlive_common.utils import get_logger

logger = get_logger(__name__)

# Number of episodes to run (episode 0 = random/default spawn, episode 1+ = fixed initial_state)
NUM_EPISODES = 2

def main() -> None:
    """Run simple simulation with separate physics and render backends."""

    logger.info("Starting simple simulation with separate backends")

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

    for episode in range(NUM_EPISODES):
        # Episode 0: default spawn (ball starts at centre of the box)
        # Episode 1+: fixed initial state (ball placed at top-left quadrant, facing 45°)
        reset_options = None
        if episode > 0:
            reset_options = {
                "initial_state": PhysicsState(
                    position=[-0.1, 0.1, 0.0],  # x, y, z in metres (2-D engine ignores z)
                    velocity=[0.0, 0.0, 0.0],
                    rotation=[0.0, 0.0, 45.0],  # roll, pitch, yaw in degrees
                    angular_velocity=[0.0, 0.0, 0.0],
                )
            }

        obs, info = env.reset(options=reset_options)
        env.render()
        logger.info(f"Episode {episode + 1}/{NUM_EPISODES} — reset -> obs={obs.shape if hasattr(obs, 'shape') else len(obs)}, info={info}")

        done = False
        step_count = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render(visualize=True)
            cv.waitKey(500)  # Wait 50ms between frames
            done = terminated or truncated
            step_count += 1

            if step_count % 10 == 0:
                logger.info(f"Step {step_count}: action={action} reward={reward:.3f}")

    env.close()
    logger.info("Simulation completed!")


if __name__ == "__main__":
    main()
