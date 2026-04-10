"""Simple Simulation Example

This example demonstrates how to use the SimulationEnv with the Simple physics
backend and OpenCV render backend.
"""
import cv2

from rlive_common import ActionSpaceType
from rlive_sim import (
    SimulationConfig,
    PhysicsConfig,
    RenderConfig,
    PhysicsBackend,
    RenderBackend,
    SimulationEnv,
)
from rlive_common.utils import get_logger

logger = get_logger(__name__)

# Number of episodes to run
NUM_EPISODES = 1

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
        obs, info = env.reset()
        env.render()
        logger.info(f"Episode {episode + 1}/{NUM_EPISODES} — reset -> obs={obs.shape if hasattr(obs, 'shape') else len(obs)}, info={info}")

        done = False
        step_count = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render(visualize=True)
            cv2.waitKey(500)  # Wait 50ms between frames
            done = terminated or truncated
            step_count += 1

            if step_count % 10 == 0:
                logger.info(f"Step {step_count}: action={action} reward={reward:.3f}")

    env.close()
    logger.info("Simulation completed!")


if __name__ == "__main__":
    main()
