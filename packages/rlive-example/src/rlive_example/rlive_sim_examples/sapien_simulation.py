"""SAPIEN Simulation Example

This example demonstrates how to use the SimulationEnv with SAPIEN integrated backend.
It shows how to use 3 different robot loading options:
1. Rigid Body - Programmatically built sphere (default, fastest)
2. URDF - Load from URDF file (resources/sapian/sphere_robot.urdf)
3. GLB - Load from GLB/GLTF file (resources/sapian/bolt_shell.glb)

Run with different ROBOT_TYPE values to test each option.
"""
import cv2

from rlive_common import ActionSpaceType
from rlive_sim import (
    SimulationConfig,
    IntegratedConfig,
    IntegratedBackend,
    SimulationEnv,
)
from rlive_common.utils import get_logger

logger = get_logger(__name__)

# Number of episodes to run
NUM_EPISODES = 1


def main() -> None:
    """Run SAPIEN simulation with configurable robot loading."""

    # Configure integrated backend with robot loading option
    integrated_config = IntegratedConfig(
        backend=IntegratedBackend.SAPIEN,
        width=640,
        height=480,
        extra={
            # # Controller configuration
            # "max_speed_ms": 0.5,
            # "acceleration_time": 0.3,
            # "deceleration_time": 0.2,
            # "controller_kp": 0.1,
            # "controller_kd": 0.1,
            
            # Viewer configuration
            "use_viewer": True,
        },
    )
    
    # Create simulation config using integrated backend
    sim_config = SimulationConfig(
        use_integrated=True,
        integrated=integrated_config,
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
        logger.info(f"Episode {episode + 1}/{NUM_EPISODES} — reset -> obs={obs.shape}, info={info}")

        done = False
        step_count = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render(visualize=True)
            cv2.waitKey(500) # Wait 1000ms between frames
            done = terminated or truncated
            step_count += 1
            logger.info(f"Step {step_count}: action={action} reward={reward:.3f} term={terminated} trunc={truncated}")

    env.close()
    logger.info("Simulation completed!")


if __name__ == "__main__":
    main()
