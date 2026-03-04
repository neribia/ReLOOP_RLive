"""Example demonstrating RemoteWorldEnv usage with a real robot or dummy robot.

This example shows how to:
1. Initialize a RemoteWorldEnv connected to a world server
2. Configure robot and camera settings using WorldConfig
3. Run an environment loop with random actions
4. Handle episode resets and termination conditions

The example uses a dummy robot by default for testing without hardware.
To use a real robot, set `use_dummy=False` and ensure the robot name is correct.

Usage:
    python rlive_env_example.py

Environment Variables:
    WORLD_BASE_URL: Base URL for the world server (default: http://127.0.0.1:8000)

Example Configuration:
    - Robot: Sphero Bolt Plus (BP-D217)
    - Camera: Webcam
    - Action Space: Cartesian coordinates
    - Max Steps: 10 per episode
    - Render Mode: OpenCV display
"""

import os

from rlive_env import RemoteWorldEnv, ActionSpaceType
from rlive_common.core.hardware_config import WorldConfig, CameraType, CameraResolution
from rlive_common.utils import get_logger

logger = get_logger(__name__)

# Number of episodes to run
NUM_EPISODES = 2

def main() -> None:
    base_url = os.getenv("WORLD_BASE_URL", "http://127.0.0.1:8000")

    # Configure world settings
    world_config = WorldConfig(
        camera_type=CameraType.WEBCAM,
        camera_id=1,
        camera_resolution=CameraResolution.RES_640x480,  # Can use custom tuple or CameraResolution.RES_640x480
        bolt_name="BP-D217",
        bolt_use_dummy=False
    )

    env = RemoteWorldEnv(
        max_episode_steps=10,
        base_url=base_url,
        render_mode="opencv",
        action_space_type=ActionSpaceType.CARTESIAN,
        world_config=world_config,
    )

    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        env.render()
        logger.info(f"Episode {episode + 1}/{NUM_EPISODES} — reset -> obs={obs.shape}, info={info}")

        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            done = terminated or truncated
            logger.info(f"action={action} reward={reward:.3f} term={terminated} trunc={truncated} info={info}")

    env.close()


if __name__ == "__main__":
    main()
