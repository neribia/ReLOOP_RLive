"""Example demonstrating RemoteWorldEnv usage with a real robot or dummy robot.

This example shows how to:
1. Initialize a RemoteWorldEnv connected to a world server
2. Configure robot and camera settings
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
    - Action Space: Polar coordinates (speed, direction)
    - Max Steps: 10 per episode
    - Render Mode: OpenCV display
"""

import os

from rlive_env import RemoteWorldEnv, ActionSpaceType
from rlive_common.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    base_url = os.getenv("WORLD_BASE_URL", "http://127.0.0.1:8000")
    options = {"camera_type": "webcam",
               "robot_name": "BP-D217",
               "use_dummy": True,
               }
    env = RemoteWorldEnv(
        max_episode_steps=10,
        base_url=base_url,
        render_mode="opencv",
        action_space_type=ActionSpaceType.POLAR,
        options=options)

    obs, info = env.reset()
    env.render()
    logger.info(f"reset -> obs={obs.shape}, info={info}")

    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        done = truncated
        logger.info(f"action={action} reward={reward:.3f} term={terminated} trunc={truncated} info={info}")
        if terminated or truncated:
            obs, info = env.reset()
            logger.info("Episode reset.")

    env.close()


if __name__ == "__main__":
    main()
