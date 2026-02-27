"""Record environment interactions and save them as a d3rlpy MDPDataset.

This script connects to a RemoteWorldEnv, runs multiple episodes with random
actions, collects all (observation, action, reward, terminal) tuples and
persists them as an HDF5 file via ``d3rlpy.dataset.MDPDataset``.

Usage:
    python create_mdpdataset.py

The resulting dataset is saved to ``<rlive-example>/resources/dataset.h5``.

Environment Variables:
    WORLD_BASE_URL: Base URL for the world server (default: http://127.0.0.1:8000)

Configuration:
    - Robot: Sphero Bolt Plus (BP-D217)
    - Camera: Webcam
    - Episodes: 5
    - Max Steps per Episode: 10
"""

import os

import numpy as np
from d3rlpy.dataset import MDPDataset

from rlive_env.remote_env import RemoteWorldEnv, ActionSpaceType
from rlive_example.config import RESOURCES_DIR
from rlive_common.utils import get_logger

logger = get_logger(__name__)

# Number of episodes to record
NUM_EPISODES = 5


def main() -> None:
    """Record episodes and save as MDPDataset."""
    base_url = os.getenv("WORLD_BASE_URL", "http://127.0.0.1:8000")
    options = {
        "camera_type": "webcam",
        "robot_name": "BP-D217",
        "use_dummy": False,
    }
    env = RemoteWorldEnv(
        max_episode_steps=10,
        base_url=base_url,
        render_mode="opencv",
        action_space_type=ActionSpaceType.CARTESIAN,
        options=options,
    )

    # Collect transitions
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminals: list[int] = []

    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        env.render()
        logger.info(f"Episode {episode + 1}/{NUM_EPISODES} — reset -> obs={obs.shape}, info={info}")

        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            terminals.append(int(terminated or truncated))

            done = terminated or truncated
            obs = next_obs

            logger.info(
                f"action={action} reward={reward:.3f} "
                f"term={terminated} trunc={truncated} info={info}"
            )

    env.close()

    # Build MDPDataset
    dataset = MDPDataset(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        terminals=np.array(terminals),
    )

    logger.info(
        f"Collected {len(observations)} transitions across {NUM_EPISODES} episodes "
        f"({len(dataset.episodes)} dataset episodes)"
    )

    # Save to resources directory
    save_path = RESOURCES_DIR / "dataset.h5"
    with open(save_path, "w+b") as f:
        dataset.dump(f)
    logger.info(f"Dataset saved to {save_path}")


if __name__ == "__main__":
    main()
