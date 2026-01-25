import os

from rlive_env.remote_env import RemoteWorldEnv
from rlive_common.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    base_url = os.getenv("WORLD_BASE_URL", "http://127.0.0.1:8000")
    options = {"camera_type": "dummy",
               "robot_name": "BP-D217",
               "use_dummy": False,
               }
    env = RemoteWorldEnv(max_episode_steps=10, base_url=base_url, render_mode="opencv", options=options)

    obs, info = env.reset()
    env.render()
    logger.info(f"reset -> obs={obs.shape}, info={info}")

    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        done =  truncated
        logger.info(f"action={action} reward={reward:.3f} term={terminated} trunc={truncated} info={info}")
        if terminated or truncated:
            obs, info = env.reset()
            logger.info("Episode reset.")

    env.close()


if __name__ == "__main__":
    main()
