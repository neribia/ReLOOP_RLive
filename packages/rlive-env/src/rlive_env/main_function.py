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
    env = RemoteWorldEnv(base_url=base_url, render_mode="opencv", options=options)
    env.reset()

    obs, info = env.reset()
    logger.info(f"reset -> obs={obs.shape}, info={info}")

    for t in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        logger.info(f"t={t:02d} action={action} reward={reward:.3f} term={terminated} trunc={truncated} info={info}")
        if terminated or truncated:
            obs, info = env.reset()
            logger.info("Episode reset.")

    env.close()


if __name__ == "__main__":
    main()
