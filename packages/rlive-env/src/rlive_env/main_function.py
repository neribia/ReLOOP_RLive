import os

from rlive_env.remote_env import RemoteWorldEnv
from rlive_common.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    base_url = os.getenv("WORLD_BASE_URL", "http://127.0.0.1:8000")
    env = RemoteWorldEnv(base_url=base_url)

    obs, info = env.reset()
    logger.info(f"reset -> obs={obs}, info={info}")

    for t in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        logger.info(f"t={t:02d} action={action} reward={reward:.3f} term={terminated} trunc={truncated} info={info}")
        if terminated or truncated:
            obs, info = env.reset()
            logger.info("Episode reset.")

    env.close()


if __name__ == "__main__":
    main()
