import os

from rlive_env import RemoteWorldEnv, ActionSpaceType
from rlive_common.core.hardware_config import WorldConfig
from rlive_common.core.enums import CameraType
from rlive_common.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    base_url = os.getenv("WORLD_BASE_URL", "http://127.0.0.1:8000")

    # Configure world settings using WorldConfig
    world_config = WorldConfig(
        camera_type=CameraType.WEBCAM,
        bolt_name="BP-D217",
        bolt_use_dummy=True,
    )

    env = RemoteWorldEnv(
        max_episode_steps=10,
        base_url=base_url,
        render_mode="opencv",
        action_space_type=ActionSpaceType.POLAR,
        world_config=world_config,
    )

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
