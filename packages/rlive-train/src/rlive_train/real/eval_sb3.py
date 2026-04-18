"""SB3 Evaluation - Real Environment

This example demonstrates how to evaluate an existing Stable Baselines 3 agent
using the RemoteWorldEnv with a real robot or dummy robot backend.
"""
import os
import sys
import cv2
import gymnasium as gym

# Spoof the gym module to suppress unmaintained gym warnings
sys.modules["gym"] = gym

from stable_baselines3 import PPO

from rlive_env import RemoteWorldEnv, ActionSpaceType
from rlive_common.core.hardware_config import WorldConfig, CameraType, CameraResolution
from rlive_common.utils import get_logger

from rlive_train.config.config import LOGS_DIR

logger = get_logger(__name__)

# Replace with the actual run folder name you want to evaluate
RUN_ID = "PPO_Real_YYYYMMDD_HHMMSS"

def main() -> None:
    base_url = os.getenv("WORLD_BASE_URL", "http://127.0.0.1:8000")

    logger.info("Setting up RemoteWorldEnv for evaluation...")
    world_config = WorldConfig(
        camera_type=CameraType.WEBCAM,
        camera_id=1,
        camera_resolution=CameraResolution.RES_640x480,
        bolt_name="BP-D217",
        # Keep use_dummy=True for testing, change to False for real robot
        bolt_use_dummy=True
    )

    env = RemoteWorldEnv(
        max_episode_steps=500,
        base_url=base_url,
        render_mode="opencv", # Used for displaying the camera stream bounding box
        action_space_type=ActionSpaceType.CARTESIAN,
        world_config=world_config,
    )

    model_path = LOGS_DIR / RUN_ID / "best_model" / "best_model.zip"

    logger.info(f"Loading model from {model_path}...")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model from '{model_path}'. Ensure you ran 'train_sb3.py' first and set RUN_ID.")
        env.close()
        raise e

    logger.info("Evaluating the trained model for 5 episodes...")
    for episode in range(5):
        obs, info = env.reset()
        env.render()

        done = False
        step_count = 0
        episode_reward = 0.0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # Show the webcam frame and actions
            env.render()

            done = terminated or truncated
            step_count += 1

        logger.info(f"Episode {episode + 1}/5 completed - Steps: {step_count}, Reward: {episode_reward:.3f}")

    env.close()

if __name__ == "__main__":
    main()
