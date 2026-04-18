"""SB3 Training - Real Environment

This example demonstrates how to train a Stable Baselines 3 agent
using the RemoteWorldEnv with a real robot or dummy robot backend.
"""
import os
import sys
import json
from datetime import datetime

import gymnasium as gym

# Spoof the gym module to suppress unmaintained gym warnings
sys.modules["gym"] = gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from rlive_env import RemoteWorldEnv, ActionSpaceType
from rlive_common.core.hardware_config import WorldConfig, CameraType, CameraResolution
from rlive_common.utils import get_logger

from rlive_train.config.config import LOGS_DIR

logger = get_logger(__name__)

def main() -> None:
    base_url = os.getenv("WORLD_BASE_URL", "http://127.0.0.1:8000")

    logger.info("Setting up RemoteWorldEnv for training...")
    world_config = WorldConfig(
        camera_type=CameraType.WEBCAM,
        camera_id=1,
        camera_resolution=CameraResolution.RES_640x480,
        bolt_name="BP-D217",
        # Keep use_dummy=True for testing, change to False for real robot
        bolt_use_dummy=True
    )

    def make_env():
        return RemoteWorldEnv(
            max_episode_steps=500,
            base_url=base_url,
            render_mode="opencv", # Used for displaying the camera stream bounding box
            action_space_type=ActionSpaceType.CARTESIAN,
            world_config=world_config,
        )

    env = make_env()
    eval_env = make_env()

    run_name = f"PPO_Real_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = LOGS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_log = str(run_dir / "tensorboard")

    logger.info("Saving hyperparameters...")

    # Define hyperparameters explicitly to resolve static type hinting issues
    policy = "MlpPolicy"
    n_steps = 2 * 20 * 2
    batch_size = 10
    n_epochs = 2
    ent_coef = 0.1
    learning_rate = 3e-4
    total_timesteps = 10_000

    hyperparams = {
        "policy": policy,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "ent_coef": ent_coef,
        "learning_rate": learning_rate,
        "total_timesteps": total_timesteps,
    }

    with open(run_dir / "hyperparams.json", "w") as f:
        json.dump(hyperparams, f, indent=4)

    logger.info(f"Initializing PPO Model with {policy} ...")
    model = PPO(
        policy=policy, # Update to CnnPolicy if using image observations
        env=env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        verbose=1,
        tensorboard_log=tensorboard_log,
        ent_coef=ent_coef,
        learning_rate=learning_rate
    )

    best_model_dir = str(run_dir / "best_model")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=str(run_dir),
        eval_freq=200,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    logger.info(f"Starting training for {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name="PPO_Real",
    )

    model_path = str(run_dir / "ppo_real_env_model")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}.zip")

    logger.info(f"To view tensorboard logs, run: tensorboard --logdir {tensorboard_log}")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()

