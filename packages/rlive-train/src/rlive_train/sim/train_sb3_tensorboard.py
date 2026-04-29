"""SB3 Training - SAPIEN Simulation

This example demonstrates how to train a Stable Baselines 3 agent
using the SimulationEnv with SAPIEN integrated backend.
"""
import os
import sys
import json

import gymnasium

# Spoof the gym module to suppress unmaintained gym warnings
# (SB3 optionally checks for gym, which is present due to d3rlpy)
sys.modules["gym"] = gymnasium

from stable_baselines3 import PPO

from datetime import datetime
from rlive_train.config.config import LOGS_DIR
from rlive_common.utils import get_logger
from rlive_train.utils.sb3_callbacks import build_eval_callback
from rlive_train.utils.sb3_env import EvalVecBackend, build_sim_eval_env, build_sim_train_env

logger = get_logger(__name__)


def main() -> None:
    # Create a unique run folder based on time
    run_name = f"PPO_Sapien_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = LOGS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_log = str(run_dir / "tensorboard")

    env = build_sim_train_env(num_envs=2, n_stack=4)
    eval_env = build_sim_eval_env(n_stack=4, backend=EvalVecBackend.DUMMY)

    logger.info("Saving hyperparameters...")

    # Define hyperparameters explicitly to resolve static type hinting issues
    policy = "CnnPolicy"
    n_steps = 2 * 20 * 2  # Ca 5 * erfahrung sammeln
    batch_size = 10       # Kleinere Häppchen für die CPU
    n_epochs = 2          # Nur 2-mal "durchlesen" statt 10-mal
    ent_coef = 0.1
    learning_rate = 3e-4
    total_timesteps = 1_000_000

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

    logger.info(f"Initializing PPO Model with {policy} (assuming Box image obs) ...")
    model = PPO(
        policy=policy,
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
    eval_callback = build_eval_callback(
        env_eval=eval_env,
        log_path=str(run_dir),
        save_path=best_model_dir,
        eval_freq=200,
        n_eval_episodes=5,
    )

    logger.info(f"Starting training for {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name="PPO_Sapien",
        # progress_bar=True,
    )

    model_path = str(run_dir / "last_model")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}.zip")

    logger.info("Evaluating the trained model for 1 episode...")
    obs = eval_env.reset()
    done = False
    step_count = 0
    while not done.all() if hasattr(done, 'all') else not done:
        # Predict action based on the observation
        action, _states = model.predict(obs, deterministic=True) # type: ignore
        obs, reward, done, info = eval_env.step(action)
        step_count += 1

    logger.info(f"Evaluation completed in {step_count} steps.")
    logger.info(f"To view tensorboard logs, run: tensorboard --logdir {tensorboard_log}")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()

