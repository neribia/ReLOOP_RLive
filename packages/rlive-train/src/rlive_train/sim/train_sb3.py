"""SB3 Training - SAPIEN Simulation

This example demonstrates how to train a Stable Baselines 3 agent
using the SimulationEnv with SAPIEN integrated backend.
"""
import os
import sys
if sys.platform != "win32":
    os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "graphics,utility,compute"
    os.environ["DISPLAY"] = ":0"
    os.environ["SAPIEN_NO_DISPLAY"] = "1"
    os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"

import argparse
from datetime import datetime
from pathlib import Path

import gymnasium
import wandb

# Spoof the gym module to suppress unmaintained gym warnings
# (SB3 optionally checks for gym, which is present due to d3rlpy)
sys.modules["gym"] = gymnasium

from stable_baselines3 import PPO
from rlive_train.utils.sb3_callbacks import build_wandb_eval_callbacks
from rlive_train.utils.sb3_env import EvalVecBackend, build_sim_env
from rlive_train.utils.utils import get_device
from rlive_train.config.config import LOGS_DIR, MODELS_DIR


def get_model_path(model_id):
    path = Path(MODELS_DIR) / model_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def train(
        lr=3e-4,
        ent_coef=0.05,
        max_episode_steps=20,
        total_timesteps=10_000,
        num_envs=1,
):
    from rlive_train.utils.make_envs import make_sapiens_env, make_env_factory

    env_args: dict = {
        "max_episode_steps": max_episode_steps,
    }

    factory = make_env_factory(env_fn=make_sapiens_env, **env_args)

    env_train = build_sim_env(env_factory=factory, n_stack=4, num_envs=num_envs, backend=EvalVecBackend.SUBPROC)
    env_eval = build_sim_env(env_factory=factory, n_stack=4, backend=EvalVecBackend.DUMMY)

    n_steps = 2 * max_episode_steps  # steps per env per rollout
    batch_size = n_steps * num_envs  # one mini-batch = full rollout buffer

    params = {
        "policy": "CnnPolicy",
        "ent_coef": ent_coef,
        "learning_rate": lr,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": 2,
    }

    run_name = f"PPO_Sapien_{datetime.now().strftime('%Y%m%d_%H%M%S')}_Laptop"

    with wandb.init(
            project="rlive-train",
            dir=str(LOGS_DIR),
            name=run_name,
            config=params,
            sync_tensorboard=True,
    )as wandb_logger:

        path_model = get_model_path(wandb_logger.id)
        path_logs = path_model / "logs"
        path_weights = path_model / "checkpoints"

        algorithm = PPO(
            env=env_train,
            device=get_device(),
            verbose=1,
            tensorboard_log=path_logs / "tensorboard",
            **params
		)

        callbacks = build_wandb_eval_callbacks(
            env_eval=env_eval,
            log_path=str(path_logs),
            save_path=str(path_weights),
            eval_freq=100,
            n_eval_episodes=5,

        )

        algorithm.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
        )



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()

    train(
        lr=args.lr,
        num_envs=args.num_envs,
    )


if __name__ == "__main__":
    main()
