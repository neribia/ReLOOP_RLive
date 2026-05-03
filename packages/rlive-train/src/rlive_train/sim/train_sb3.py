"""SB3 Training - SAPIEN Simulation

This example demonstrates how to train a Stable Baselines 3 agent
using the SimulationEnv with SAPIEN integrated backend.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import gymnasium
import torch as th
import wandb
from torch.backends.mkl import verbose

# Spoof the gym module to suppress unmaintained gym warnings
# (SB3 optionally checks for gym, which is present due to d3rlpy)
sys.modules["gym"] = gymnasium

from stable_baselines3 import PPO
from rlive_train.utils.sb3_callbacks import build_wandb_eval_callbacks
from rlive_train.utils.sb3_env import EvalVecBackend, build_sim_eval_env, build_sim_train_env
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
        total_timesteps=1_000,
        num_envs=1,
        eval_backend=EvalVecBackend.SUBPROC,
):
    from rlive_train.utils.make_envs import make_sapiens_env, make_env_factory

    env_args: dict = {
        "max_episode_steps": max_episode_steps,
    }

    factory = make_env_factory(env_fn=make_sapiens_env, **env_args)

    env_train = build_sim_train_env(env_factory=factory, n_stack=4, num_envs=num_envs)
    env_eval = build_sim_eval_env(env_factory=factory, n_stack=4, backend=eval_backend)

    params = {
        "policy": "CnnPolicy",
        "ent_coef": ent_coef,
        "learning_rate": lr,
        "n_steps": 2 * max_episode_steps,
        # "batch_size": 64, # We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.
        "n_epochs": 2,
    }

    run_name = f"PPO_Sapien_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument(
        "--eval-backend",
        type=str,
        choices=[backend.value for backend in EvalVecBackend],
        default=EvalVecBackend.SUBPROC.value,
    )
    args = parser.parse_args()

    train(
        lr=args.lr,
        num_envs=args.num_envs,
        eval_backend=EvalVecBackend(args.eval_backend),
    )


if __name__ == "__main__":
    main()
