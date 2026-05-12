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
from rlive_train.utils.make_envs import make_sapiens_env, make_simple_env, make_env_factory

def get_model_path(model_id):
    path = Path(MODELS_DIR) / model_id
    path.mkdir(parents=True, exist_ok=True)
    return path


ENV_CHOICES = ("sapien", "simple")


def int_or_sci(value: str) -> int:
    """Accept plain integers and scientific notation (e.g. 1e6)."""
    return int(float(value))

# All keyword arguments accepted by PPO.__init__ (excluding env/device/verbose/tensorboard_log
# which are passed explicitly).
PPO_KEYS: frozenset[str] = frozenset({
    "policy",
    "learning_rate",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "clip_range",
    "clip_range_vf",
    "normalize_advantage",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "use_sde",
    "sde_sample_freq",
    "rollout_buffer_class",
    "rollout_buffer_kwargs",
    "target_kl",
    "stats_window_size",
    "policy_kwargs",
    "seed",
})


def train(
        env_type: str = "sapien",
        lr: float = 3e-4,
        ent_coef: float = 0.05,
        log_std_init: float = 0.0,
        max_episode_steps: int = 20,
        total_timesteps: int = 1_000_000,
        num_envs: int = 1,
        eval_freq: int = 1_000,
        n_eval_episodes: int = 5,
):


    env_fn = make_sapiens_env if env_type == "sapien" else make_simple_env
    env_args: dict = {
        "max_episode_steps": max_episode_steps,
    }

    factory = make_env_factory(env_fn=env_fn, **env_args)

    env_train = build_sim_env(env_factory=factory, n_stack=4, num_envs=num_envs, backend=EvalVecBackend.SUBPROC)
    env_eval = build_sim_env(env_factory=factory, n_stack=4, backend=EvalVecBackend.DUMMY)

    n_steps = 2 * max_episode_steps  # steps per env per rollout
    batch_size = n_steps * num_envs  # one mini-batch = full rollout buffer

    params = {
        "policy": "CnnPolicy",
        "ent_coef": ent_coef,
        "learning_rate": lr,
        "policy_kwargs": {"log_std_init": log_std_init},
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": 4,
        # extra run metadata logged to W&B
        "env_type": env_type,
        "max_episode_steps": max_episode_steps,
        "total_timesteps": total_timesteps,
        "num_envs": num_envs,
        "eval_freq": eval_freq,
        "n_eval_episodes": n_eval_episodes,
    }

    run_name = f"{env_type.capitalize()}_lr{lr:.0e}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with wandb.init(
            project="rlive-train",
            dir=str(LOGS_DIR),
            name=run_name,
            config=params,
            sync_tensorboard=True,
    ) as wandb_logger:

        path_model = get_model_path(wandb_logger.id)
        path_logs = path_model / "logs"
        path_weights = path_model / "checkpoints"

        # Keep only keys that PPO.__init__ actually accepts
        ppo_params = {k: v for k, v in params.items() if k in PPO_KEYS}

        algorithm = PPO(
            env=env_train,
            device=get_device(),
            verbose=1,
            tensorboard_log=path_logs / "tensorboard",
            **ppo_params,
        )

        callbacks = build_wandb_eval_callbacks(
            env_eval=env_eval,
            log_path=str(path_logs),
            save_path=str(path_weights),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
        )

        algorithm.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
        )

    env_train.close()
    env_eval.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an SB3 PPO agent in simulation.")
    parser.add_argument(
        "--env",
        choices=ENV_CHOICES,
        default="sapien",
        help="Simulation backend to use (default: sapien)",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.05,
        help="Entropy coefficient (default: 0.05)",
    )
    parser.add_argument(
        "--log-std-init", type=float, default=0.0,
        help="Initial log std for action distribution (default: 0.0)",
    )
    parser.add_argument(
        "--num-envs", type=int, default=4,
        help="Number of parallel envs (default: 4)",
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=20,
        help="Max steps per episode (default: 20)",
    )
    parser.add_argument(
        "--total-timesteps", type=int_or_sci, default=1_000_000,
        help="Total training timesteps, accepts e.g. 1e6 (default: 1_000_000)",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=1_000,
        help="Eval frequency in timesteps (default: 1_000)",
    )
    parser.add_argument(
        "--n-eval-episodes", type=int, default=5,
        help="Number of eval episodes (default: 5)",
    )
    args = parser.parse_args()

    train(
        env_type=args.env,
        lr=args.lr,
        ent_coef=args.ent_coef,
        log_std_init=args.log_std_init,
        num_envs=args.num_envs,
        max_episode_steps=args.max_episode_steps,
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
    )


if __name__ == "__main__":
    main()
