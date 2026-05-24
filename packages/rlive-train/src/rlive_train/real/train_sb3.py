"""SB3 Training - Real Environment.

Trains a Stable Baselines 3 PPO agent using the RemoteWorldEnv with a real
robot or dummy robot backend. Results are logged to Weights & Biases.

Usage
-----
    python train_sb3.py
    python train_sb3.py --bolt-use-dummy --total-timesteps 50000
"""
import os
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym

# Spoof the gym module to suppress unmaintained gym warnings
sys.modules["gym"] = gym

import argparse  # noqa: E402
import wandb  # noqa: E402

from stable_baselines3 import PPO  # noqa: E402

from rlive_common.utils import get_logger  # noqa: E402
from rlive_train.config.config import LOGS_DIR, MODELS_DIR  # noqa: E402
from rlive_train.utils.make_envs import make_env_factory, make_real_env  # noqa: E402
from rlive_train.utils.sb3_callbacks import build_wandb_eval_callbacks  # noqa: E402
from rlive_train.utils.sb3_env import EvalVecBackend, build_sim_env  # noqa: E402
from rlive_train.utils.utils import get_device  # noqa: E402

logger = get_logger(__name__)

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


def get_model_path(model_id: str) -> Path:
    """Return (and create) the model directory for the given W&B run id."""
    path = Path(MODELS_DIR) / model_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def int_or_sci(value: str) -> int:
    """Accept plain integers and scientific notation (e.g. 1e6)."""
    return int(float(value))


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train(  # noqa: PLR0913
    base_url: str = "http://127.0.0.1:8000",
    bolt_name: str = "BP-D217",
    bolt_use_dummy: bool = True,
    camera_id: int = 1,
    lr: float = 3e-4,
    ent_coef: float = 0.01,
    log_std_init: float = 0.0,
    max_episode_steps: int = 20,
    total_timesteps: int = 10_000,
    n_epochs: int = 10,
    eval_freq: int = 200,
    n_eval_episodes: int = 5,
    n_stack: int = 4,
) -> None:
    """Train a PPO agent on the real RemoteWorldEnv and log to W&B.

    The environment is wrapped with the same DummyVecEnv → VecFrameStack →
    VecTransposeImage pipeline as the sim training so that a sim-trained
    CnnPolicy model can be fine-tuned or evaluated here directly (sim2real).

    Args:
        base_url: Base URL of the world server.
        bolt_name: Sphero Bolt device name.
        bolt_use_dummy: If True, use a dummy robot (no physical hardware needed).
        camera_id: OpenCV camera index.
        lr: PPO learning rate.
        ent_coef: PPO entropy coefficient.
        log_std_init: Initial log std for the action distribution.
        max_episode_steps: Maximum steps per episode.
        total_timesteps: Total training timesteps.
        n_epochs: PPO epochs per update.
        eval_freq: Evaluation frequency in timesteps.
        n_eval_episodes: Number of episodes per evaluation.
        n_stack: Number of frames to stack — must match the sim model (default: 4).
    """
    env_args = {
        "base_url": base_url,
        "bolt_name": bolt_name,
        "bolt_use_dummy": bolt_use_dummy,
        "camera_id": camera_id,
        "max_episode_steps": max_episode_steps,
    }
    factory = make_env_factory(env_fn=make_real_env, **env_args)

    # Wrap with the same pipeline as sim so CnnPolicy models are compatible.
    env_train = build_sim_env(env_factory=factory, n_stack=n_stack, backend=EvalVecBackend.DUMMY)
    env_eval  = build_sim_env(env_factory=factory, n_stack=n_stack, backend=EvalVecBackend.DUMMY)

    # Derive n_steps / batch_size the same way as sim/train_sb3.py
    n_steps = 2 * max_episode_steps      # steps per env per rollout
    batch_size = n_steps                  # num_envs=1 for real

    params = {
        "policy": "CnnPolicy",
        "learning_rate": lr,
        "ent_coef": ent_coef,
        "policy_kwargs": {"log_std_init": log_std_init},
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        # extra run metadata logged to W&B
        "bolt_name": bolt_name,
        "bolt_use_dummy": bolt_use_dummy,
        "max_episode_steps": max_episode_steps,
        "total_timesteps": total_timesteps,
        "eval_freq": eval_freq,
        "n_eval_episodes": n_eval_episodes,
        "n_stack": n_stack,
    }

    run_name = f"Real_lr{lr:.0e}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

        ppo_params = {k: v for k, v in params.items() if k in PPO_KEYS}

        logger.info(f"Initializing PPO with CnnPolicy — run '{run_name}'")
        algorithm = PPO(
            env=env_train,
            device=get_device(),
            verbose=1,
            tensorboard_log=str(path_logs / "tensorboard"),
            **ppo_params,
        )

        callbacks = build_wandb_eval_callbacks(
            env_eval=env_eval,
            log_path=str(path_logs),
            save_path=str(path_weights),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
        )

        logger.info(f"Starting training for {total_timesteps:,} timesteps...")
        algorithm.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
        )

    env_train.close()
    env_eval.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the real-environment training CLI."""
    parser = argparse.ArgumentParser(
        description="Train an SB3 PPO agent on the real RemoteWorldEnv."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("WORLD_BASE_URL", "http://127.0.0.1:8000"),
        help="World server base URL (default: $WORLD_BASE_URL or http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--bolt-name",
        type=str,
        default="BP-D217",
        help="Sphero Bolt device name (default: BP-D217)",
    )
    parser.add_argument(
        "--bolt-use-dummy",
        action="store_true",
        default=False,
        help="Use a dummy robot instead of real hardware (default: False)",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=1,
        help="OpenCV camera index (default: 1)",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.01,
        help="Entropy coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--log-std-init", type=float, default=0.0,
        help="Initial log std for action distribution (default: 0.0)",
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=20,
        help="Max steps per episode (default: 20)",
    )
    parser.add_argument(
        "--total-timesteps", type=int_or_sci, default=10_000,
        help="Total training timesteps, accepts e.g. 1e5 (default: 10_000)",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=10,
        help="PPO epochs per update (default: 10)",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=200,
        help="Eval frequency in timesteps (default: 200)",
    )
    parser.add_argument(
        "--n-eval-episodes", type=int, default=5,
        help="Number of eval episodes (default: 5)",
    )
    parser.add_argument(
        "--n-stack", type=int, default=4,
        help="Number of frames to stack — must match the sim model (default: 4)",
    )
    args = parser.parse_args()

    train(
        base_url=args.base_url,
        bolt_name=args.bolt_name,
        bolt_use_dummy=args.bolt_use_dummy,
        camera_id=args.camera_id,
        lr=args.lr,
        ent_coef=args.ent_coef,
        log_std_init=args.log_std_init,
        max_episode_steps=args.max_episode_steps,
        total_timesteps=args.total_timesteps,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        n_stack=args.n_stack,
    )


if __name__ == "__main__":
    main()

