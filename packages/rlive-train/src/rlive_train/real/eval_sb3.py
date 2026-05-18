"""SB3 Evaluation - Real Environment.

Evaluates an existing PPO model via RemoteWorldEnv and logs results to W&B.

Usage
-----
    python eval_sb3.py --model-path /path/to/best_model.zip
    python eval_sb3.py --model-path /path/to/best_model.zip --bolt-use-dummy --n-episodes 10

The eval W&B run is placed in the same project ("rlive-train") and the same
*group* as the training run so both appear side-by-side in the W&B UI.
The group name is inferred automatically from the model path
(grandparent folder == W&B run-id used during training) but can be
overridden with --run-group.
"""
import os
import sys
from pathlib import Path

import gymnasium as gym

# Spoof the gym module to suppress unmaintained gym warnings
sys.modules["gym"] = gym

import argparse  # noqa: E402
import numpy as np  # noqa: E402
import wandb  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.evaluation import evaluate_policy  # noqa: E402

from rlive_common.utils import get_logger  # noqa: E402
from rlive_train.utils.make_envs import make_env_factory, make_real_env  # noqa: E402
from rlive_train.utils.sb3_env import EvalVecBackend, build_sim_env  # noqa: E402

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate(  # noqa: PLR0913
    model_path: str | Path,
    base_url: str = "http://127.0.0.1:8000",
    bolt_name: str = "BP-D217",
    bolt_use_dummy: bool = True,
    camera_id: int = 1,
    n_episodes: int = 10,
    max_episode_steps: int = 20,
    n_stack: int = 4,
    run_group: str | None = None,
    run_name: str | None = None,
) -> None:
    """Evaluate a trained PPO model on the real RemoteWorldEnv and log to W&B.

    The environment is wrapped with the same DummyVecEnv → VecFrameStack →
    VecTransposeImage pipeline as used during sim training, so a sim-trained
    CnnPolicy model can be loaded and evaluated here directly (sim2real).

    Args:
        model_path: Path to the model .zip file.
        base_url: Base URL of the world server.
        bolt_name: Sphero Bolt device name.
        bolt_use_dummy: If True, use a dummy robot (no physical hardware needed).
        camera_id: OpenCV camera index.
        n_episodes: Number of deterministic evaluation episodes.
        max_episode_steps: Max steps per episode (must match training).
        n_stack: Number of frames to stack — must match the model (default: 4).
        run_group: W&B group name linking this eval to its training run.
            When None, inferred from the grandparent folder of model_path.
        run_name: Custom W&B run name. When None, defaults to
            ``eval_<run_group>_real``.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return

    # Infer W&B group from path when not provided explicitly.
    # Training stores weights at: MODELS_DIR / <wandb_run_id> / checkpoints / *.zip
    # So grandparent folder == the W&B run id used during training.
    if run_group is None:
        run_group = model_path.parent.parent.name
        logger.info(f"Inferred run group from model path: '{run_group}'")

    if run_name is None:
        run_name = f"eval_{run_group}_real"

    # ------------------------------------------------------------------
    # Build evaluation environment — same wrapper pipeline as sim so
    # a sim-trained CnnPolicy model loads without shape mismatches.
    # ------------------------------------------------------------------
    factory = make_env_factory(
        env_fn=make_real_env,
        base_url=base_url,
        bolt_name=bolt_name,
        bolt_use_dummy=bolt_use_dummy,
        camera_id=camera_id,
        max_episode_steps=max_episode_steps,
    )
    # DummyVecEnv → VecFrameStack(n_stack) → VecTransposeImage
    env = build_sim_env(env_factory=factory, n_stack=n_stack, backend=EvalVecBackend.DUMMY)

    # ------------------------------------------------------------------
    # W&B run — linked to the training run via group
    # ------------------------------------------------------------------
    eval_params = {
        "model_path": str(model_path),
        "base_url": base_url,
        "bolt_name": bolt_name,
        "bolt_use_dummy": bolt_use_dummy,
        "n_episodes": n_episodes,
        "max_episode_steps": max_episode_steps,
        "n_stack": n_stack,
    }

    with wandb.init(
        project="rlive-train",
        job_type="eval",
        group=run_group,
        name=run_name,
        config=eval_params,
    ) as run:
        logger.info(f"W&B run: {run.url}")

        # ------------------------------------------------------------------
        # Load model
        # ------------------------------------------------------------------
        logger.info(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=env)

        # ------------------------------------------------------------------
        # Run evaluation — collect per-episode rewards & lengths
        # ------------------------------------------------------------------
        logger.info(f"Evaluating for {n_episodes} episodes...")

        episode_rewards, episode_lengths = evaluate_policy(
            model,
            env,
            n_eval_episodes=n_episodes,
            deterministic=True,
            return_episode_rewards=True,
        )

        # Log per-episode metrics as a W&B table
        table = wandb.Table(columns=["episode", "reward", "length"])
        for i, (reward, length) in enumerate(zip(episode_rewards, episode_lengths)):
            table.add_data(i + 1, reward, length)
            wandb.log({"eval/episode_reward": reward, "eval/episode_length": length, "episode": i + 1})

        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        mean_length = float(np.mean(episode_lengths))

        # Summary metrics (shown prominently in the W&B run overview)
        run.summary["eval/mean_reward"] = mean_reward
        run.summary["eval/std_reward"] = std_reward
        run.summary["eval/mean_episode_length"] = mean_length

        wandb.log({"eval/episodes_table": table})

        logger.info(
            f"Evaluation complete — "
            f"mean reward: {mean_reward:.3f} ± {std_reward:.3f}, "
            f"mean length: {mean_length:.1f} steps"
        )

    env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the real-environment evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SB3 PPO agent on the real RemoteWorldEnv and log to W&B."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model .zip file, e.g. resources/models/<wandb_run_id>/checkpoints/best_model.zip",
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
        "--n-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=20,
        help="Max steps per episode — must match the trained model (default: 20)",
    )
    parser.add_argument(
        "--n-stack",
        type=int,
        default=4,
        help="Number of frames to stack — must match the model (default: 4)",
    )
    parser.add_argument(
        "--run-group",
        type=str,
        default=None,
        help=(
            "W&B group name to link this eval run to its training run. "
            "Defaults to the grandparent folder name of --model-path "
            "(i.e. the W&B run-id used during training)."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Custom W&B run name (default: eval_<run-group>_real). "
            "Tip: use something like 'PPO_abc12345_real_dummy' to identify "
            "the model and target env at a glance."
        ),
    )
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        base_url=args.base_url,
        bolt_name=args.bolt_name,
        bolt_use_dummy=args.bolt_use_dummy,
        camera_id=args.camera_id,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_episode_steps,
        n_stack=args.n_stack,
        run_group=args.run_group,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
