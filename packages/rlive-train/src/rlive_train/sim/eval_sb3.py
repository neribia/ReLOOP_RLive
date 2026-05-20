"""SB3 Evaluation - SAPIEN / Simple Simulation.

Evaluates an existing Stable Baselines 3 PPO model and logs results to
Weights & Biases.

Usage
-----
    python eval_sb3.py --model-path /path/to/best_model.zip
    python eval_sb3.py --model-path /path/to/best_model.zip --env simple --n-episodes 10
"""
import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import wandb

# Spoof the gym module to suppress unmaintained gym warnings
sys.modules["gym"] = gym

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.evaluation import evaluate_policy  # noqa: E402

from rlive_common.utils import get_logger  # noqa: E402
from rlive_train.utils.make_envs import make_env_factory, make_sapiens_env, make_simple_env  # noqa: E402
from rlive_train.utils.sb3_env import EvalVecBackend, build_sim_env  # noqa: E402

logger = get_logger(__name__)

ENV_CHOICES = ("sapien", "simple")


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate(  # noqa: PLR0913
    model_path: str | Path,
    env_type: str = "sapien",
    n_episodes: int = 10,
    max_episode_steps: int = 50,
    fixed_goal: bool = False,
    run_name: str | None = None,
) -> None:
    """Evaluate a trained PPO model and log results to Weights & Biases.

    Args:
        model_path: Path to the model .zip file.
        env_type: Simulation backend — 'sapien' or 'simple'.
        n_episodes: Number of deterministic evaluation episodes.
        max_episode_steps: Max steps per episode (must match training).
        fixed_goal: Whether the goal is fixed at the image centre.
        run_name: Custom W&B run name. When None, defaults to
            ``eval_<env_type>``.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return

    if run_name is None:
        run_name = f"eval_{env_type}"

    # ------------------------------------------------------------------
    # Build evaluation environment (identical setup to training)
    # ------------------------------------------------------------------
    env_fn = make_sapiens_env if env_type == "sapien" else make_simple_env
    factory = make_env_factory(env_fn=env_fn, max_episode_steps=max_episode_steps, fixed_goal=fixed_goal)
    env = build_sim_env(env_factory=factory, n_stack=4, backend=EvalVecBackend.DUMMY)

    # ------------------------------------------------------------------
    # W&B run — linked to the training run via group
    # ------------------------------------------------------------------
    eval_params = {
        "model_path": str(model_path),
        "env_type": env_type,
        "n_episodes": n_episodes,
        "max_episode_steps": max_episode_steps,
        "fixed_goal": fixed_goal,
    }

    with wandb.init(
        project="rlive-train",
        job_type="eval",
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
        # Run evaluation — collect per-episode rewards, lengths & success
        # ------------------------------------------------------------------
        logger.info(f"Evaluating for {n_episodes} episodes...")

        # Callback to harvest `is_success` from the final info dict of each episode
        successes: list[bool] = []

        def _success_callback(locals_: dict, _globals: dict) -> None:
            infos = locals_.get("infos", [{}])
            dones = locals_.get("dones", [False])
            for info, done in zip(infos, dones):
                if done:
                    successes.append(bool(info.get("is_success", False)))

        episode_rewards, episode_lengths = evaluate_policy(
            model,
            env,
            n_eval_episodes=n_episodes,
            deterministic=True,
            return_episode_rewards=True,
            callback=_success_callback,
        )

        mean_reward  = float(np.mean(episode_rewards))
        std_reward   = float(np.std(episode_rewards))
        mean_length  = float(np.mean(episode_lengths))
        success_rate = float(np.mean(successes)) if successes else float("nan")

        # ── Per-episode table (visible in W&B Artifacts / Tables tab) ────
        has_success = len(successes) == len(episode_rewards)
        columns = ["episode", "reward", "length"] + (["success"] if has_success else [])
        table = wandb.Table(columns=columns)
        for i, (reward, length) in enumerate(zip(episode_rewards, episode_lengths)):
            row = [str(i + 1), reward, length]   # str → column stays String
            if has_success:
                row.append(float(successes[i]))
            table.add_data(*row)

        # Summary row
        summary_row = ["MEAN", mean_reward, mean_length]
        if has_success:
            summary_row.append(success_rate)
        table.add_data(*summary_row)

        # ── Summary — single values, no step axis ────────────────────────
        run.summary["eval/mean_reward"]          = mean_reward
        run.summary["eval/std_reward"]           = std_reward
        run.summary["eval/mean_episode_length"]  = mean_length
        run.summary["eval/success_rate"]         = success_rate
        run.summary["eval/n_episodes"]           = n_episodes
        wandb.log({"eval/episodes_table": table})

        logger.info(
            f"Evaluation complete — "
            f"mean reward: {mean_reward:.3f} ± {std_reward:.3f}, "
            f"mean length: {mean_length:.1f} steps, "
            f"success rate: {success_rate:.1%}"
        )

    env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the evaluation CLI."""
    from rlive_train.config.config import MODELS_DIR  # noqa: PLC0415
    # ===========================================================================
    # CONFIG — edit these defaults, then run:  uv run eval_sb3.py
    # All values can still be overridden via CLI flags.
    # ===========================================================================

    MODEL_PATH = MODELS_DIR / "simple_1160000_steps.zip"        # required — e.g. r"C:\Downloads\best_model.zip"
    ENV_TYPE   = "simple"  # options: sapien | simple
    N_EPISODES = 20        # recommended: 20–30
    MAX_STEPS  = 50        # must match training!
    FIXED_GOAL = True     # True to fix goal at image centre
    RUN_NAME   = "Simple_on_Simple"      # None = auto  e.g. "PPO_v1_sim_sapien"
    # ===========================================================================

    parser = argparse.ArgumentParser(
        description="Evaluate a trained SB3 PPO agent and log results to W&B."
    )
    parser.add_argument(
        "--model-path", type=str, default=MODEL_PATH,
        required=not bool(MODEL_PATH),
        help="Path to the model .zip file.",
    )
    parser.add_argument(
        "--env", choices=ENV_CHOICES, default=ENV_TYPE,
        help=f"Simulation backend (default: {ENV_TYPE})",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=N_EPISODES,
        help=f"Number of evaluation episodes (default: {N_EPISODES})",
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=MAX_STEPS,
        help=f"Max steps per episode — must match training (default: {MAX_STEPS})",
    )
    parser.add_argument(
        "--fixed-goal", action="store_true", default=FIXED_GOAL,
        help="Fix the goal at the centre of the image",
    )
    parser.add_argument(
        "--run-name", type=str, default=RUN_NAME,
        help="Custom W&B run name (default: eval_<env>)",
    )
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        env_type=args.env,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_episode_steps,
        fixed_goal=args.fixed_goal,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
