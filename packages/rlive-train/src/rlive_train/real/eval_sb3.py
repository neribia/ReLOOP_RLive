"""SB3 Evaluation - Real Environment.

Evaluates an existing PPO model via RemoteWorldEnv and logs results to W&B.

Usage
-----
    python eval_sb3.py --model-path /path/to/best_model.zip
    python eval_sb3.py --model-path /path/to/best_model.zip --bolt-use-dummy --n-episodes 10
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
from rlive_train.utils.utils import RandomAgent  # noqa: E402

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate(  # noqa: PLR0913
    model_path: str | Path | None,
    base_url: str = "http://127.0.0.1:8000",
    bolt_name: str = "BP-D217",
    camera_id: int = 1,
    agent_type: str = "ppo",
    n_episodes: int = 10,
    max_episode_steps: int = 20,
    n_stack: int = 4,
    fixed_goal: bool = False,
    render: bool = False,
    run_name: str | None = None,
) -> None:
    """Evaluate a trained PPO model (or random agent) on the real RemoteWorldEnv and log to W&B.

    Args:
        model_path: Path to the model .zip file. Required when agent_type='ppo',
            ignored when agent_type='random'.
        base_url: Base URL of the world server.
        bolt_name: Sphero Bolt device name.
        camera_id: OpenCV camera index.
        agent_type: Agent to evaluate — 'ppo' (loads model_path) or 'random'
            (samples uniformly from the action space).
        n_episodes: Number of deterministic evaluation episodes.
        max_episode_steps: Max steps per episode (must match training).
        n_stack: Number of frames to stack — must match the model (default: 4).
        fixed_goal: Whether the goal is fixed at the image centre.
        render: If True, render the environment observation during evaluation.
        run_name: Custom W&B run name. When None, defaults to ``eval_real``.
    """
    if agent_type == "ppo":
        model_path = Path(model_path)
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return

    if run_name is None:
        run_name = f"eval_{agent_type}_real"

    # ------------------------------------------------------------------
    # Build evaluation environment — same wrapper pipeline as sim so
    # a sim-trained CnnPolicy model loads without shape mismatches.
    # ------------------------------------------------------------------
    factory = make_env_factory(
        env_fn=make_real_env,
        base_url=base_url,
        bolt_name=bolt_name,
        camera_id=camera_id,
        max_episode_steps=max_episode_steps,
        fixed_goal=fixed_goal,
        render_mode="opencv" if render else None,
    )
    # DummyVecEnv → VecFrameStack(n_stack) → VecTransposeImage
    env = build_sim_env(env_factory=factory, n_stack=n_stack, backend=EvalVecBackend.DUMMY)

    # ------------------------------------------------------------------
    # W&B run — linked to the training run via group
    # ------------------------------------------------------------------
    eval_params = {
        "model_path": str(model_path) if agent_type == "ppo" else "N/A (random agent)",
        "agent_type": agent_type,
        "base_url": base_url,
        "bolt_name": bolt_name,
        "n_episodes": n_episodes,
        "max_episode_steps": max_episode_steps,
        "n_stack": n_stack,
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
        # Load model or build random agent
        # ------------------------------------------------------------------
        if agent_type == "ppo":
            logger.info(f"Loading PPO model from {model_path}")
            model = PPO.load(model_path, env=env)
        else:
            logger.info("Using RandomAgent — sampling from action_space")
            model = RandomAgent(env)

        # ------------------------------------------------------------------
        # Run evaluation — collect per-episode rewards, lengths & success
        # ------------------------------------------------------------------
        logger.info(f"Evaluating for {n_episodes} episodes...")

        # Callback to harvest `is_success` and `error` from the final info dict of each episode
        successes: list[bool] = []
        errors: list[str] = []

        def _success_callback(locals_: dict, _globals: dict) -> None:
            infos = locals_.get("infos", [{}])
            dones = locals_.get("dones", [False])
            for info, done in zip(infos, dones):
                if done:
                    successes.append(bool(info.get("is_success", False)))
                    errors.append(info.get("error", ""))

        episode_rewards, episode_lengths = evaluate_policy(
            model,
            env,
            n_eval_episodes=n_episodes,
            deterministic=True,
            return_episode_rewards=True,
            callback=_success_callback,
            render=render,
        )

        mean_reward  = float(np.mean(episode_rewards))
        std_reward   = float(np.std(episode_rewards))
        mean_length  = float(np.mean(episode_lengths))
        success_rate = float(np.mean(successes)) if successes else float("nan")
        error_rate   = float(sum(1 for e in errors if e) / len(errors)) if errors else float("nan")

        # ── Per-episode table ─────────────────────────────────────────────
        has_success = len(successes) == len(episode_rewards)
        has_errors  = len(errors) == len(episode_rewards)
        columns = ["episode", "reward", "length"]
        if has_success:
            columns.append("success")
        if has_errors:
            columns += ["error_flag", "error"]
        table = wandb.Table(columns=columns)
        for i, (reward, length) in enumerate(zip(episode_rewards, episode_lengths)):
            row = [str(i + 1), reward, length]
            if has_success:
                row.append(float(successes[i]))
            if has_errors:
                row.append(1.0 if errors[i] else 0.0)
                row.append(errors[i])
            table.add_data(*row)

        # Summary row
        summary_row = ["MEAN", mean_reward, mean_length]
        if has_success:
            summary_row.append(success_rate)
        if has_errors:
            summary_row.append(error_rate)   # average of error_flag = fraction of episodes with error
            summary_row.append("")
        table.add_data(*summary_row)

        # ── Summary — single values, no step axis ────────────────────────
        run.summary["eval/mean_reward"]         = mean_reward
        run.summary["eval/std_reward"]          = std_reward
        run.summary["eval/mean_episode_length"] = mean_length
        run.summary["eval/success_rate"]        = success_rate
        run.summary["eval/error_rate"]          = error_rate
        run.summary["eval/n_episodes"]          = n_episodes
        wandb.log({"eval/episodes_table": table})

        logger.info(
            f"Evaluation complete — "
            f"mean reward: {mean_reward:.3f} ± {std_reward:.3f}, "
            f"mean length: {mean_length:.1f} steps, "
            f"success rate: {success_rate:.1%}, "
            f"error rate: {error_rate:.1%}"
        )

    env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the real-environment evaluation CLI."""
    from rlive_train.config.config import MODELS_DIR  # noqa: PLC0415
    # ===========================================================================
    # CONFIG — edit these defaults, then run:  uv run eval_sb3.py
    # All values can still be overridden via CLI flags.
    # ===========================================================================
    MODEL_PATH    = MODELS_DIR / "sapien_1120000_steps.zip"          # required — path to .zip
    BASE_URL      = os.getenv("WORLD_BASE_URL", "http://127.0.0.1:8000")
    BOLT_NAME     = "BP-D217"
    AGENT_TYPE    = "random"                    # options: ppo | random
    CAMERA_ID     = 2
    N_EPISODES    = 20                       # recommended: 20–30
    MAX_STEPS     = 50                       # must match training!
    N_STACK       = 4                        # must match training!
    FIXED_GOAL    = True                    # True to fix goal at image centre
    RENDER        = True                    # True = show OpenCV window during eval
    RUN_NAME      = "RandomWalk_on_Real"         # None = auto → "eval_real"
    # ===========================================================================

    parser = argparse.ArgumentParser(
        description="Evaluate a trained SB3 PPO agent on the real RemoteWorldEnv and log to W&B."
    )
    parser.add_argument(
        "--model-path", type=str, default=MODEL_PATH,
        required=False,
        help="Path to the model .zip file. Required when --agent=ppo.",
    )
    parser.add_argument(
        "--agent", choices=("ppo", "random"), default=AGENT_TYPE,
        help=f"Agent type: 'ppo' (loads --model-path) or 'random' (action_space.sample). (default: {AGENT_TYPE})",
    )
    parser.add_argument(
        "--base-url", type=str, default=BASE_URL,
        help=f"World server base URL (default: {BASE_URL})",
    )
    parser.add_argument(
        "--bolt-name", type=str, default=BOLT_NAME,
        help=f"Sphero Bolt device name (default: {BOLT_NAME})",
    )

    parser.add_argument(
        "--camera-id", type=int, default=CAMERA_ID,
        help=f"OpenCV camera index (default: {CAMERA_ID})",
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
        "--n-stack", type=int, default=N_STACK,
        help=f"Number of frames to stack — must match the model (default: {N_STACK})",
    )
    parser.add_argument(
        "--fixed-goal", action="store_true", default=FIXED_GOAL,
        help="Fix the goal at the centre of the image",
    )
    parser.add_argument(
        "--run-name", type=str, default=RUN_NAME,
        help="Custom W&B run name (default: eval_real)",
    )
    parser.add_argument(
        "--render", action="store_true", default=RENDER,
        help="Show the observation in an OpenCV window during evaluation",
    )
    args = parser.parse_args()

    if args.agent == "ppo" and not args.model_path:
        parser.error("--model-path is required when --agent=ppo")

    evaluate(
        model_path=args.model_path,
        base_url=args.base_url,
        bolt_name=args.bolt_name,
        camera_id=args.camera_id,
        agent_type=args.agent,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_episode_steps,
        n_stack=args.n_stack,
        fixed_goal=args.fixed_goal,
        render=args.render,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
