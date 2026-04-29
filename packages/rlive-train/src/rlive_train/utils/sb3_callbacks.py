"""Helpers for constructing common SB3 callbacks."""

from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from wandb.integration.sb3 import WandbCallback


def build_eval_callback(
    env_eval,
    log_path: str,
    save_path: str,
    eval_freq: int,
    n_eval_episodes: int = 10,
) -> EvalCallback:
    """Build a single EvalCallback with common defaults."""
    return EvalCallback(
        env_eval,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )


def build_wandb_eval_callbacks(
    env_eval,
    log_path: str,
    save_path: str,
    eval_freq: int,
    n_eval_episodes: int = 10,
    wandb_verbose: int = 2,
    wandb_model_save_path: str | None = None,
    wandb_model_save_freq: int | None = None,
) -> CallbackList:
    """Build EvalCallback + WandbCallback as a CallbackList."""
    eval_callback = build_eval_callback(
        env_eval=env_eval,
        log_path=log_path,
        save_path=save_path,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
    )

    wandb_kwargs = {"verbose": wandb_verbose}
    if wandb_model_save_path is not None:
        wandb_kwargs["model_save_path"] = wandb_model_save_path
    if wandb_model_save_freq is not None:
        wandb_kwargs["model_save_freq"] = wandb_model_save_freq

    wandb_callback = WandbCallback(**wandb_kwargs)
    return CallbackList([wandb_callback, eval_callback])
