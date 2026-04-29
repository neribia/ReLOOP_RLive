"""Helpers for SB3 vectorized simulation environments."""

from collections.abc import Callable
from enum import Enum

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack

class EvalVecBackend(Enum):
    """Supported vectorized backends for evaluation environments."""

    DUMMY = "dummy"
    SUBPROC = "subproc"


def build_sim_train_env(
    env_factory: Callable[[], object],
    n_stack: int = 4,
    num_envs: int = 4,
):
    """Create a frame-stacked SubprocVecEnv for simulation training.

    Args:
        num_envs: Number of parallel environments (default: 4)
        n_stack: Number of frames to stack (default: 4)
        env_factory: Optional factory function that returns env instances.
                    If None, uses default sapiens_env_factory()

    Returns:
        VecFrameStack: Stacked vectorized training environment
    """
    if num_envs < 1:
        raise ValueError("num_envs must be >= 1")

    env = SubprocVecEnv([env_factory for _ in range(num_envs)])
    return VecFrameStack(env, n_stack)


def build_sim_eval_env(
    env_factory: Callable[[], object],
    n_stack: int = 4,
    num_envs: int = 1,
    backend: EvalVecBackend = EvalVecBackend.DUMMY,
):
    """Create a frame-stacked evaluation vec env with a selectable backend.

    Args:
        n_stack: Number of frames to stack (default: 4)
        num_envs: Number of parallel eval environments (default: 1)
        env_factory: Optional factory function that returns env instances.
                    If None, uses default sapiens_env_factory()
        backend: Vectorization backend—DUMMY or SUBPROC (default: DUMMY)

    Returns:
        VecFrameStack: Stacked vectorized evaluation environment
    """
    if num_envs < 1:
        raise ValueError("num_envs must be >= 1")

    if backend is EvalVecBackend.SUBPROC:
        env = SubprocVecEnv([env_factory for _ in range(num_envs)])
    else:
        env = DummyVecEnv([env_factory for _ in range(num_envs)])

    return VecFrameStack(env, n_stack)
