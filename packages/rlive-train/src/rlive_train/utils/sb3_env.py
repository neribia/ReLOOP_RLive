"""Helpers for SB3 vectorized simulation environments."""

from collections.abc import Callable
from enum import Enum

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack


class EvalVecBackend(Enum):
    """Supported vectorized backends for evaluation environments."""

    DUMMY = "dummy"
    SUBPROC = "subproc"


def build_sim_env(
    env_factory: Callable[[], object],
    n_stack: int = 4,
    num_envs: int = 1,
    backend: EvalVecBackend = EvalVecBackend.DUMMY,
):
    """Create a frame-stacked vectorized environment.

    Uses DummyVecEnv for num_envs=1 or backend=DUMMY, SubprocVecEnv otherwise.

    Args:
        env_factory: Factory callable that returns an env instance.
        n_stack: Number of frames to stack (default: 4)
        num_envs: Number of parallel environments (default: 1)
        backend: Vectorization backend—DUMMY or SUBPROC (default: DUMMY)

    Returns:
        VecTransposeImage: Transposed, frame-stacked vectorized environment
    """
    if num_envs < 1:
        raise ValueError("num_envs must be >= 1")

    if num_envs > 1 and backend is EvalVecBackend.SUBPROC:
        env = SubprocVecEnv([env_factory for _ in range(num_envs)], start_method="spawn")
    else:
        env = DummyVecEnv([env_factory for _ in range(num_envs)])

    return VecTransposeImage(VecFrameStack(env, n_stack))
