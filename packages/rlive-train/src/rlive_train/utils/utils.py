from typing import Any

import numpy as np
import torch


class RandomAgent:
    """A random-walk agent that samples uniformly from the action space.

    Mimics the SB3 ``model.predict`` interface so it can be used as a
    drop-in replacement inside ``evaluate_policy`` without any changes to the
    evaluation loop.

    Args:
        env: A (vectorized) Gymnasium environment whose ``action_space`` is
            used for sampling.  Pass the *vectorized* env (e.g.
            ``VecTransposeImage``) after calling ``build_sim_env``.

    Example::

        agent = RandomAgent(env)
        obs = env.reset()
        action, _ = agent.predict(obs)
    """

    def __init__(self, env: Any) -> None:
        self.env = env

    def predict(
        self,
        observation: np.ndarray,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = False,  # noqa: ARG002 — kept for API compat.
    ) -> tuple[np.ndarray, None]:
        """Return a random action sampled from the environment's action space.

        Args:
            observation: Current observation (ignored).
            state: Recurrent state (ignored).
            episode_start: Episode-start mask (ignored).
            deterministic: Ignored — random agent always samples uniformly.

        Returns:
            Tuple of (action_array, None) matching the SB3 model API.
        """
        action = self.env.action_space.sample()
        return np.array([action]), None


def get_device(prefer: str = "auto") -> torch.device:
    """Return the best available torch device.

    Args:
        prefer: "auto", "cuda", "mps", or "cpu".

    Returns:
        torch.device: Selected device.
    """
    prefer = prefer.lower()

    if prefer == "cpu":
        return torch.device("cpu")

    if prefer == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if prefer == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # auto: prefer CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

if __name__ == "__main__":
    print(get_device())