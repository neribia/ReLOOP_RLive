"""RLive Environment package for reinforcement learning with remote world communication.

This package provides a Gymnasium-compatible environment that communicates with a remote
World server over HTTP and supports multiple action space transformers for flexible
agent training.
"""

# Main environment class
from rlive_env.remote_env import RemoteWorldEnv

# Action space transformers
from rlive_common.core.action_space import ActionSpaceType

__all__ = [
    # Main environment
    "RemoteWorldEnv",
    # Action space types
    "ActionSpaceType",
]

__version__ = "0.1.0"
__author__ = "ReLoop Team"
