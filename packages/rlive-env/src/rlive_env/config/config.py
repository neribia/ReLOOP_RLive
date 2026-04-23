"""Config-file for the rlive-env package."""
import os
import sys
from pathlib import Path

from rlive_common.utils.path_utils import find_project_root

# --- Base Configuration --------------------------------------------------------

PACKAGE_NAME = "rlive-env"

# Project root directory (auto-detected)
PROJECT_ROOT = find_project_root(__file__)

# Bundle directory (for PyInstaller compatibility)
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))

APP_HOME = os.getenv("APP_HOME", str(BUNDLE_DIR))

# --- interface config --------------------------------------------------------
WORLD_BASE_URL: str = os.getenv("WORLD_BASE_URL", "http://localhost:8000")
"""
Base url for the server.

default: http://localhost:8000
"""

WORLD_INTERFACE_TIMEOUT: float = float(os.getenv("WORLD_INTERFACE_TIMEOUT", "30"))
"""
Timeout of request from the interface in seconds.

default: 30
"""

WORLD_INTERFACE_MAX_RETRIES: int = int(os.getenv("WORLD_INTERFACE_MAX_RETRIES", "3"))
"""
Maximum number of retries for requests from the interface.

default: 3
"""

WORLD_INTERFACE_MAX_RETRIES_TIME: float = float(os.getenv("WORLD_INTERFACE_MAX_RETRIES_TIME", "180"))
"""
Maximum total time for retries from the interface in seconds.

default: 180
"""

WORLD_INTERFACE_BACKOFF_FACTOR: float = float(os.getenv("WORLD_INTERFACE_BACKOFF_FACTOR", "0.3"))
"""
Backoff factor for retries from the interface.  

default: 0.3
"""

# --- Reset settings -----------------------------------------------------------
NUMBER_RESET_ACTIONS: int = int(os.getenv("DECAY_STEPS", "5"))
"""
Number of steps to decouple the episodes in the reset phase.

default: 5
"""

# --- Environment config ------------------------------------------------------
MAX_STEPS_PER_EPISODE: int = int(os.getenv("MAX_STEPS_PER_EPISODE", "100"))
"""
Number of max steps per episode. After this number of steps, the episode will be terminated.

default: 100
"""


# --- Reward config -----------------------------------------------------------

REWARD_MODE: str = os.getenv("REWARD_MODE", "dense")
"""
Reward calculation mode: 'dense' or 'sparse'.
- dense: Distance-based reward (closer to goal = higher reward)
- sparse: Binary reward (+1.0 only when goal reached)

default: dense
"""

# --- Sphero Bolt+ config -----------------------------------------------------

SPHEROBOLTPLUS_SPEED: int = int(os.getenv("SPHEROBOLTPLUS_SPEED", "50"))
"""
Speed of the Sphero Bolt+ robot. (-255, 255)

default: 50
"""

SPHEROBOLTPLUS_DURATION: float = float(os.getenv("SPHEROBOLTPLUS_DURATION", "1"))
"""
Moving duration (in s) of the Sphero Bolt+ robot.

default: 1
"""

SPHEROBOLTPLUS_SPEED_FACTOR: float = float(os.getenv("SPHEROBOLTPLUS_SPEED_FACTOR", "0.5"))
"""
Factor to reduce the max_speed for ActionSpaceType.CARTESIAN. 
Default is set to 0.5 to reduce slip.

default: 0.5
"""


# --- debug config ------------------------------------------------------------

DEBUG: bool = os.getenv("DEBUG", "").lower() in ["true", "yes", "1"]
"""
Enable debug mode (with: true, yes, 1)

default: False
"""
