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

# --- Goal settings ------------------------------------------------------------
GOAL_RADIUS: int = int(os.getenv("GOAL_RADIUS", "50"))
"""
Radius around the center point to consider as goal reached (in pixels).

default: 50
"""

GOAL_COLOUR: tuple[int, int, int] = (0, 255, 0)
"""
Colour of the goal circle in RGB format.

default: (0, 255, 0)  # Green
"""

GOAL_ALPHA: float = float(os.getenv("GOAL_ALPHA", "0.5"))
""" 
Alpha value of the goal circle overlay (0.0 = invisible, 1.0 = fully visible).

default: 0.5
"""

# --- Reward config -----------------------------------------------------------

REWARD_MODE: str = os.getenv("REWARD_MODE", "dense")
"""
Reward calculation mode: 'dense' or 'sparse'.
- dense: Distance-based reward (closer to goal = higher reward)
- sparse: Binary reward (+1.0 only when goal reached)

default: dense
"""

# --- debug config ------------------------------------------------------------

DEBUG: bool = os.getenv("DEBUG", "").lower() in ["true", "yes", "1"]
"""
Enable debug mode (with: true, yes, 1)

default: False
"""
