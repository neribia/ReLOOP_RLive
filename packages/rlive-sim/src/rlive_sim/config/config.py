"""Configuration module for rlive-sim.

This module provides Pydantic configuration models for:
- Physics engines (PhysicsConfig)
- Render engines (RenderConfig)
- Integrated engines (IntegratedConfig)
- Overall simulation setup (SimulationConfig)

Configuration values can be set via:
- Direct constructor injection
- Environment variables (prefix: RLIVE_SIM_)
- Config files (YAML/JSON via Pydantic)

Resources are located at:
- RESOURCES_DIR: Root resources directory
- RESOURCES_DIR / "mujoco": MuJoCo model files
"""

import os
import sys
from pathlib import Path

from rlive_common.utils.path_utils import find_project_root

# --- Base Configuration --------------------------------------------------------

PACKAGE_NAME = "rlive-sim"

# Project root directory (auto-detected)
PROJECT_ROOT = find_project_root(__file__)

# Bundle directory (for PyInstaller compatibility)
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))

APP_HOME = os.getenv("APP_HOME", str(BUNDLE_DIR))

RESOURCES_DIR = Path(__file__).parents[3] / "resources"
"""Root resources directory for rlive-sim package.

Location: packages/rlive-sim/resources/

Contains subdirectories:
- mujoco/: MuJoCo model files (.xml, .urdf)
- [other engine-specific resources]
"""

# --- Environment config ------------------------------------------------------
MAX_STEPS_PER_EPISODE: int = int(os.getenv("MAX_STEPS_PER_EPISODE", "100"))
"""
Number of max steps per episode. After this number of steps, the episode will be terminated.

default: 100
"""

# --- Reset settings -----------------------------------------------------------
NUMBER_RESET_ACTIONS: int = int(os.getenv("DECAY_STEPS", "5"))
"""
Number of steps to decouple the episodes in the reset phase.

default: 5
"""


__all__ = [
    "RESOURCES_DIR",
]
