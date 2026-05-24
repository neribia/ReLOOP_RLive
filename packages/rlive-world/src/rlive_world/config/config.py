""" Config-file for rlive-world package."""
import os
import sys
from pathlib import Path

from rlive_common.utils.path_utils import find_project_root
from rlive_common.utils.import_utils import get_rgb_env

# --- Base Configuration --------------------------------------------------------

PACKAGE_NAME = "rlive-world"

# Project root directory (auto-detected)
PROJECT_ROOT = find_project_root(__file__)

# Bundle directory (for PyInstaller compatibility)
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))

APP_HOME = os.getenv("APP_HOME", str(BUNDLE_DIR))

# --- Server config ------------------------------------------------------------
WORLD_HOST: str = os.getenv("WORLD_HOST", "0.0.0.0")
"""
Set the Host-IP.

default: 0.0.0.0
"""

WORLD_PORT: int = int(os.getenv("WORLD_PORT", "8000"))
"""
Set the Port.

default: 8000
"""

# --- Hardware config ---------------------------------------------------------
USE_DUMMY_HARDWARE: bool = os.getenv("USE_DUMMY_HARDWARE", "false").lower() in ["true", "yes", "1"]
"""
Use dummy hardware (DummyFinder, DummySpheroEduAPI) instead of real hardware.

default: false
"""

# --- Camera config -----------------------------------------------------

CAMERA_TYPE: str = str(os.getenv("DEFAULT_CAMERA_TYPE", "dummy"))
"""
Default camera type: 'dummy', 'webcam', or 'picam'

default: dummy
"""

CAMERA_ID: int = int(os.getenv("CAMERA_ID", "0"))
"""
Sets the camera id.

Default: 0
"""

CAMERA_WIDTH: int = int(os.getenv("CAMERA_WIDTH", 640))
CAMERA_HEIGHT: int = int(os.getenv("CAMERA_HEIGHT", 480))

CAMERA_RESOLUTION: tuple[int, int] = (CAMERA_WIDTH, CAMERA_HEIGHT)
"""
Camera resolution as (width, height).

Default: (640, 480)
"""

CAMERA_EXPOSURE_TIME_MS: float = float(os.getenv("CAMERA_EXPOSURE_TIME_MS", 20))
"""
Camera exposure time in milliseconds

Default: 20
"""

# --- Sphero Bolt+ config -----------------------------------------------------

SPHEROBOLTPLUS_NAME: str = str(os.getenv("SPHEROBOLTPLUS_NAME", "SB-0001"))
"""
Name of the Sphero Bolt+ robot to connect to.

default: SB-0001
"""

SPHEROBOLTPLUS_SCANNING_TIME: float = float(os.getenv("SPHEROBOLTPLUS_SCANNING_TIME", "3"))
"""
Scanning time for the Sphero Bolt+ robot.

default: 3
"""

SPHEROBOLTPLUS_DISPLAY_COLOR_R: int = get_rgb_env("SPHEROBOLTPLUS_DISPLAY_COLOR_R", "255")
"""
Red component of the default display color (0-255).

default: 255
"""

SPHEROBOLTPLUS_DISPLAY_COLOR_G: int = get_rgb_env("SPHEROBOLTPLUS_DISPLAY_COLOR_G", "255")
"""
Green component of the default display color (0-255).

default: 255
"""

SPHEROBOLTPLUS_DISPLAY_COLOR_B: int = get_rgb_env("SPHEROBOLTPLUS_DISPLAY_COLOR_B", "255")
"""
Blue component of the default display color (0-255).

default: 255
"""

# --- debug config ------------------------------------------------------------
DEBUG: bool = os.getenv("DEBUG", "").lower() in ["true", "yes", "1"]
"""
Enable debug mode (with: true, yes, 1)

default: False
"""

# Logging configuration has been moved to rlive-common config.
