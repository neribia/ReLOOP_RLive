""" Config-file for rlive-world package."""
import os
import sys
from pathlib import Path

# --- Helper Functions --------------------------------------------------------

def _find_project_root() -> Path:
    """
    Finds the package root directory (not the workspace root).

    Priority order:
    1. PyInstaller bundles: Uses sys._MEIPASS
    2. Package root: Searches for directory containing src/<package_dir>
    3. Fallback: Any pyproject.toml (closest to config.py)
    4. Final fallback: Relative directory structure

    Returns:
        Path: The package root directory (e.g., packages/rlive-world)
    """
    # PyInstaller Bundle
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)

    resolved = Path(__file__).resolve()

    # Derive package directory name (e.g., 'rlive_world')
    # Path structure: .../packages/rlive-world/src/rlive_world/config/config.py
    try:
        package_dir = resolved.parents[1].name  # rlive_world
    except Exception:
        package_dir = None

    # 1) Search for package root: ancestor containing src/<package_dir>
    if package_dir:
        current = resolved.parent
        for _ in range(10):  # Max 10 levels up
            if (current / "src" / package_dir).exists():
                return current
            if current.parent == current:
                break
            current = current.parent

    # 2) Fallback: Search for any pyproject.toml (closest one)
    current = resolved.parent
    for _ in range(10):  # Max 10 levels up
        if (current / "pyproject.toml").exists():
            return current
        if current.parent == current:
            break
        current = current.parent

    # 3) Final fallback: Relative to config.py location
    # For packages/rlive-*/src/rlive_*/config/config.py -> go up 3 levels to package root
    return resolved.parents[3]

# --- Base Configuration --------------------------------------------------------

PACKAGE_NAME = "rlive-world"

# Project root directory (auto-detected)
PROJECT_ROOT = _find_project_root()

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
USE_DUMMY_HARDWARE: bool = os.getenv("USE_DUMMY_HARDWARE", "true").lower() in ["true", "yes", "1"]
"""
Use dummy hardware (DummyFinder, DummySpheroEduAPI) instead of real hardware.

default: true
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

CAMERA_RESOLUTION: tuple[int, int] = (CAMERA_HEIGHT, CAMERA_WIDTH)
"""
Camera resolution as (height, width).

Default: (480, 640)
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

SPHEROBOLTPLUS_SCANNING_TIME: float = float(os.getenv("SPHEROBOLTPLUS_SCANNING_TIME", "3"))
"""
Scanning time for the Sphero Bolt+ robot.

default: 3
"""

# --- debug config ------------------------------------------------------------
DEBUG: bool = os.getenv("DEBUG", "").lower() in ["true", "yes", "1"]
"""
Enable debug mode (with: true, yes, 1)

default: False
"""

# Logging configuration has been moved to rlive-common config.
