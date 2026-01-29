"""Config-file for the rlive-env package."""
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
        Path: The package root directory (e.g., packages/rlive-env)
    """
    # PyInstaller Bundle
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)

    resolved = Path(__file__).resolve()

    # Derive package directory name (e.g., 'rlive_env')
    # Path structure: .../packages/rlive-env/src/rlive_env/config/config.py
    try:
        package_dir = resolved.parents[1].name  # rlive_env
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

PACKAGE_NAME = "rlive-env"

# Project root directory (auto-detected)
PROJECT_ROOT = _find_project_root()

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

# --- debug config ------------------------------------------------------------

DEBUG: bool = os.getenv("DEBUG", "").lower() in ["true", "yes", "1"]
"""
Enable debug mode (with: true, yes, 1)

default: False
"""
