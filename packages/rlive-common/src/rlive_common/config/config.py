"""Config-file for the rlive-common package."""
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
        Path: The package root directory (e.g., packages/rlive-common)
    """
    # PyInstaller Bundle
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)

    resolved = Path(__file__).resolve()

    # Derive package directory name (e.g., 'rlive_common')
    # Path structure: .../packages/rlive-common/src/rlive_common/config/config.py
    try:
        package_dir = resolved.parents[1].name  # rlive_common
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

PACKAGE_NAME = "rlive-common"

# Project root directory (auto-detected)
PROJECT_ROOT = _find_project_root()

# Bundle directory (for PyInstaller compatibility)
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))

APP_HOME = os.getenv("APP_HOME", str(BUNDLE_DIR))

# --- debug config ------------------------------------------------------------

DEBUG = os.getenv("DEBUG", "").lower() in ["true", "yes", "1"]
"""
enable debug mode (with: true, yes, 1)

default: False
"""

# --- Logging Configuration ---------------------------------------------------
LOGGING_LEVEL: str = os.getenv("LOGGING_LEVEL", "INFO" if not DEBUG else "DEBUG")
"""
Set the logging level.
Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
Default: INFO (or DEBUG if DEBUG is True)
"""

LOGGING_FORMAT: str = os.getenv("LOGGING_FORMAT", "detailed")
"""
Set the logging format. 
Options: 'simple', 'detailed', 'json', 'color', 'thread'
Default: detailed
"""

LOGGING_DATE: str = os.getenv("LOGGING_DATE", "long")
"""
Date style for logs.
Options: short, long, iso, date_only, time_ms
Default: long
"""

LOGGING_STREAM: bool = os.getenv("LOGGING_STREAM", "true").lower() in ["true", "yes", "1"]
"""
Stream logs to console.
Options: true, false
Default: true
"""

LOGGING_TO_FILE: bool = os.getenv("LOGGING_TO_FILE", "false").lower() in ["true", "yes", "1"]
"""
Enable/Disable logging to a file.
Options: true, false
Default: false
"""

LOGGING_FILE_PATH: Path = Path(os.getenv("LOGGING_FILE_PATH", str(PROJECT_ROOT / "logs")))
"""
Path to the logging directory.
Default: <project_root>/logs
"""

LOGGING_FILE_NAME: str = os.getenv("LOGGING_FILE_NAME", "app.log")
"""
Name of the logging file.
Default: app.log
"""

LOGGING_FILE: str | None = str(LOGGING_FILE_PATH / "app.log") if LOGGING_TO_FILE else None
"""
Full path to the logging file.

Default: ./logs/app.log
"""
