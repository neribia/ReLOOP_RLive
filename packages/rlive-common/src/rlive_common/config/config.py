"""Config-file for the rlive-common package."""
import os
import sys
from pathlib import Path

from rlive_common.utils.path_utils import find_project_root

# --- Base Configuration --------------------------------------------------------

PACKAGE_NAME = "rlive-common"

# Project root directory (auto-detected)
PROJECT_ROOT = find_project_root(__file__)

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

_LOGGING_FILE_PATH_ENV = os.getenv("LOGGING_FILE_PATH")
LOGGING_FILE_PATH: Path = Path(_LOGGING_FILE_PATH_ENV) if _LOGGING_FILE_PATH_ENV else PROJECT_ROOT / "logs"
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
