"""Config-file for the rlive-common package."""
import os
import sys
from os.path import abspath, dirname, join
from pathlib import Path

# --- Base Configuration --------------------------------------------------------

PACKAGE_NAME = "rlive-common"
BUNDLE_DIR = getattr(sys, "_MEIPASS", abspath(join(dirname(__file__), "..", "..")))
APP_HOME = os.getenv("APP_HOME", f"{BUNDLE_DIR}")

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

LOGGING_FILE_PATH: Path = Path(os.getenv("LOGGING_FILE_PATH", Path(sys.argv[0]).resolve().parent / "logs"))
"""
Path to the logging directory.
Default: ./logs
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
