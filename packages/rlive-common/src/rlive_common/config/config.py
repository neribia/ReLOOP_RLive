from typing import Optional
import os
import sys
from os.path import abspath, dirname, join
from pathlib import Path


# --- base directories --------------------------------------------------------

PACKAGE_NAME = "rlive-common"

BUNDLE_DIR = getattr(sys, "_MEIPASS", abspath(join(dirname(__file__), "..", "..")))

APP_HOME = os.getenv("APP_HOME", f"{BUNDLE_DIR}")

# logger.debug("BUNDLE_DIR: %s, APP_HOME: %s", BUNDLE_DIR, APP_HOME)

# --- debug config ------------------------------------------------------------

DEBUG = os.getenv("DEBUG", "").lower() in ["true", "yes", "1"]
"""
enable debug mode (with: true, yes, 1)

default: False
"""

# ---  logging ---------------------------------------------------

LOGGING_LEVEL = os.getenv("LOG_LEVEL", "INFO" if not DEBUG else "DEBUG").upper()
LOGGING_FORMAT = os.getenv("LOG_FORMAT", "detailed").lower()
LOGGING_DATE = os.getenv("LOG_DATE", "long").lower()
LOGGING_STREAM = os.getenv("LOG_STREAM", "true").lower() in ("1", "true", "yes")

# Default log file path (e.g., logs/app.log)
LOGGING_TO_FILE = os.getenv("LOG_TO_FILE", "false").lower() in ("1", "true", "yes")
EXECUTION_DIR = Path(sys.argv[0]).resolve().parent
LOGGING_DIR = Path(os.getenv("LOG_DIR", EXECUTION_DIR / "logs"))
LOGGING_DIR.mkdir(parents=True, exist_ok=True)
LOGGING_FILE: Optional[str] = (
    os.getenv("LOG_FILE", str(LOGGING_DIR / "app.log")) if LOGGING_TO_FILE else None
)

