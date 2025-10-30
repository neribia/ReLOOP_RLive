import os
import sys
from os.path import abspath, dirname, join


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

# ---  config ---------------------------------------------------
