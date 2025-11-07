import os
import sys
from os.path import abspath, dirname, join


# --- base directories --------------------------------------------------------

BUNDLE_DIR = getattr(sys, "_MEIPASS", abspath(join(dirname(__file__), "..", "..")))

APP_HOME = os.getenv("APP_HOME", f"{BUNDLE_DIR}")

# --- interface config
WORLD_BASE_URL = os.getenv("WORLD_BASE_URL", "http://localhost:8000")
"""
Base url for the server.

default: http://localhost:8000
"""

# --- debug config ------------------------------------------------------------

DEBUG = os.getenv("DEBUG", "").lower() in ["true", "yes", "1"]
"""
enable debug mode (with: true, yes, 1)

default: False
"""
