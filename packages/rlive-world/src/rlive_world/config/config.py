import os
import sys
from os.path import abspath, dirname, join


# --- base directories --------------------------------------------------------

BUNDLE_DIR = getattr(sys, "_MEIPASS", abspath(join(dirname(__file__), "..", "..")))

APP_HOME = os.getenv("APP_HOME", f"{BUNDLE_DIR}")

# --- Server config ------------------------------------------------------------
WORLD_HOST = os.getenv("WORLD_HOST", "0.0.0.0")
"""
Set the Host-IP

default: 0.0.0.0
"""

WORLD_PORT = int(os.getenv("WORLD_PORT", "8000"))
"""
Set the Port

default: 8000
"""

# --- debug config ------------------------------------------------------------

DEBUG = os.getenv("DEBUG", "").lower() in ["true", "yes", "1"]
"""
enable debug mode (with: true, yes, 1)

default: False
"""
