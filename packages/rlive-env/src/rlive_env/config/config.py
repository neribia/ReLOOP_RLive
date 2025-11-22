import os
import sys
from os.path import abspath, dirname, join

# --- base directories --------------------------------------------------------

BUNDLE_DIR = getattr(sys, "_MEIPASS", abspath(join(dirname(__file__), "..", "..")))

APP_HOME = os.getenv("APP_HOME", f"{BUNDLE_DIR}")

# --- interface config
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

# --- debug config ------------------------------------------------------------

DEBUG = os.getenv("DEBUG", "").lower() in ["true", "yes", "1"]
"""
enable debug mode (with: true, yes, 1)

default: False
"""
