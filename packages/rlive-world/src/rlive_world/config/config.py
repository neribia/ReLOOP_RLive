""" Config-file for rlive-world package."""
import os
import sys
from os.path import abspath, dirname, join

# --- Base Configuration --------------------------------------------------------

PACKAGE_NAME = "rlive-world"
BUNDLE_DIR = getattr(sys, "_MEIPASS", abspath(join(dirname(__file__), "..", "..")))
APP_HOME = os.getenv("APP_HOME", f"{BUNDLE_DIR}")

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
Sets the camer id.

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

SPHEROBOLTPLUS_NAME: str = os.getenv("SPHEROBOLTPLUS_NAME", "BP-D217")
"""
Default robot name to connect to. 
- For dummy hardware: "DummyBolt"
- For real hardware: Your Sphero's Bluetooth name (e.g., "BP-D217")

default: BP-D217
"""

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
