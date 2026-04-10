"""Basic Simulation Engine Configuration module.

This module provides default configurations for the basic simulation engines
(SimplePhysicsEngine and OpenCVRenderEngine).
"""

import os

# ============================================================================
# PHYSICS CONFIGURATION
# ============================================================================

BOX_WIDTH: float = float(os.getenv("RLIVE_SIM_BASIC_BOX_WIDTH", "0.8"))
"""
Width of the bounding box in meters.

default: 0.8
"""

BOX_HEIGHT: float = float(os.getenv("RLIVE_SIM_BASIC_BOX_HEIGHT", "0.6"))
"""
Height of the bounding box in meters.

default: 0.4
"""

# ============================================================================
# RENDER CONFIGURATION
# ============================================================================

IMAGE_WIDTH: int = int(os.getenv("RLIVE_SIM_BASIC_IMAGE_WIDTH", "640"))
"""
Default image width in pixels.

default: 640
"""

IMAGE_HEIGHT: int = int(os.getenv("RLIVE_SIM_BASIC_IMAGE_HEIGHT", "480"))
"""
Default image height in pixels.

default: 480
"""

FOV_DEGREES: float = float(os.getenv("RLIVE_SIM_BASIC_FOV_DEGREES", "60.0"))
"""
Camera horizontal field of view in degrees.

default: 60.0
"""

CAMERA_POSITION_X: float = float(os.getenv("RLIVE_SIM_BASIC_CAMERA_POSITION_X", "0.0"))
CAMERA_POSITION_Y: float = float(os.getenv("RLIVE_SIM_BASIC_CAMERA_POSITION_Y", "0.0"))
CAMERA_POSITION_Z: float = float(os.getenv("RLIVE_SIM_BASIC_CAMERA_POSITION_Z", "0.7"))
"""
Default XYZ coordinates of the camera in meters.

default: [0.0, 0.0, 0.8]
"""

CAMERA_ROTATION_ROLL: float = float(os.getenv("RLIVE_SIM_BASIC_CAMERA_ROTATION_ROLL", "0.0"))
CAMERA_ROTATION_PITCH: float = float(os.getenv("RLIVE_SIM_BASIC_CAMERA_ROTATION_PITCH", "0.0"))
CAMERA_ROTATION_YAW: float = float(os.getenv("RLIVE_SIM_BASIC_CAMERA_ROTATION_YAW", "0.0"))
"""
Default Euler angles (roll, pitch, yaw) in radians.

default: [0.0, 0.0, 0.0]
"""
