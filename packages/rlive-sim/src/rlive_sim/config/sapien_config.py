"""SAPIEN-specific default configuration.

Reads SAPIEN engine parameters from environment variables with RLIVE_SIM_ prefix.
These defaults can be overridden via __init__ parameters.
"""

import os
from dataclasses import dataclass

# ============================================================================
# CONTROLLER CONFIGURATION
# ============================================================================

MAX_SPEED_MS: float = float(os.getenv("RLIVE_SIM_SAPIEN_MAX_SPEED_MS", "0.5"))
"""
m/s that maps to speed=255.

default: 0.5
"""

ACCELERATION_TIME: float = float(os.getenv("RLIVE_SIM_SAPIEN_ACCELERATION_TIME", "0.3"))
"""
Seconds to ramp 0 to max_speed.

default: 0.3
"""

DECELERATION_TIME: float = float(os.getenv("RLIVE_SIM_SAPIEN_DECELERATION_TIME", "0.2"))
"""
Seconds to brake max_speed to 0.

default: 0.2
"""

CONTROLLER_KP: float = float(os.getenv("RLIVE_SIM_SAPIEN_CONTROLLER_KP", "0.1"))
"""
Proportional gain for PD controller.

default: 0.1
"""

CONTROLLER_KD: float = float(os.getenv("RLIVE_SIM_SAPIEN_CONTROLLER_KD", "0.1"))
"""
Derivative gain for PD controller.

default: 0.1
"""

# ============================================================================
# ROBOT CONFIGURATION
# ============================================================================

ROBOT_RADIUS: float = float(os.getenv("RLIVE_SIM_SAPIEN_ROBOT_RADIUS", "0.0365"))
"""
Robot radius in meters.

default: 0.0365
"""

ROBOT_MASS: float = float(os.getenv("RLIVE_SIM_SAPIEN_ROBOT_MASS", "0.12"))
"""
Robot mass in kg.

default: 0.12
"""

FRICTION: float = float(os.getenv("RLIVE_SIM_SAPIEN_FRICTION", "0.8"))
"""
Friction coefficient.

default: 0.8
"""

RESTITUTION: float = float(os.getenv("RLIVE_SIM_SAPIEN_RESTITUTION", "0.1"))
"""
Restitution coefficient.

default: 0.1
"""

# ============================================================================
# ROBOT LOADING CONFIGURATION
# ============================================================================

ROBOT_TYPE: str = str(os.getenv("RLIVE_SIM_SAPIEN_ROBOT_TYPE", "glb"))
"""
Robot loading type: 'sapien', 'urdf', or 'glb'.

- sapien: Programmatically built sphere with ground plane (fastest, default)
- urdf: Load from URDF file with ground plane (resources/sapian/sphere_robot.urdf)
- glb: Load from GLB/GLTF file (supports euro_box without ground plane)

default: 'sapien'
"""

ROBOT_PATH: str | None = os.getenv("RLIVE_SIM_SAPIEN_ROBOT_PATH", None)
"""
Optional custom path to robot file (for URDF or GLB).
If None, uses default paths from RESOURCES_DIR.

default: None
"""

# ============================================================================
# SIMULATION CONFIGURATION
# ============================================================================

SIM_DT: float = float(os.getenv("RLIVE_SIM_SAPIEN_SIM_DT", "0.004"))
"""
Physics timestep in seconds.

default: 0.004
"""

# ============================================================================
# DEFAULTS DATACLASS
# ============================================================================

@dataclass
class SapienDefaults:
    """SAPIEN engine default configuration."""
    
    # Controller configuration
    max_speed_ms: float = MAX_SPEED_MS
    acceleration_time: float = ACCELERATION_TIME
    deceleration_time: float = DECELERATION_TIME
    controller_kp: float = CONTROLLER_KP
    controller_kd: float = CONTROLLER_KD
    
    # Robot loading configuration
    robot_type: str = ROBOT_TYPE
    robot_path: str | None = ROBOT_PATH
    
    # Robot configuration
    robot_radius: float = ROBOT_RADIUS
    robot_mass: float = ROBOT_MASS
    friction: float = FRICTION
    restitution: float = RESTITUTION
    
    # Simulation configuration
    sim_dt: float = SIM_DT


SAPIEN_DEFAULTS = SapienDefaults()

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "MAX_SPEED_MS",
    "ACCELERATION_TIME",
    "DECELERATION_TIME",
    "CONTROLLER_KP",
    "CONTROLLER_KD",
    "ROBOT_TYPE",
    "ROBOT_PATH",
    "ROBOT_RADIUS",
    "ROBOT_MASS",
    "FRICTION",
    "RESTITUTION",
    "SIM_DT",
    "SapienDefaults",
    "SAPIEN_DEFAULTS",
]
