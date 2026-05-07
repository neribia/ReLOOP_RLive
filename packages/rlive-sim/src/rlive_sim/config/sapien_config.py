"""SAPIEN-specific default configuration.

Reads SAPIEN engine parameters from environment variables with RLIVE_SIM_ prefix.
These defaults can be overridden via __init__ parameters.
"""

import os
from dataclasses import dataclass, field

# ============================================================================
# CONTROLLER CONFIGURATION
# ============================================================================

MAX_SPEED_MS: float = float(os.getenv("RLIVE_SIM_SAPIEN_MAX_SPEED_MS", "1.825"))
"""
m/s that maps to speed=255.

default: 1.825 
"""

MAX_ACCEL_MS2: float = float(os.getenv("RLIVE_SIM_SAPIEN_MAX_ACCEL_MS2", "0.575"))
"""
Max acceleration for the Controller in m/s2

default: 0.575
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

ROBOT_MASS: float = float(os.getenv("RLIVE_SIM_SAPIEN_ROBOT_MASS", "0.28"))
"""
Robot mass in kg.

default: 0.28
"""

# ============================================================================
# ROBOT LOADING CONFIGURATION
# ============================================================================

ROBOT_TYPE: str = str(os.getenv("RLIVE_SIM_SAPIEN_ROBOT_TYPE", "glb"))
"""
Robot loading type: 'sapien' or 'glb'.

- sapien: Programmatically built sphere with ground plane (fastest, default)
- glb: Load from GLB/GLTF file

default: 'glb'
"""

ROBOT_PATH: str | None = os.getenv("RLIVE_SIM_SAPIEN_ROBOT_PATH", None)
"""
Optional custom path to robot file (for URDF or GLB).
If None, uses default paths from RESOURCES_DIR.

default: None
"""

RESET_AREA: list[float] = [
    float(x) for x in os.getenv("RLIVE_SIM_SAPIEN_RESET_AREA", "-0.35,0.35,-0.15,0.15").split(",")
]
"""
Bounding box for random reset area [min_x, max_x, min_y, max_y].

default: [-0.35, 0.35, -0.15, 0.15]
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
    max_accel_ms2: float = MAX_ACCEL_MS2
    controller_kp: float = CONTROLLER_KP
    controller_kd: float = CONTROLLER_KD
    
    # Robot loading configuration
    robot_type: str = ROBOT_TYPE
    robot_path: str | None = ROBOT_PATH
    reset_area: list[float] = field(default_factory=lambda: RESET_AREA)

    # Robot configuration
    robot_radius: float = ROBOT_RADIUS
    robot_mass: float = ROBOT_MASS
    
    # Simulation configuration
    sim_dt: float = SIM_DT


SAPIEN_DEFAULTS = SapienDefaults()

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "MAX_SPEED_MS",
    "CONTROLLER_KP",
    "CONTROLLER_KD",
    "ROBOT_TYPE",
    "ROBOT_PATH",
    "ROBOT_RADIUS",
    "ROBOT_MASS",
    "SIM_DT",
    "SapienDefaults",
    "SAPIEN_DEFAULTS",
]
