"""Math utility functions for simulation.

This module provides common mathematical operations for simulation,
such as coordinate transformations and physics calculations.
"""
from typing import Any

import numpy as np
from numpy.typing import NDArray


def deg_to_rad(degrees: float | NDArray) -> float | NDArray:
    """Convert degrees to radians.

    Args:
        degrees: Angle(s) in degrees

    Returns:
        Angle(s) in radians
    """
    return np.deg2rad(degrees)


def rad_to_deg(radians: float | NDArray) -> float | NDArray:
    """Convert radians to degrees.

    Args:
        radians: Angle(s) in radians

    Returns:
        Angle(s) in degrees
    """
    return np.rad2deg(radians)


def euler_to_quat(roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0, degrees: bool = False) -> NDArray[np.float32]:
    """Convert Euler angles (roll, pitch, yaw) to Quaternion [w, x, y, z].
    
    This implementation corresponds to Z-Y-X intrinsic (or X-Y-Z extrinsic) rotation order,
    which is standard in robotics (e.g., ROS).
    
    Args:
        roll: Rotation around X-axis (in radians, unless degrees=True).
        pitch: Rotation around Y-axis (in radians, unless degrees=True).
        yaw: Rotation around Z-axis (in radians, unless degrees=True).
        degrees: If True, interpret input angles as degrees. Defaults to False.
        
    Returns:
        numpy.ndarray: Quaternion as [w, x, y, z].
    """
    if degrees:
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    # Calculate quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z], dtype=np.float32)

