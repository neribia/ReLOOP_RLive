"""Math utility functions for simulation.

This module provides common mathematical operations for simulation,
such as coordinate transformations and physics calculations.
"""

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


def quat_to_euler(q: NDArray[np.float32], degrees: bool = False) -> NDArray[np.float32]:
    """Convert a Quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw].

    Uses the Z-Y-X intrinsic (X-Y-Z extrinsic) convention, which is the
    inverse of ``euler_to_quat``.

    Args:
        q: Quaternion as array-like [w, x, y, z].
        degrees: If True, return angles in degrees. Defaults to False (radians).

    Returns:
        numpy.ndarray: Euler angles as [roll, pitch, yaw].

    Examples:
        Round-trip conversion:

            q = euler_to_quat(0, 0, 45, degrees=True)
            roll, pitch, yaw = quat_to_euler(q, degrees=True)  # [0, 0, 45]
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])

    # Roll (rotation around x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (rotation around y-axis)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2, sinp)  # Gimbal lock
    else:
        pitch = np.arcsin(sinp)

    # Yaw (rotation around z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    result = np.array([roll, pitch, yaw], dtype=np.float32)
    if degrees:
        result = np.rad2deg(result).astype(np.float32)
    return result


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

