"""Sphero Bolt Plus hardware configuration.

Reads Sphero Bolt Plus parameters from environment variables with RLIVE_SIM_ prefix.
These defaults can be overridden via __init__ parameters.

Reference: Sphero Bolt Plus Technical Specifications
"""

import os
from dataclasses import dataclass



# ============================================================================
# PHYSICAL DIMENSIONS
# ============================================================================

DIAMETER_MM: float = float(os.getenv("RLIVE_SIM_BOLT_DIAMETER_MM", "75.0"))
"""
Sphero Bolt Plus diameter in millimeters.

default: 75.0 mm
"""

RADIUS_M: float = float(os.getenv("RLIVE_SIM_BOLT_RADIUS_M", "0.0375"))
"""
Sphero Bolt Plus radius in meters.

default: 0.0375 m (37.5 mm)
"""

# ============================================================================
# WEIGHT & MASS
# ============================================================================

MASS_KG: float = float(os.getenv("RLIVE_SIM_BOLT_MASS_KG", "0.175"))
"""
Sphero Bolt Plus mass in kilograms.

default: 0.175 kg (175 grams)
"""

# ============================================================================
# PROPULSION & MOTION
# ============================================================================

MAX_SPEED_MS: float = float(os.getenv("RLIVE_SIM_BOLT_MAX_SPEED_MS", "2.5"))
"""
Maximum speed in meters per second.

default: 2.5 m/s (approximately 9 km/h)
"""

MAX_SPEED_CMD: int = int(os.getenv("RLIVE_SIM_BOLT_MAX_SPEED_CMD", "255"))
"""
Maximum speed command value (8-bit).

default: 255
"""

ACCELERATION_MS2: float = float(os.getenv("RLIVE_SIM_BOLT_ACCELERATION_MS2", "2.0"))
"""
Maximum acceleration in meters per second squared.

default: 2.0 m/s²
"""

DECELERATION_MS2: float = float(os.getenv("RLIVE_SIM_BOLT_DECELERATION_MS2", "2.0"))
"""
Maximum deceleration in meters per second squared.

default: 2.0 m/s²
"""

# ============================================================================
# ROTATIONAL DYNAMICS
# ============================================================================

MOMENT_OF_INERTIA: float = float(os.getenv("RLIVE_SIM_BOLT_MOMENT_OF_INERTIA", "0.000109"))
"""
Moment of inertia for a sphere (I = 2/5 * m * r²).

Calculated: 2/5 * 0.175 * 0.0375² ≈ 0.000109 kg·m²

default: 0.000109 kg·m²
"""

FRICTION_COEFFICIENT: float = float(os.getenv("RLIVE_SIM_BOLT_FRICTION_COEFFICIENT", "0.6"))
"""
Friction coefficient between Bolt and surface.

default: 0.6
"""

RESTITUTION_COEFFICIENT: float = float(os.getenv("RLIVE_SIM_BOLT_RESTITUTION_COEFFICIENT", "0.15"))
"""
Coefficient of restitution (elasticity) on collision.

default: 0.15
"""

# ============================================================================
# POWER & BATTERY
# ============================================================================

BATTERY_VOLTAGE_V: float = float(os.getenv("RLIVE_SIM_BOLT_BATTERY_VOLTAGE_V", "3.7"))
"""
Battery nominal voltage in volts (1S LiPo).

default: 3.7 V
"""

BATTERY_CAPACITY_MAH: float = float(os.getenv("RLIVE_SIM_BOLT_BATTERY_CAPACITY_MAH", "2200.0"))
"""
Battery capacity in milliamp-hours.

default: 2200 mAh
"""

BATTERY_ENERGY_WH: float = float(os.getenv("RLIVE_SIM_BOLT_BATTERY_ENERGY_WH", "8.14"))
"""
Battery energy in watt-hours (voltage × capacity ÷ 1000).

Calculated: 3.7 × 2200 ÷ 1000 = 8.14 Wh

default: 8.14 Wh
"""

# ============================================================================
# MOTOR SPECIFICATIONS
# ============================================================================

NUM_MOTORS: int = int(os.getenv("RLIVE_SIM_BOLT_NUM_MOTORS", "2"))
"""
Number of drive motors.

default: 2 (differential drive)
"""

MOTOR_MAX_TORQUE_NM: float = float(os.getenv("RLIVE_SIM_BOLT_MOTOR_MAX_TORQUE_NM", "0.5"))
"""
Maximum torque per motor in Newton-meters.

default: 0.5 N·m
"""

# ============================================================================
# CONTROL PARAMETERS
# ============================================================================

CONTROL_FREQUENCY_HZ: float = float(os.getenv("RLIVE_SIM_BOLT_CONTROL_FREQUENCY_HZ", "100.0"))
"""
Control loop frequency in Hz.

default: 100 Hz (10 ms cycle time)
"""

CONTROL_DT_S: float = float(os.getenv("RLIVE_SIM_BOLT_CONTROL_DT_S", "0.01"))
"""
Control timestep in seconds (1 / CONTROL_FREQUENCY_HZ).

default: 0.01 s
"""

# ============================================================================
# SENSOR SPECIFICATIONS
# ============================================================================

IMU_UPDATE_RATE_HZ: float = float(os.getenv("RLIVE_SIM_BOLT_IMU_UPDATE_RATE_HZ", "100.0"))
"""
Inertial Measurement Unit update rate in Hz.

default: 100 Hz
"""

MAX_ANGULAR_VELOCITY_RADS: float = float(os.getenv("RLIVE_SIM_BOLT_MAX_ANGULAR_VELOCITY_RADS", "20.0"))
"""
Maximum angular velocity in radians per second.

default: 20.0 rad/s
"""

MAX_LINEAR_ACCELERATION_G: float = float(os.getenv("RLIVE_SIM_BOLT_MAX_LINEAR_ACCELERATION_G", "8.0"))
"""
Maximum linear acceleration in gravitational units (g).

default: 8.0 g
"""

# ============================================================================
# LED & DISPLAY
# ============================================================================

NUM_LEDS: int = int(os.getenv("RLIVE_SIM_BOLT_NUM_LEDS", "12"))
"""
Number of programmable LED matrix cells.

default: 12 (4x3 matrix)
"""

# ============================================================================
# DEFAULTS DATACLASS
# ============================================================================

@dataclass
class BoltDefaults:
    """Sphero Bolt Plus hardware configuration parameters."""
    
    # Physical dimensions
    diameter_mm: float = DIAMETER_MM
    radius_m: float = RADIUS_M
    
    # Weight & mass
    mass_kg: float = MASS_KG
    
    # Propulsion & motion
    max_speed_ms: float = MAX_SPEED_MS
    max_speed_cmd: int = MAX_SPEED_CMD
    acceleration_ms2: float = ACCELERATION_MS2
    deceleration_ms2: float = DECELERATION_MS2
    
    # Rotational dynamics
    moment_of_inertia: float = MOMENT_OF_INERTIA
    friction_coefficient: float = FRICTION_COEFFICIENT
    restitution_coefficient: float = RESTITUTION_COEFFICIENT
    
    # Power & battery
    battery_voltage_v: float = BATTERY_VOLTAGE_V
    battery_capacity_mah: float = BATTERY_CAPACITY_MAH
    battery_energy_wh: float = BATTERY_ENERGY_WH
    
    # Motor specifications
    num_motors: int = NUM_MOTORS
    motor_max_torque_nm: float = MOTOR_MAX_TORQUE_NM
    
    # Control parameters
    control_frequency_hz: float = CONTROL_FREQUENCY_HZ
    control_dt_s: float = CONTROL_DT_S
    
    # Sensor specifications
    imu_update_rate_hz: float = IMU_UPDATE_RATE_HZ
    max_angular_velocity_rads: float = MAX_ANGULAR_VELOCITY_RADS
    max_linear_acceleration_g: float = MAX_LINEAR_ACCELERATION_G
    
    # LED & display
    num_leds: int = NUM_LEDS


BOLT_DEFAULTS = BoltDefaults()

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Physical dimensions
    "DIAMETER_MM",
    "RADIUS_M",
    
    # Weight & mass
    "MASS_KG",
    
    # Propulsion & motion
    "MAX_SPEED_MS",
    "MAX_SPEED_CMD",
    "ACCELERATION_MS2",
    "DECELERATION_MS2",
    
    # Rotational dynamics
    "MOMENT_OF_INERTIA",
    "FRICTION_COEFFICIENT",
    "RESTITUTION_COEFFICIENT",
    
    # Power & battery
    "BATTERY_VOLTAGE_V",
    "BATTERY_CAPACITY_MAH",
    "BATTERY_ENERGY_WH",
    
    # Motor specifications
    "NUM_MOTORS",
    "MOTOR_MAX_TORQUE_NM",
    
    # Control parameters
    "CONTROL_FREQUENCY_HZ",
    "CONTROL_DT_S",
    
    # Sensor specifications
    "IMU_UPDATE_RATE_HZ",
    "MAX_ANGULAR_VELOCITY_RADS",
    "MAX_LINEAR_ACCELERATION_G",
    
    # LED & display
    "NUM_LEDS",
    
    # Dataclass and instance
    "BoltDefaults",
    "BOLT_DEFAULTS",
]

