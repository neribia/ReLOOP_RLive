"""Sphero firmware emulator for simulation.

Emulates the internal trapezoidal-velocity controller of the real Sphero BOLT+.
When the engine receives a roll(heading, speed, duration) command it delegates
to this class, which then:

    1. Snaps the internal yaw axis to the new heading instantly (matching the
       real dt=0 macro-step behaviour where no time passes between commands).
    2. Ramps speed up from the current value to the target using a linear
       acceleration ramp (trapezoidal profile).
    3. Holds the cruise speed for the remaining manoeuvre time.
    4. Ramps speed back down to 0 during the deceleration window.

The engine calls ``get_forces(sim_dt, vx, vy)`` every MuJoCo timestep (1 ms
by default) to obtain the (fx, fy) force pair that should be written to
``data.ctrl``.

State machine
-------------
IDLE         → no movement, forces = 0
ACCELERATING → ramping up to target speed
CRUISING     → holding target speed
DECELERATING → ramping down to 0
"""

import math
from enum import Enum, auto

import numpy as np

from rlive_sim.config.sapien_config import SAPIEN_DEFAULTS as sapien_cfg
from rlive_common.utils import get_logger

logger = get_logger(__name__)


class Phase(Enum):
    IDLE = auto()
    MOVE = auto()
    DECELERATING = auto()


class SpheroController:
    """Firmware emulator that converts roll commands to per-timestep forces.

    Always operates in macro-step mode: heading changes are instantaneous,
    matching the real Sphero BOLT+ where roll(heading, speed, duration) is
    an atomic command that executes fully before the next observation.

    Args:
        max_speed_ms:      Speed in m/s that maps to speed=255.
        max_accel_ms2:     Maximum acceleration in m/s² (used for ramp timing).
        kp:                Proportional gain of the velocity-tracking PID .
        kd:                Derivative gain of the velocity-tracking PID.
        ki:                Integral gain of the velocity-tracking PID .
        ki_max_mag:            Maximum magnitude of the integral term (anti-windup).
        mass:              Sphere mass in kg (used for force scaling).
    """

    def __init__(
        self,
        max_speed_ms: float = sapien_cfg.max_speed_ms,
        max_accel_ms2: float = sapien_cfg.max_accel_ms2,
        kp: float = 12,
        kd: float = 0.8,
        ki: float = 0.01,
        ki_max_mag: float = 5.0,
        mass: float = 0.12,
    ) -> None:
        self.max_speed_ms = max_speed_ms
        self.max_accel_ms2 = max_accel_ms2
        self.proportional_gain = kp
        self.derivative_gain = kd
        self.integral_gain = ki
        self.ki_max_mag = ki_max_mag
        self.mass = mass

        # --- Vectorized state ---
        self._heading_deg: float = 0.0
        self._target_speed_ms: float = 0.0
        self._goal_speed_ms: float = 0.0

        # PID State as 2D Vectors [x, y]
        self._integral_error = np.zeros(2)
        self._prev_error = np.zeros(2)

        # Trapezoidal profile bookkeeping
        self._phase: Phase = Phase.IDLE
        self._time_elapsed: float = 0.0
        self._time_duration: float = 0.0
        self._steps:int  = 0

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def _reset(self) -> None:
        """Reset controller state (call when env resets)."""
        self._phase = Phase.IDLE
        self._time_elapsed = 0.0
        self._target_speed_ms = 0.0
        self._integral_error = np.zeros(2)
        self._prev_error = np.zeros(2)

    def command(self, heading_deg: float, speed_255: float, duration_s: float) -> None:
        """Issue a new roll command.

        Heading changes are always instantaneous (macro-step mode only),
        matching the real Sphero BOLT+ roll() API behaviour.

        Args:
            heading_deg: **Absolute** heading in CCW degrees (right-hand +Z,
                Y-up world coords). 0° = +X (right), 90° = +Y (forward).
                Callers must convert CW Bolt relative deltas before passing here.
            speed_255:   Speed on the 0-255 BOLT+ scale.
            duration_s:  Time of target_speed duration in seconds.
        """
        logger.debug(f"Heading: {heading_deg}° | Speed: {speed_255} | Duration: {duration_s}s")
        self._reset()
        self._heading_deg = _normalise_heading(heading_deg)
        self._goal_speed_ms = (speed_255 / 255.0) * self.max_speed_ms
        self._phase = Phase.MOVE
        self._time_duration = duration_s

    def _calc_target_speed_ms(self, sim_dt, goal, target):
        diff = goal - target
        step = self.max_accel_ms2 * sim_dt
        if abs(diff) < step:
            target = goal
        else:
            target += math.copysign(step, diff)
        return target

    def _make_update(self, sim_dt, current_speed_vec: np.ndarray) -> float:
        if self._phase == Phase.IDLE:
            return 0.0

        current_speed = np.linalg.norm(current_speed_vec)
        self._time_elapsed += sim_dt
        self._steps += 1

        if self._phase == Phase.MOVE:
            self._target_speed_ms = self._calc_target_speed_ms(sim_dt, self._goal_speed_ms, self._target_speed_ms)
            if self._time_elapsed >= self._time_duration:
                self._phase = Phase.DECELERATING
                self._goal_speed_ms = 0.0
            return self._target_speed_ms

        elif self._phase == Phase.DECELERATING:
            self._target_speed_ms = self._calc_target_speed_ms(sim_dt, self._goal_speed_ms, self._target_speed_ms)
            if current_speed <= 0.03: # TODO: Set value
                self._phase = Phase.IDLE
                self._integral_error = np.zeros(2)
        return self._target_speed_ms

    def get_velocity(self, sim_dt: float, vx: float, vy: float) -> tuple[float, float]:
        """Compute (vx, vy) target velocities for the next timestep.

        Returns:
            (vx, vy): Target velocity in m/s.
        """
        current_vel = np.array([vx, vy])

        # 1. Update the internal trapezoidal ramp to get the current target scalar speed
        target_speed = self._make_update(sim_dt, current_vel)

        # 2. Convert scalar speed + heading into a 2D velocity vector
        rad = math.radians(self._heading_deg)
        target_vel_x = math.cos(rad) * target_speed
        target_vel_y = math.sin(rad) * target_speed

        # 3. Optional logging (matching your previous style)
        # if self._steps % 5 == 0:
            # print(f"Time: {self._time_elapsed:.3f}/{self._time_duration} | Phase: {self._phase} | Target Vel: ({target_vel_x:.4f}, {target_vel_y:.4f})")

        return float(target_vel_x), float(target_vel_y)

    def get_forces(self, sim_dt: float, vx: float, vy: float) -> tuple[float, float]:
        """Compute (fx, fy) forces for the next MuJoCo timestep.

        Args:
            sim_dt: Simulation timestep in seconds (typically 0.004).
            vx:     Current sphere velocity along world-X (from qvel[0]).
            vy:     Current sphere velocity along world-Y (from qvel[1]).

        Returns:
            (fx, fy): Forces in Newton.
        """
        current_vel = np.array([vx, vy])
        target_speed = self._make_update(sim_dt, current_vel)

        # 1. Calculate Target Velocity Vector
        rad = math.radians(self._heading_deg)
        target_vel = np.array([math.cos(rad), math.sin(rad)]) * target_speed

        # 2. Calculate Velocity Error Vector
        error = target_vel - current_vel

        # 3. PID Terms (Vector Math)
        # Proportional
        proportional_term = self.proportional_gain * error

        # Integral with Anti-Windup (Clamping the vector magnitude)
        self._integral_error += error * sim_dt
        integral_magnitude = np.linalg.norm(self._integral_error)
        if integral_magnitude > self.ki_max_mag and integral_magnitude != 0:  # Maximum integration limit
            self._integral_error = (self._integral_error / integral_magnitude) * 5.0
        integral_term = self.integral_gain * self._integral_error

        # Derivative (Change in error)
        derivative_term = self.derivative_gain * (error - self._prev_error) / sim_dt

        # 4. Final Force Vector
        target_accel = proportional_term + integral_term + derivative_term
        accel_magnitude = np.linalg.norm(target_accel)

        if accel_magnitude > self.max_accel_ms2 and accel_magnitude > 1e-9:
            # Scale acceleration down to the hardware limit
            effective_accel = (target_accel / accel_magnitude) * self.max_accel_ms2
        else:
            effective_accel = target_accel

        final_force = self.mass *  effective_accel

        # 5. Update state for next step
        self._prev_error = error
        if self._steps % 5 == 0:
            print(f"Time: {self._time_elapsed:.3f}/{self._time_duration}"
                  f"Phase: {self._phase} | Force: ({final_force[0]:.4f}, {final_force[1]:.4f}) | "
                  f"Speed(c,t,g): ({current_vel[0]:.4f}, {current_vel[1]:.4f}), {target_vel}, {np.linalg.norm(current_vel):.3f}/{self._goal_speed_ms:.3f}")

        return float(final_force[0]), float(final_force[1])

    # ── Read-only properties ──────────────────────────────────────────────

    @property
    def heading_deg(self) -> float:
        """Current yaw heading in degrees."""
        return self._heading_deg

    @property
    def phase(self) -> Phase:
        """Current phase of the robot."""
        return self._phase


# ── Module-level helpers ──────────────────────────────────────────────────────


def _normalise_heading(deg: float) -> float:
    """Wrap heading to [-180, 180)."""
    return ((deg + 180.0) % 360.0) - 180.0


def _heading_delta(current: float, target: float) -> float:
    """Shortest signed angular distance from current -> target (degrees)."""
    return (target - current + 180.0) % 360.0 - 180.0