"""Sphero firmware emulator for MuJoCo simulation.

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

from __future__ import annotations

import math
from enum import Enum, auto


class _Phase(Enum):
    IDLE = auto()
    ACCELERATING = auto()
    CRUISING = auto()
    DECELERATING = auto()


class SpheroController:
    """Firmware emulator that converts roll commands to per-timestep forces.

    Always operates in macro-step mode: heading changes are instantaneous,
    matching the real Sphero BOLT+ where roll(heading, speed, duration) is
    an atomic command that executes fully before the next observation.

    Args:
        max_speed_ms:      Speed in m/s that maps to speed=255.
        acceleration_time: Seconds to ramp 0 -> max_speed_ms.
        deceleration_time: Seconds to ramp max_speed_ms -> 0.
        kp:                Proportional gain of the velocity-tracking PD.
        kd:                Derivative gain of the velocity-tracking PD.
        mass:              Sphere mass in kg (used for force scaling).
    """

    def __init__(
        self,
        max_speed_ms: float = 0.5,
        acceleration_time: float = 0.3,
        deceleration_time: float = 0.2,
        kp: float = 0.1,
        kd: float = 0.1,
        mass: float = 0.12,
    ) -> None:
        self.max_speed_ms = max_speed_ms
        self.acceleration_time = acceleration_time
        self.deceleration_time = deceleration_time
        self.kp = kp
        self.kd = kd
        self.mass = mass

        # --- mutable state ---
        self._heading_deg: float = 0.0
        self._current_speed_ms: float = 0.0
        self._target_speed_ms: float = 0.0
        self._prev_error_x: float = 0.0
        self._prev_error_y: float = 0.0

        # Trapezoidal profile bookkeeping
        self._phase: _Phase = _Phase.IDLE
        self._phase_elapsed: float = 0.0
        self._cruise_remaining: float = 0.0
        self._speed_at_accel_start: float = 0.0
        self._speed_at_decel_start: float = 0.0

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset controller state (call when env resets)."""
        self._heading_deg = 0.0
        self._current_speed_ms = 0.0
        self._target_speed_ms = 0.0
        self._prev_error_x = 0.0
        self._prev_error_y = 0.0
        self._phase = _Phase.IDLE
        self._phase_elapsed = 0.0
        self._cruise_remaining = 0.0
        self._speed_at_accel_start = 0.0
        self._speed_at_decel_start = 0.0

    def command(
        self,
        heading_deg: float,
        speed_255: float,
        duration_s: float,
    ) -> None:
        """Issue a new roll command.

        Heading changes are always instantaneous (macro-step mode only),
        matching the real Sphero BOLT+ roll() API behaviour.

        Args:
            heading_deg: Absolute heading in degrees (0 = +X axis).
            speed_255:   Speed on the 0-255 BOLT+ scale.
            duration_s:  Total manoeuvre duration in seconds.
        """
        self._heading_deg = _normalise_heading(heading_deg)
        self._target_speed_ms = (speed_255 / 255.0) * self.max_speed_ms
        self._speed_at_accel_start = self._current_speed_ms
        self._prev_error_x = 0.0
        self._prev_error_y = 0.0
        self._start_acceleration(duration_s)

    def get_forces(self, sim_dt: float, vx: float, vy: float) -> tuple[float, float]:
        """Compute (fx, fy) forces for the next MuJoCo timestep.

        Args:
            sim_dt: MuJoCo timestep in seconds (typically 0.001).
            vx:     Current sphere velocity along world-X (from qvel[0]).
            vy:     Current sphere velocity along world-Y (from qvel[1]).

        Returns:
            (fx, fy): Forces in Newtons to write into ``data.ctrl``.
        """
        if self._phase == _Phase.IDLE:
            self._prev_error_x = 0.0
            self._prev_error_y = 0.0
            return 0.0, 0.0

        self._phase_elapsed += sim_dt

        if self._phase == _Phase.ACCELERATING:
            progress = min(self._phase_elapsed / self.acceleration_time, 1.0)
            self._current_speed_ms = (
                self._speed_at_accel_start
                + (self._target_speed_ms - self._speed_at_accel_start) * progress
            )
            if self._phase_elapsed >= self.acceleration_time:
                self._current_speed_ms = self._target_speed_ms
                self._phase = _Phase.CRUISING
                self._phase_elapsed = 0.0

        elif self._phase == _Phase.CRUISING:
            self._current_speed_ms = self._target_speed_ms
            self._cruise_remaining -= sim_dt
            if self._cruise_remaining <= 0.0:
                self._phase = _Phase.DECELERATING
                self._phase_elapsed = 0.0
                self._speed_at_decel_start = self._current_speed_ms

        elif self._phase == _Phase.DECELERATING:
            progress = min(self._phase_elapsed / self.deceleration_time, 1.0)
            self._current_speed_ms = self._speed_at_decel_start * (1.0 - progress)
            if self._phase_elapsed >= self.deceleration_time:
                self._current_speed_ms = 0.0
                self._phase = _Phase.IDLE
                return 0.0, 0.0

        # ── PD velocity controller ────────────────────────────────────────
        heading_rad = math.radians(self._heading_deg)
        tvx = self._current_speed_ms * math.cos(heading_rad)
        tvy = self._current_speed_ms * math.sin(heading_rad)

        ex = tvx - vx
        ey = tvy - vy

        dex = (ex - self._prev_error_x) / max(sim_dt, 1e-9)
        dey = (ey - self._prev_error_y) / max(sim_dt, 1e-9)
        self._prev_error_x = ex
        self._prev_error_y = ey

        fx = self.kp * ex + self.kd * dex
        fy = self.kp * ey + self.kd * dey

        return float(fx), float(fy)

    # ── Read-only properties ──────────────────────────────────────────────

    @property
    def heading_deg(self) -> float:
        """Current yaw heading in degrees."""
        return self._heading_deg

    @property
    def speed_ms(self) -> float:
        """Current translational speed in m/s."""
        return self._current_speed_ms

    @property
    def is_idle(self) -> bool:
        """True when the controller has finished the current manoeuvre."""
        return self._phase == _Phase.IDLE

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _start_acceleration(self, duration_s: float) -> None:
        """Transition to ACCELERATING phase and compute cruise window."""
        accel_t = min(self.acceleration_time, duration_s)
        decel_t = min(self.deceleration_time, max(duration_s - accel_t, 0.0))
        cruise_t = max(duration_s - accel_t - decel_t, 0.0)

        self._phase = _Phase.ACCELERATING
        self._phase_elapsed = 0.0
        self._cruise_remaining = cruise_t
        self._speed_at_decel_start = self._target_speed_ms  # set for safety


# ── Module-level helpers ──────────────────────────────────────────────────────


def _normalise_heading(deg: float) -> float:
    """Wrap heading to [-180, 180)."""
    return ((deg + 180.0) % 360.0) - 180.0


def _heading_delta(current: float, target: float) -> float:
    """Shortest signed angular distance from current -> target (degrees)."""
    return (target - current + 180.0) % 360.0 - 180.0

