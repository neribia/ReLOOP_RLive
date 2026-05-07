"""calib_sim.py — Pure-Python Sphero simulation using ``SpheroController``.

This module wraps ``SpheroController.get_velocity()`` (from the ``rlive-sim``
workspace package, installed via ``uv``) in a simple 1-D integration loop.
No SAPIEN or MuJoCo installation is required — the controller only depends on
``numpy`` and ``rlive-common``.

How the simulation works
-------------------------
1. A fresh ``SpheroController`` is created with the candidate
   ``max_speed_ms`` and ``max_accel_ms2`` values.
2. ``controller.command(heading_deg=0, speed_255=..., duration_s=...)`` is
   called once to start the trapezoidal velocity ramp.
3. The simulation loop calls ``get_velocity(sim_dt, vx, 0.0)`` every
   ``sim_dt`` seconds.  The returned target velocity ``tvx`` is immediately
   fed back as the *current* velocity on the next call.  Because the
   controller's deceleration stop condition is ``|current_speed| <= 0.03``,
   and ``current_speed = |tvx|`` during the ramp-down, the phase transition
   to IDLE fires naturally when the target reaches near-zero.
4. Distance is integrated as ``d += vx * sim_dt`` each step and plateaus
   once the controller returns to IDLE.

The simulation is purely 1-D (heading fixed at 0 °, +X axis), which is
sufficient for calibrating the speed profile against straight-line roll data.
"""

from __future__ import annotations

import numpy as np

from rlive_sim.engine.sapien.sphero_controller import SpheroController

SIM_DT = 0.004 # Same as Sapien


def simulate_run(
    speed_255: float,
    duration_s: float,
    max_speed_ms: float,
    max_accel_ms2: float,
    sim_dt: float = SIM_DT,
    total_time_s: float = 7.5,
) -> dict:
    """Simulate one straight-line roll command and return time-series data.

    Args:
        speed_255:     Speed on the 0-255 Sphero BOLT+ scale.
        duration_s:    Cruise duration passed to ``SpheroController.command()``.
                       Use the estimated real-motion duration so that the
                       simulated command window matches the physical run.
        max_speed_ms:  Candidate max-speed parameter (m/s); speed 255 maps
                       to this value inside the controller.
        max_accel_ms2: Candidate max-acceleration parameter (m/s²); controls
                       how fast the trapezoidal ramp rises and falls.
        sim_dt:        Integration timestep in seconds (default 1 ms).
        total_time_s:  Total recording window; the loop runs until ``t``
                       exceeds this value (default 7.5 s).

    Returns:
        dict with keys:

        - ``"time"``           – ``np.ndarray``, shape (N,), seconds
        - ``"speed"``          – ``np.ndarray``, shape (N,), m/s
        - ``"distance"``       – ``np.ndarray``, shape (N,), metres
        - ``"max_speed"``      – ``float``, peak speed during the run (m/s)
        - ``"final_distance"`` – ``float``, distance at end of recording (m)
    """
    # Create a fresh controller with the candidate parameters.
    # PID gains and mass are kept at their defaults — only the two calibration
    # parameters are varied.
    ctrl = SpheroController(
        max_speed_ms=max_speed_ms,
        max_accel_ms2=max_accel_ms2,
    )

    # Issue a straight-line roll (heading 0 ° = +X axis)
    ctrl.command(heading_deg=0.0, speed_255=float(speed_255), duration_s=duration_s)

    # Pre-allocate arrays — avoids repeated Python list appends in the hot loop.
    n_steps = int(round(total_time_s / sim_dt)) + 1
    t_arr = np.arange(n_steps, dtype=float) * sim_dt
    v_arr = np.zeros(n_steps, dtype=float)
    d_arr = np.zeros(n_steps, dtype=float)

    for i in range(n_steps - 1):
        # Forward-Euler: integrate distance with the velocity at step i (current
        # time), then advance the controller to get the velocity for step i+1.
        # Previously the order was reversed (d += new_vx * dt), which
        # over-estimated distance during acceleration and under-estimated during
        # deceleration.
        tvx, _tvy = ctrl.get_velocity(sim_dt, v_arr[i], 0.0)
        d_arr[i + 1] = d_arr[i] + v_arr[i] * sim_dt
        v_arr[i + 1] = tvx

    return {
        "time":           t_arr,
        "speed":          v_arr,
        "distance":       d_arr,
        "max_speed":      float(np.max(v_arr)),
        "final_distance": float(d_arr[-1]),
    }

