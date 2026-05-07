"""calib_sapien.py — Calibrate SpheroController using SapienIntegratedEngine.

Reuses the existing ``SapienIntegratedEngine`` (the same code path as runtime)
instead of building a separate scene.  One engine instance is created once per
``grid_search_sapien`` call; between grid cells only the controller parameters
are swapped and ``reset()`` is called to reposition the robot.

Public API
----------
``simulate_sapien_run(engine, speed_255, duration_s, max_speed_ms, max_accel_ms2)``
    Run one straight-line command on an existing engine instance.
    Returns a dict matching ``calib_sim.simulate_run`` (same keys).

``grid_search_sapien(runs, max_speed_range, max_accel_range, friction, use_percent_error)``
    Drop-in replacement for ``calib_search.grid_search``.
    Same signature + return value; uses SapienIntegratedEngine internally.
"""

from __future__ import annotations

import numpy as np

from rlive_sim.engine.sapien.sapien_integrated_engine import SapienIntegratedEngine
from rlive_sim.engine.core.base_physics_engine import PhysicsState
from rlive_sim.config import SAPIEN_DEFAULTS

from .calib_data import RunMetrics
from .calib_search import _compute_rmse, save_grid_results, load_grid_results
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────────────────
SIM_DT = SAPIEN_DEFAULTS.sim_dt  # must match runtime


def _make_engine() -> SapienIntegratedEngine:
    """Create a headless SapienIntegratedEngine using KinematicSpheroRobot on flat ground.

    Uses ``robot_type="glb_flat"`` — same robot as runtime (KinematicSpheroRobot)
    but on a flat ground, no euro box.
    """
    engine = SapienIntegratedEngine(
        use_viewer=False,
        robot_type="glb_flat",   # KinematicSpheroRobot + flat ground, NO eurobox

    )
    engine.setup_scene({})
    return engine


def _origin_state() -> PhysicsState:
    """Fixed initial state at world origin (x=0, y=0) for reproducible runs."""
    z = SAPIEN_DEFAULTS.robot_radius + 0.005  # resting on the ground
    return PhysicsState(
        position=[0.0, 0.0, z],
        velocity=[0.0, 0.0, 0.0],
        rotation=[0.0, 0.0, 0.0],
        angular_velocity=[0.0, 0.0, 0.0],
    )


# ── Single-run simulator ──────────────────────────────────────────────────────

def simulate_sapien_run(
    engine: SapienIntegratedEngine,
    speed_255: float,
    duration_s: float,
    max_speed_ms: float,
    max_accel_ms2: float,
    total_time_s: float = 7.5,
) -> dict:
    """Run one straight-line command on *engine* and return time-series data.

    Swaps the controller parameters on the existing engine (no scene rebuild),
    resets the robot to the origin, issues one roll command, and records
    position + speed at every physics step for ``total_time_s`` seconds.

    The interface matches ``calib_sim.simulate_run`` (same return keys).

    Args:
        engine:        An already-initialised ``SapienIntegratedEngine`` instance.
        speed_255:     Speed on the 0-255 Sphero BOLT+ scale.
        duration_s:    Cruise duration passed to the controller command.
        max_speed_ms:  Candidate max-speed parameter (m/s).
        max_accel_ms2: Candidate max-acceleration parameter (m/s²).
        total_time_s:  Total simulation window in seconds (default 7.5 s).
                       Should match the real recording window so that plots
                       cover the same time range.

    Returns:
        dict with keys:

        - ``"time"``           – ``np.ndarray``, shape (N,), seconds
        - ``"speed"``          – ``np.ndarray``, shape (N,), m/s (scalar XY)
        - ``"distance"``       – ``np.ndarray``, shape (N,), metres from start
        - ``"max_speed"``      – ``float``, peak speed during the run (m/s)
        - ``"final_distance"`` – ``float``, Euclidean XY distance from start (m)
    """
    from rlive_sim.engine.sapien.sphero_controller import Phase

    # Swap controller params without rebuilding the scene
    engine.controller.max_speed_ms  = max_speed_ms
    engine.controller.max_accel_ms2 = max_accel_ms2

    # Reset robot to fixed origin (x=0, y=0) for reproducible distance measurement
    engine.reset(initial_state=_origin_state())
    start_xy = np.array(engine._get_state().position[:2], dtype=float)

    # Issue straight-line command (heading 0° = +X axis, same as calib_sim)
    engine.controller.command(0.0, float(speed_255), duration_s)

    # Pre-allocate arrays to avoid repeated list appends in the hot loop
    n_steps = int(round(total_time_s / engine.sim_dt)) + 1
    t_arr = np.arange(n_steps, dtype=float) * engine.sim_dt
    v_arr = np.zeros(n_steps, dtype=float)
    d_arr = np.zeros(n_steps, dtype=float)

    for i in range(n_steps):
        # Sample state *before* stepping (so t_arr[i] matches the recorded values)
        state = engine._get_state()
        cur_xy  = np.array(state.position[:2], dtype=float)
        cur_vel = np.array(state.velocity[:2], dtype=float)

        d_arr[i] = float(np.linalg.norm(cur_xy - start_xy))
        v_arr[i] = float(np.linalg.norm(cur_vel))

        # Drive the robot: apply controller velocity while active, zero when idle
        if engine.controller.phase != Phase.IDLE:
            vx, vy = engine.controller.get_velocity(engine.sim_dt, cur_vel[0], cur_vel[1])
            engine.robot.set_root_linear_velocity(vx=vx, vy=vy, vz=0)
            engine.robot.set_root_angular_velocity(wx=0, wy=0, wz=0)
        else:
            engine.robot.set_root_linear_velocity(vx=0, vy=0, vz=0)
            engine.robot.set_root_angular_velocity(wx=0, wy=0, wz=0)

        engine.scene.step()
        engine.robot.update(engine.controller.heading_deg)

    return {
        "time":           t_arr,
        "speed":          v_arr,
        "distance":       d_arr,
        "max_speed":      float(np.max(v_arr)),
        "final_distance": float(d_arr[-1]),
    }


def make_simulate_fn():
    """Return a ``simulate_run``-compatible callable backed by a SAPIEN engine.

    Creates one ``SapienIntegratedEngine`` and closes it when the returned
    function is garbage-collected.  Use as a drop-in for ``calib_sim.simulate_run``
    in ``calib_plot.plot_results`` and ``speed_plot`` functions.

    Example::

        sim_fn = make_simulate_fn(friction=0.0)
        plot_results(runs, best_speed, best_accel, rmse_grid,
                     speed_range, accel_range, simulate_fn=sim_fn)


    Returns:
        Callable with signature
        ``(speed_255, duration_s, max_speed_ms, max_accel_ms2, **_) -> dict``.
    """
    engine = _make_engine()

    def _fn(speed_255, duration_s, max_speed_ms, max_accel_ms2, total_time_s: float = 7.5, **_):
        return simulate_sapien_run(engine, speed_255, duration_s, max_speed_ms, max_accel_ms2,
                                   total_time_s=total_time_s)

    # Attach engine reference so caller can close it explicitly if needed
    _fn.engine = engine  # type: ignore[attr-defined]
    return _fn


# ── Grid search ───────────────────────────────────────────────────────────────

def grid_search_sapien(
    runs: list[RunMetrics],
    max_speed_range: np.ndarray,
    max_accel_range: np.ndarray,
    use_percent_error: bool = False,
    cache_path: str | Path | None = None,
) -> tuple[float, float, np.ndarray]:
    """Grid search using SapienIntegratedEngine — drop-in for ``calib_search.grid_search``.

    Creates **one** engine instance and reuses it across all grid cells by
    swapping controller parameters and calling ``reset()`` between runs.

    Args:
        runs:              Real-robot data (list of ``RunMetrics``).
        max_speed_range:   1-D array of candidate ``max_speed_ms`` values.
        max_accel_range:   1-D array of candidate ``max_accel_ms2`` values.
        use_percent_error: If ``True`` use RMSPE (%) instead of RMSE (m).
        cache_path:        Optional ``.npz`` cache file path.  Cache hit skips
                           the SAPIEN search entirely; cache miss auto-saves
                           after the search. Pass ``None`` to disable.

    Returns:
        ``(best_max_speed_ms, best_max_accel_ms2, rmse_grid)`` — identical
        structure to ``calib_search.grid_search``.
    """
    from .speed_data import STEADY_S

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if cache_path is not None and Path(cache_path).exists():
        best_speed, best_accel, rmse_grid, _, _ = load_grid_results(cache_path)
        return best_speed, best_accel, rmse_grid

    # ── Cache miss — run the full SAPIEN search ───────────────────────────────

    n_speed = len(max_speed_range)
    n_accel = len(max_accel_range)
    rmse_grid = np.zeros((n_accel, n_speed), dtype=float)
    real_dists = np.array([r.real_final_dist for r in runs])

    unit = "%" if use_percent_error else "m"
    metric_name = "RMSPE" if use_percent_error else "RMSE"
    total = n_speed * n_accel
    done = 0
    report_every = max(1, total // 10)

    print(f"SAPIEN grid search ({metric_name}): "
          f"{n_speed} × {n_accel} = {total} combinations …")
    print("Building SAPIEN engine (once) …")

    engine = _make_engine()

    try:
        for i, max_accel in enumerate(max_accel_range):
            for j, max_speed in enumerate(max_speed_range):
                sim_dists = np.array([
                    simulate_sapien_run(
                        engine=engine,
                        speed_255=r.speed_cmd,
                        duration_s=STEADY_S,
                        max_speed_ms=max_speed,
                        max_accel_ms2=max_accel,
                        total_time_s=STEADY_S + 2.0,  # only run until robot stops + small buffer
                    )["final_distance"]
                    for r in runs
                ])

                rmse_grid[i, j] = _compute_rmse(real_dists, sim_dists, use_percent_error)
                done += 1
                if done % report_every == 0:
                    pct = 100 * done / total
                    bar = ("█" * int(pct // 10)).ljust(10)
                    print(f"  [{bar}] {pct:.0f}%  ({done}/{total})")
    finally:
        engine.close()

    best_idx = np.unravel_index(np.argmin(rmse_grid), rmse_grid.shape)
    best_accel = float(max_accel_range[best_idx[0]])
    best_speed = float(max_speed_range[best_idx[1]])
    best_error = float(rmse_grid[best_idx])

    print(f"\n  ✅ Best  max_speed_ms  = {best_speed:.4f} m/s")
    print(f"     Best  max_accel_ms2 = {best_accel:.4f} m/s²")
    print(f"     Distance {metric_name:<5}      = {best_error:.4f} {unit}")

    if cache_path is not None:
        save_grid_results(
            cache_path, best_speed, best_accel,
            rmse_grid, max_speed_range, max_accel_range,
            use_percent_error,
        )

    return best_speed, best_accel, rmse_grid









