"""calib_data.py — Data structures for Sim2Real calibration.

Responsibilities
----------------
- ``RunMetrics`` dataclass: holds all per-run state needed by the grid search
  and calibration plots (real metrics, raw arrays, ``sim_final_dist`` slot).
- ``df_to_run_metrics``: converts a DataFrame produced by
  ``speed_data.prepare_run_df`` into a ``RunMetrics`` instance.
- ``load_runs_from_dir``: thin wrapper — calls ``speed_data.load_all_runs``
  and converts each DataFrame via ``df_to_run_metrics``.
- ``load_combined``: same for combined-CSV mode (one file, speed-command column).

CSV loading lives entirely in ``speed_data`` — there is only one loader.

Expected single-run CSV columns
---------------------------------
    t_s, cmd, vx_ms, vy_ms, speed_ms, travel_distance_m,
    accel_x_ms2, accel_y_ms2, accel_z_ms2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from speed_data import prepare_run_df, load_all_runs, CSV_COLS


# ── Data structure ────────────────────────────────────────────────────────────

@dataclass
class RunMetrics:
    """All per-run data: real robot metrics, raw arrays, and a sim placeholder.

    Attributes:
        speed_cmd:       Speed command on the 0-255 Sphero scale.
        real_max_speed:  Peak |speed_ms| observed in the recording (m/s).
        real_max_accel:  Peak positive acceleration (m/s²).
        real_max_decel:  Peak negative acceleration (most negative, m/s²).
        real_final_dist: Median distance in the final 20 % of the recording (m).
        real_duration:   Estimated active-motion duration (s).
        sim_final_dist:  Simulated final distance — filled by the grid search.
        time:            Time axis from the CSV (s).
        real_speed:      (Smoothed) speed signal (m/s).
        real_distance:   Cumulative distance signal (m).
    """

    speed_cmd: int

    # Real robot metrics
    real_max_speed: float = 0.0
    real_max_accel: float = 0.0
    real_max_decel: float = 0.0
    real_final_dist: float = 0.0
    real_duration: float = 0.0

    # Sim result (filled in by calib_search.grid_search)
    sim_final_dist: float = 0.0

    # Raw arrays for plotting
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    real_speed: np.ndarray = field(default_factory=lambda: np.array([]))
    real_distance: np.ndarray = field(default_factory=lambda: np.array([]))


# ── Shared extraction logic ───────────────────────────────────────────────────

def _extract_metrics(
    t: np.ndarray,
    speed: np.ndarray,
    dist: np.ndarray,
    speed_cmd: int,
    smooth_window: int,
) -> RunMetrics:
    """Derive ``RunMetrics`` from raw time/speed/distance arrays."""
    # Remove duplicate timestamps (np.gradient divides by dt; dt=0 → NaN)
    _, unique_idx = np.unique(t, return_index=True)
    t     = t[unique_idx]
    speed = speed[unique_idx]
    dist  = dist[unique_idx]

    if smooth_window > 1:
        speed = uniform_filter1d(speed, size=smooth_window)

    accel = np.gradient(speed, t)

    # Final distance: median of the last 20 % of the *time* span (plateau region).
    # Using row count (int(len * 0.8)) was wrong with sparse/reduced data where
    # the last 20 % of rows could cover a tiny or huge time window.
    t_80 = t[0] + 0.8 * (t[-1] - t[0])
    plateau_mask = t >= t_80
    final_dist = (
        float(np.median(dist[plateau_mask])) if plateau_mask.any() else float(dist[-1])
    )

    # Duration estimation
    max_s = float(np.max(np.abs(speed)))
    onset_idx = int(np.argmax(np.abs(speed) > 0.01 * max_s)) if max_s > 0 else 0

    # Search for motion end within the first 80 % of the time span
    t_plateau_idx = int(np.searchsorted(t, t_80))
    t_plateau_idx = max(t_plateau_idx, 1)
    active_in_window = np.abs(speed[:t_plateau_idx]) > 0.10 * max_s
    if active_in_window.any():
        end_idx = t_plateau_idx - int(np.argmax(active_in_window[::-1])) - 1
    else:
        end_idx = t_plateau_idx

    raw_duration = (
        float(t[end_idx] - t[onset_idx]) if end_idx > onset_idx
        else float(t[-1] - t[0])
    )
    # Clamp to a sensible minimum — badly estimated durations from noisy/sparse
    # data caused downstream callers to issue degenerate short sim commands.
    duration_est = max(raw_duration, 0.1)

    return RunMetrics(
        speed_cmd=speed_cmd,
        real_max_speed=float(np.max(np.abs(speed))),
        real_max_accel=float(np.max(accel)),
        real_max_decel=float(np.min(accel)),
        real_final_dist=final_dist,
        real_duration=duration_est,
        time=t,
        real_speed=speed,
        real_distance=dist,
    )


# ── Public: DataFrame → RunMetrics ───────────────────────────────────────────

def df_to_run_metrics(
    df: pd.DataFrame,
    speed_cmd: int,
    smooth_window: int = 5,
) -> RunMetrics:
    """Convert a ``speed_data.prepare_run_df`` DataFrame to ``RunMetrics``.

    Uses the raw ``speed_ms`` sensor signal (not the derived variant) so that
    ``_extract_metrics`` can apply its own smoothing independently.

    Args:
        df:            DataFrame produced by :func:`speed_data.prepare_run_df`.
        speed_cmd:     Speed command integer (0-255).
        smooth_window: ``uniform_filter1d`` window; 1 disables smoothing.

    Returns:
        :class:`RunMetrics`
    """
    t     = df["t_s"].to_numpy(dtype=float)
    speed = df["speed_ms"].to_numpy(dtype=float)
    dist  = df["travel_distance_m"].to_numpy(dtype=float)
    return _extract_metrics(t, speed, dist, speed_cmd, smooth_window)


# ── Public: loaders (delegate CSV reading to speed_data) ─────────────────────

def load_runs_from_dir(
    data_dir: str | Path,
    pattern: str = "exp_speed_*.csv",
    smooth_window: int = 5,
) -> list[RunMetrics]:
    """Load per-run CSVs and return a speed-sorted list of ``RunMetrics``.

    Delegates file discovery and DataFrame preparation to
    :func:`speed_data.load_all_runs`, then converts each DataFrame via
    :func:`df_to_run_metrics`.  There is only one CSV loader in the codebase.

    Args:
        data_dir:      Directory containing the per-run CSV files.
        pattern:       Glob pattern (used only to validate; loading is handled
                       by ``speed_data``).
        smooth_window: Smoothing window passed to each run's metric extraction.

    Returns:
        List of :class:`RunMetrics` sorted ascending by ``speed_cmd``.
    """
    runs_dict = load_all_runs(data_dir=str(data_dir))
    runs = [
        df_to_run_metrics(df, speed_cmd, smooth_window)
        for speed_cmd, df in sorted(runs_dict.items())
    ]
    print(f"Converted {len(runs)} runs to RunMetrics")
    return runs
