"""calib_search.py — Grid search over ``(max_speed_ms, max_accel_ms2)``.

Responsibilities
----------------
- ``build_ranges``:    Constructs the ``np.linspace`` arrays for the 2-D grid.
- ``grid_search``:     Iterates over every ``(max_speed, max_accel)`` cell,
                       calls ``calib_sim.simulate_run`` for each run, and
                       returns the RMSE grid plus the best parameter pair.
- ``save_best_params``: Writes the result to a JSON file for downstream use.

The RMSE metric used is the **distance RMSE** — the root-mean-square error
between the real robot's final distance and the simulated final distance across
all speed commands.  This captures the dominant sim2real gap for straight-line
motion without requiring time-aligned signal registration.

Two error modes are supported (toggle via ``use_percent_error`` in
``grid_search``):

* **Absolute** (default, ``use_percent_error=False``):
    RMSE in metres — penalises large absolute gaps at high speeds uniformly.

* **Percent** (``use_percent_error=True``):
    RMSPE in % — each run contributes ``((real − sim) / real)²`` so slow
    runs (small absolute distances) are not dominated by fast runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .calib_data import RunMetrics
from .calib_sim import simulate_run

from .speed_data import STEADY_S


# ── Parameter range builder ───────────────────────────────────────────────────

def build_ranges(
    speed_min: float,
    speed_max: float,
    speed_steps: int,
    accel_min: float,
    accel_max: float,
    accel_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``np.linspace`` arrays for the 2-D parameter grid.

    Args:
        speed_min:   Lower bound of ``max_speed_ms`` search (m/s).
        speed_max:   Upper bound of ``max_speed_ms`` search (m/s).
        speed_steps: Number of evenly-spaced values along the speed axis.
        accel_min:   Lower bound of ``max_accel_ms2`` search (m/s²).
        accel_max:   Upper bound of ``max_accel_ms2`` search (m/s²).
        accel_steps: Number of evenly-spaced values along the acceleration axis.

    Returns:
        ``(max_speed_range, max_accel_range)`` — two 1-D ``np.ndarray`` s.
    """
    return (
        np.linspace(speed_min, speed_max, speed_steps),
        np.linspace(accel_min, accel_max, accel_steps),
    )


# ── Grid search ───────────────────────────────────────────────────────────────

def _compute_rmse(
    real_dists: np.ndarray,
    sim_dists: np.ndarray,
    use_percent_error: bool,
) -> float:
    """Compute RMSE (m) or RMSPE (%) between *real_dists* and *sim_dists*.

    Args:
        real_dists:        1-D array of real robot final distances (m).
        sim_dists:         1-D array of simulated final distances (m).
        use_percent_error: If ``True``, compute the root-mean-square *percentage*
                           error instead of the absolute RMSE.  Runs where
                           ``real_dist == 0`` are skipped (NaN-safe).

    Returns:
        Scalar error value — metres when ``use_percent_error=False``, percent
        when ``use_percent_error=True``.
    """
    if use_percent_error:
        # Replace zeros with NaN so we don't divide by zero; nanmean skips them.
        real_safe = np.where(real_dists != 0, real_dists, np.nan)
        rel_errors = (real_dists - sim_dists) / real_safe
        return float(np.sqrt(np.nanmean(rel_errors ** 2)) * 100.0)
    return float(np.sqrt(np.mean((real_dists - sim_dists) ** 2)))


def grid_search(
    runs: list[RunMetrics],
    max_speed_range: np.ndarray,
    max_accel_range: np.ndarray,
    use_percent_error: bool = False,
    cache_path: str | Path | None = None,
) -> tuple[float, float, np.ndarray]:
    """Grid search over ``(max_speed_ms, max_accel_ms2)`` to minimise RMSE.

    For every cell ``(i, j)`` in the grid:

    1. Simulate each run in *runs* using ``max_accel_range[i]`` and
       ``max_speed_range[j]`` as the controller parameters.
    2. Compute the distance error between real and simulated final distances
       across all speed commands (absolute RMSE or percent RMSPE depending on
       ``use_percent_error``).
    3. Store the error in ``rmse_grid[i, j]``.

    Progress is printed every 10 % of the total combinations.

    Args:
        runs:              Real-robot data; ``STEADY_S`` is used as the
                           ``duration_s`` argument to every simulation call.
        max_speed_range:   1-D array of candidate ``max_speed_ms`` values.
        max_accel_range:   1-D array of candidate ``max_accel_ms2`` values.
        use_percent_error: If ``False`` (default) minimise absolute distance
                           RMSE (m).  If ``True`` minimise root-mean-square
                           *percentage* error (RMSPE, %) so that all speed
                           commands contribute equally regardless of magnitude.
        cache_path:        Optional path to a ``.npz`` cache file.  If the file
                           exists the search is skipped and the cached result is
                           returned immediately.  On a cache miss the result is
                           saved to this path after the search completes.
                           Pass ``None`` (default) to disable caching entirely.

    Returns:
        ``(best_max_speed_ms, best_max_accel_ms2, rmse_grid)`` where
        ``rmse_grid`` has shape ``(len(max_accel_range), len(max_speed_range))``
        and values are in metres or percent depending on ``use_percent_error``.
    """
    # ── Cache hit ─────────────────────────────────────────────────────────────
    if cache_path is not None and Path(cache_path).exists():
        best_speed, best_accel, rmse_grid, _, _ = load_grid_results(cache_path)
        return best_speed, best_accel, rmse_grid

    # ── Cache miss — run the full search ──────────────────────────────────────
    n_speed = len(max_speed_range)
    n_accel = len(max_accel_range)
    rmse_grid = np.zeros((n_accel, n_speed), dtype=float)
    real_dists = np.array([r.real_final_dist for r in runs])

    unit = "%" if use_percent_error else "m"
    metric_name = "RMSPE" if use_percent_error else "RMSE"

    total = n_speed * n_accel
    done = 0
    report_every = max(1, total // 10)

    print(f"Grid search ({metric_name}): {n_speed} × {n_accel} = {total} combinations …")

    for i, max_accel in enumerate(max_accel_range):
        for j, max_speed in enumerate(max_speed_range):

            sim_dists = np.array([
                simulate_run(
                    speed_255=r.speed_cmd,
                    duration_s=STEADY_S,
                    max_speed_ms=max_speed,
                    max_accel_ms2=max_accel,
                )["final_distance"]
                for r in runs
            ])

            rmse_grid[i, j] = _compute_rmse(real_dists, sim_dists, use_percent_error)

            done += 1
            if done % report_every == 0:
                pct = 100 * done / total
                print(f"  {done}/{total}  ({pct:.0f} %)")

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


# ── Grid result cache (full rmse_grid + ranges) ───────────────────────────────

def save_grid_results(
    path: str | Path,
    best_speed: float,
    best_accel: float,
    rmse_grid: np.ndarray,
    max_speed_range: np.ndarray,
    max_accel_range: np.ndarray,
    use_percent_error: bool = False,
) -> None:
    """Save the full grid search result (arrays + scalars) to a ``.npz`` file.

    Use :func:`load_grid_results` to restore. The file is a standard NumPy
    archive — delete it to force a fresh grid search on the next run.

    Args:
        path:              Output path, e.g. ``"grid_cache.npz"``.
        best_speed:        Best ``max_speed_ms`` found by the search.
        best_accel:        Best ``max_accel_ms2`` found by the search.
        rmse_grid:         2-D RMSE array, shape ``(n_accel, n_speed)``.
        max_speed_range:   Speed axis used in the search.
        max_accel_range:   Acceleration axis used in the search.
        use_percent_error: Whether RMSPE (``True``) or RMSE (``False``) was used.
    """
    np.savez(
        str(path),
        rmse_grid=rmse_grid,
        max_speed_range=max_speed_range,
        max_accel_range=max_accel_range,
        best_speed=np.float64(best_speed),
        best_accel=np.float64(best_accel),
        use_percent_error=np.bool_(use_percent_error),
    )
    print(f"Grid results cached → {path}")


def load_grid_results(
    path: str | Path,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Load a previously cached grid search result from a ``.npz`` file.

    Prints a summary so you can verify the cached ranges match the current
    config before use.

    Args:
        path: Path written by :func:`save_grid_results`.

    Returns:
        ``(best_speed, best_accel, rmse_grid, max_speed_range, max_accel_range)``
    """
    data = np.load(str(path))
    best_speed       = float(data["best_speed"])
    best_accel       = float(data["best_accel"])
    rmse_grid        = data["rmse_grid"]
    max_speed_range  = data["max_speed_range"]
    max_accel_range  = data["max_accel_range"]

    n_s, n_a = len(max_speed_range), len(max_accel_range)
    print(f"⚡ Cache loaded from '{path}'")
    print(f"   speed : {max_speed_range[0]:.3f} → {max_speed_range[-1]:.3f}  ({n_s} steps)")
    print(f"   accel : {max_accel_range[0]:.3f} → {max_accel_range[-1]:.3f}  ({n_a} steps)")
    print(f"   best  : max_speed={best_speed:.4f} m/s  max_accel={best_accel:.4f} m/s²")
    print("   ⚠  Delete the file to force a fresh search.")
    return best_speed, best_accel, rmse_grid, max_speed_range, max_accel_range


# ── Result persistence ────────────────────────────────────────────────────────

def save_best_params(
    path: str | Path,
    best_speed: float,
    best_accel: float,
    best_rmse: float,
    grid_config: dict,
    use_percent_error: bool = False,
) -> None:
    """Write best parameters to a JSON file.

    The saved file can be loaded directly to configure a ``SpheroController``
    in the simulation environment.

    Args:
        path:              Output file path (e.g. ``"best_params.json"``).
        best_speed:        Best ``max_speed_ms`` found by the grid search (m/s).
        best_accel:        Best ``max_accel_ms2`` found by the grid search (m/s²).
        best_rmse:         Distance error at the best parameter point — metres
                           when ``use_percent_error=False``, percent otherwise.
        grid_config:       Dict describing the search bounds and step counts;
                           stored for reproducibility (see notebook config cell).
        use_percent_error: Records which error metric was used in the JSON.

    Example output::

        {
          "max_speed_ms": 2.1234,
          "max_accel_ms2": 1.5678,
          "distance_rmse_m": 0.0312,
          "error_metric": "RMSE_m",
          "grid_search": {
            "speed_range": [1.5, 4.0, 25],
            "accel_range": [0.5, 4.0, 25]
          }
        }
    """
    metric_key = "distance_rmspe_pct" if use_percent_error else "distance_rmse_m"
    metric_label = "RMSPE_pct" if use_percent_error else "RMSE_m"
    params = {
        "max_speed_ms": round(best_speed, 4),
        "max_accel_ms2": round(best_accel, 4),
        metric_key: round(best_rmse, 4),
        "error_metric": metric_label,
        "grid_search": grid_config,
    }
    Path(path).write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"Best params saved → {path}")


# ── Speed → Distance factor plot ──────────────────────────────────────────────

def fit_speed_distance_factor(runs: list[RunMetrics]) -> float:
    """Fit a through-origin linear model ``distance = factor × speed_cmd``.

    Uses the ordinary-least-squares solution for a no-intercept model:
    ``factor = Σ(s·d) / Σ(s²)``

    Args:
        runs: List of :class:`RunMetrics` (one entry per speed command).

    Returns:
        ``factor`` — metres per speed-unit (speed on 0-255 scale).
    """
    speeds = np.array([r.speed_cmd      for r in runs], dtype=float)
    dists  = np.array([r.real_final_dist for r in runs], dtype=float)
    return float(np.dot(speeds, dists) / np.dot(speeds, speeds))


def plot_speed_distance_factor(
    runs: list[RunMetrics],
    out_path: str | Path | None = None,
) -> float:
    """Plot ``distance = factor × speed_cmd`` fitted through the origin.

    Scatter-plots the real robot data (speed command vs final travel distance),
    overlays the best-fit line ``d = factor × s`` that passes through (0, 0),
    and annotates the factor value.

    Args:
        runs:     List of :class:`RunMetrics` — one entry per speed command.
        out_path: If given, save the figure to this path instead of showing it.

    Returns:
        ``factor`` — the fitted scale factor (m / speed-unit, 0-255 scale).

    Example::

        factor = plot_speed_distance_factor(runs_metrics)
        # factor ≈ 0.0082  →  speed 100 ≈ 0.82 m
    """
    speeds = np.array([r.speed_cmd       for r in runs], dtype=float)
    dists  = np.array([r.real_final_dist  for r in runs], dtype=float)

    factor = fit_speed_distance_factor(runs)

    s_line = np.linspace(0, speeds.max() * 1.05, 300)
    d_line = factor * s_line

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(speeds, dists, s=60, zorder=3, label="Measured data")
    ax.plot(
        s_line, d_line,
        color="tab:red", lw=2,
        label=f"Fit: d = {factor:.5f} × speed  (R²={_r2(speeds, dists, factor):.4f})",
    )
    ax.plot(0, 0, "ko", ms=6, zorder=4, label="Origin (0, 0)")

    # Annotate each data point with its speed command
    for s, d in zip(speeds, dists):
        ax.annotate(
            f"{int(s)}",
            (s, d),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            alpha=0.7,
        )

    ax.set_xlabel("Speed command (0-255)")
    ax.set_ylabel("Final travel distance (m)")
    ax.set_title("Speed-to-Distance linear factor  (through-origin fit)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"Figure saved → {out_path}")
    else:
        plt.show()

    print(f"\nFitted factor : {factor:.6f}  m / speed-unit")
    print(f"  speed 50   → {factor * 50:.4f} m")
    print(f"  speed 100  → {factor * 100:.4f} m")
    print(f"  speed 200  → {factor * 200:.4f} m")
    return factor


def _r2(speeds: np.ndarray, dists: np.ndarray, factor: float) -> float:
    """R² for the no-intercept model d = factor × speed."""
    ss_res = np.sum((dists - factor * speeds) ** 2)
    ss_tot = np.sum((dists - dists.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

