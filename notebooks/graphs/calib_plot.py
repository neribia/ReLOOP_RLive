"""calib_plot.py — All visualisation for Sim2Real calibration results.

Figure layout — 2 × 3 grid (6 panels)
---------------------------------------
① 3D RMSE surface      (matplotlib Axes3D)   row 0, col 0
② 2D RMSE heatmap      (imshow)              row 0, col 1
③ Real vs Sim final distance                  row 0, col 2
④ Gap % per speed command                     row 1, col 0
⑤ Peak speed: real vs sim + linear ref        row 1, col 1
⑥ Accel / decel: real vs best-sim             row 1, col 2

Public API
----------
- ``plot_rmse_surface``           — fills one ``Axes3D`` subplot.
- ``plot_rmse_heatmap``           — fills one regular ``Axes`` subplot.
- ``plot_panel``                  — standalone figure for ONE named/numbered panel.
- ``plot_results``                — builds and optionally saves the full 6-panel figure.
- ``print_metrics_table``         — prints real-robot metrics to stdout.
- ``print_metrics_table_with_sim``— prints real + sim comparison (needs best params).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3-D projection

from calib_data import RunMetrics
from calib_sim import simulate_run as _default_simulate_run
from speed_data import STEADY_S


# ── Panel registry ────────────────────────────────────────────────────────────

_PANEL_IDS: dict[int, str] = {
    1: "rmse_surface",
    2: "rmse_heatmap",
    3: "distance",
    4: "gap",
    5: "peak_speed",
    6: "accel_decel",
}
_PANEL_NUMS: dict[str, int] = {v: k for k, v in _PANEL_IDS.items()}
_ALL_PANELS: list[int] = list(_PANEL_IDS.keys())  # [1, 2, 3, 4, 5, 6]


def _resolve_panel(p: str | int) -> int:
    """Convert a panel name or number to a canonical integer key."""
    if isinstance(p, int):
        if p not in _PANEL_IDS:
            raise ValueError(f"Unknown panel number {p}. Valid: {_ALL_PANELS}")
        return p
    if isinstance(p, str):
        if p not in _PANEL_NUMS:
            raise ValueError(f"Unknown panel name '{p}'. Valid: {list(_PANEL_NUMS)}")
        return _PANEL_NUMS[p]
    raise TypeError(f"panel must be int or str, got {type(p)}")


# ── Individual panel helpers (ax-filling) ─────────────────────────────────────

def plot_rmse_surface(
    ax: Axes3D,
    rmse_grid: np.ndarray,
    max_speed_range: np.ndarray,
    max_accel_range: np.ndarray,
    best_speed: float,
    best_accel: float,
    rmse_unit: str = "m",
) -> None:
    """Render a 3-D RMSE surface over the ``(max_speed_ms, max_accel_ms2)`` grid."""
    metric_label = f"RMSPE ({rmse_unit})" if rmse_unit == "%" else f"RMSE ({rmse_unit})"
    X, Y = np.meshgrid(max_speed_range, max_accel_range)
    surf = ax.plot_surface(X, Y, rmse_grid, cmap="RdYlGn_r", alpha=0.85, linewidth=0)
    plt.colorbar(surf, ax=ax, shrink=0.45, pad=0.1, label=metric_label)

    best_rmse = float(np.min(rmse_grid))
    ax.scatter(
        [best_speed], [best_accel], [best_rmse],
        color="cyan", s=80, zorder=10,
        label=f"Best ({best_speed:.2f} m/s,\n{best_accel:.2f} m/s²)",
    )
    ax.set_xlabel("max_speed_ms (m/s)", labelpad=6)
    ax.set_ylabel("max_accel_ms2 (m/s²)", labelpad=6)
    ax.set_zlabel(metric_label, labelpad=4)
    ax.set_title("① RMSE Surface")
    ax.legend(fontsize=7, loc="upper right")


def plot_rmse_heatmap(
    ax,
    rmse_grid: np.ndarray,
    max_speed_range: np.ndarray,
    max_accel_range: np.ndarray,
    best_speed: float,
    best_accel: float,
    rmse_unit: str = "m",
) -> None:
    """Render a 2-D RMSE heatmap with the best-parameter point highlighted."""
    metric_label = f"RMSPE ({rmse_unit})" if rmse_unit == "%" else f"RMSE ({rmse_unit})"
    im = ax.imshow(
        rmse_grid,
        origin="lower",
        aspect="auto",
        cmap="RdYlGn_r",
        extent=[
            max_speed_range[0], max_speed_range[-1],
            max_accel_range[0], max_accel_range[-1],
        ],
    )
    ax.scatter(
        [best_speed], [best_accel],
        color="cyan", s=100, zorder=5,
        label=f"Best ({best_speed:.2f}, {best_accel:.2f})",
    )
    ax.set_xlabel("max_speed_ms (m/s)")
    ax.set_ylabel("max_accel_ms2 (m/s²)")
    ax.set_title("② RMSE Heatmap")
    ax.legend(fontsize=7)
    plt.colorbar(im, ax=ax, label=metric_label)


def _fill_distance_comparison(ax, speed_cmds, real_dists, sim_dists) -> None:
    ax.plot(speed_cmds, real_dists, "o-", label="Real", color="steelblue")
    ax.plot(speed_cmds, sim_dists, "s--", label="Sim (best)", color="tomato")
    ax.set_xlabel("Speed cmd (0–255)")
    ax.set_ylabel("Final distance (m)")
    ax.set_title("③ Final Distance: Real vs Sim")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _fill_distance_gap(ax, speed_cmds, real_dists, sim_dists) -> None:
    gaps = [
        (r - s) / r * 100 if r != 0 else 0.0
        for r, s in zip(real_dists, sim_dists)
    ]
    bar_colors = ["tomato" if g > 0 else "steelblue" for g in gaps]
    ax.bar(speed_cmds, gaps, color=bar_colors, width=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Speed cmd")
    ax.set_ylabel("Gap %  (real − sim) / real")
    ax.set_title("④ Sim2Real Distance Gap")
    ax.grid(True, alpha=0.3)


def _fill_metrics_table(ax, runs: list[RunMetrics], print_table: bool = True) -> None:
    if print_table:
        print_metrics_table(runs)
    ax.axis("off")
    headers = ["Cmd", "Real (m)", "Sim (m)", "Gap %", "MaxSpd", "MaxAcc", "MaxDec"]
    rows = []
    for r in runs:
        sim_d = getattr(r, "sim_final_dist", None) or float("nan")
        gap = (
            (r.real_final_dist - sim_d) / r.real_final_dist * 100
            if r.real_final_dist != 0 and not np.isnan(sim_d) else 0.0
        )
        rows.append([
            str(r.speed_cmd),
            f"{r.real_final_dist:.3f}",
            f"{sim_d:.3f}" if not np.isnan(sim_d) else "—",
            f"{gap:+.1f}%" if not np.isnan(sim_d) else "—",
            f"{r.real_max_speed:.3f}",
            f"{r.real_max_accel:.3f}",
            f"{r.real_max_decel:.3f}",
        ])
    tbl = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.35)
    ax.set_title("⑤ Per-Run Metrics", pad=12)


def _fill_peak_speed(ax, speed_cmds, runs, best_speed, best_accel, simulate_fn=None) -> None:
    _sim = simulate_fn or _default_simulate_run
    real_peak  = [r.real_max_speed for r in runs]
    sim_peak   = [_sim(r.speed_cmd, STEADY_S, best_speed, best_accel)["max_speed"] for r in runs]
    linear_ref = [(c / 255.0) * best_speed for c in speed_cmds]
    ax.plot(speed_cmds, real_peak, "o-", label="Real", color="steelblue")
    ax.plot(speed_cmds, sim_peak, "s--", label="Sim", color="tomato")
    ax.plot(speed_cmds, linear_ref, "k:", alpha=0.5,
            label=f"Linear map ({best_speed:.2f} m/s @ 255)")
    ax.set_xlabel("Speed cmd (0–255)")
    ax.set_ylabel("Peak speed (m/s)")
    ax.set_title("⑤ Peak Speed: Real vs Sim")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def _fill_accel_decel(ax, speed_cmds, runs, best_accel) -> None:
    ax.plot(speed_cmds, [r.real_max_accel for r in runs],
            "o-", label="Real max accel", color="green")
    ax.plot(speed_cmds, [r.real_max_decel for r in runs],
            "s--", label="Real max decel", color="red")
    ax.axhline(best_accel,  color="green", linestyle=":", alpha=0.6,
               label=f"Sim accel = {best_accel:.2f}")
    ax.axhline(-best_accel, color="red",   linestyle=":", alpha=0.6,
               label=f"Sim decel = −{best_accel:.2f}")
    ax.set_xlabel("Speed cmd")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title("⑥ Accel / Decel: Real vs Sim")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


# ── Helpers shared internally ─────────────────────────────────────────────────

def _fill_sim_dists(
    runs: list[RunMetrics],
    best_speed: float,
    best_accel: float,
    simulate_fn=None,
) -> list[float]:
    """Simulate all runs with best params, cache ``sim_final_dist``, return list."""
    if simulate_fn is None:
        simulate_fn = _default_simulate_run
    sim_dists = []
    for r in runs:
        res = simulate_fn(r.speed_cmd, STEADY_S, best_speed, best_accel)
        r.sim_final_dist = res["final_distance"]
        sim_dists.append(r.sim_final_dist)
    return sim_dists


# ── Standalone single-panel figure ───────────────────────────────────────────

def plot_panel(
    panel: str | int,
    runs: list[RunMetrics],
    best_speed: float,
    best_accel: float,
    rmse_grid: np.ndarray | None = None,
    max_speed_range: np.ndarray | None = None,
    max_accel_range: np.ndarray | None = None,
    simulate_fn=None,
    rmse_unit: str = "m",
    out_path: str | None = None,
    figsize: tuple[float, float] = (7, 5),
) -> None:
    """Create a standalone figure for a single calibration panel.

    Use this to export individual panels for a thesis or report.

    Args:
        panel:           Panel name (e.g. ``"distance"``) or number (1–6).
        runs:            List of :class:`RunMetrics`.
        best_speed:      Best ``max_speed_ms`` from the grid search.
        best_accel:      Best ``max_accel_ms2`` from the grid search.
        rmse_grid:       Required for panels ``1`` / ``"rmse_surface"`` and
                         ``2`` / ``"rmse_heatmap"`` only.
        max_speed_range: Required for RMSE panels.
        max_accel_range: Required for RMSE panels.
        simulate_fn:     Optional SAPIEN-backed simulate function; defaults to
                         the pure-Python ``calib_sim.simulate_run``.
        rmse_unit:       ``"m"`` or ``"%"`` — used by RMSE panels.
        out_path:        If given, save the figure to this path (PNG).
        figsize:         Figure size ``(width, height)`` in inches.

    Example::

        plot_panel("distance", runs_metrics, best_speed, best_accel,
                   out_path="fig_distance.png")
        plot_panel(7, runs_metrics, best_speed, best_accel,
                   out_path="fig_accel_decel.png", figsize=(10, 5))
    """
    panel_num = _resolve_panel(panel)

    speed_cmds = [r.speed_cmd for r in runs]
    real_dists = [r.real_final_dist for r in runs]
    sim_dists  = _fill_sim_dists(runs, best_speed, best_accel, simulate_fn=simulate_fn)

    is_3d      = panel_num == 1
    needs_rmse = panel_num in (1, 2)

    if needs_rmse and (rmse_grid is None or max_speed_range is None or max_accel_range is None):
        raise ValueError(
            f"Panel '{_PANEL_IDS[panel_num]}' requires rmse_grid, "
            "max_speed_range, and max_accel_range."
        )

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": "3d"} if is_3d else {},
    )

    if   panel_num == 1:
        plot_rmse_surface(ax, rmse_grid, max_speed_range, max_accel_range,
                          best_speed, best_accel, rmse_unit=rmse_unit)
    elif panel_num == 2:
        plot_rmse_heatmap(ax, rmse_grid, max_speed_range, max_accel_range,
                          best_speed, best_accel, rmse_unit=rmse_unit)
    elif panel_num == 3:
        _fill_distance_comparison(ax, speed_cmds, real_dists, sim_dists)
    elif panel_num == 4:
        _fill_distance_gap(ax, speed_cmds, real_dists, sim_dists)
    elif panel_num == 5:
        _fill_metrics_table(ax, runs, print_table=True)
    elif panel_num == 6:
        _fill_peak_speed(ax, speed_cmds, runs, best_speed, best_accel, simulate_fn=simulate_fn)
    elif panel_num == 7:
        _fill_accel_decel(ax, speed_cmds, runs, best_accel)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Panel '{_PANEL_IDS[panel_num]}' saved → {out_path}")
    plt.show()


# ── Combined 7-panel figure ───────────────────────────────────────────────────

def plot_results(
    runs: list[RunMetrics],
    best_speed: float,
    best_accel: float,
    rmse_grid: np.ndarray,
    max_speed_range: np.ndarray,
    max_accel_range: np.ndarray,
    out_path: str | None = "calibration_results.png",
    rmse_unit: str = "m",
    simulate_fn=None,
    panels: list[str | int] | None = None,
) -> None:
    """Build and optionally save the full 7-panel calibration figure.

    Layout (GridSpec 2 × 3):

    ┌────┬────┬────┐
    │ ①  │ ②  │ ③  │
    ├────┼────┼────┤
    │ ④  │ ⑤  │ ⑥  │
    └────┴────┴────┘

    Args:
        runs:            Speed-sorted list of :class:`RunMetrics`.
        best_speed:      Best ``max_speed_ms`` from the grid search.
        best_accel:      Best ``max_accel_ms2`` from the grid search.
        rmse_grid:       RMSE array from the grid search.
        max_speed_range: Speed axis used in the grid search.
        max_accel_range: Acceleration axis used in the grid search.
        out_path:        Output PNG path. ``None`` = display only, don't save.
        rmse_unit:       ``"m"`` or ``"%"`` — controls RMSE panel labels.
        simulate_fn:     Optional SAPIEN simulate function (drop-in for
                         ``calib_sim.simulate_run``).
        panels:          Subset of panels to render, e.g. ``[1, 2, "distance"]``.
                         ``None`` (default) renders all 7 panels.
    """
    active = {_resolve_panel(p) for p in panels} if panels is not None else set(_ALL_PANELS)

    speed_cmds = [r.speed_cmd for r in runs]
    real_dists = [r.real_final_dist for r in runs]
    sim_dists  = _fill_sim_dists(runs, best_speed, best_accel, simulate_fn=simulate_fn)

    # Layout: row 0 = ①②③   row 1 = ④⑤⑥
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Sim2Real Calibration — Sphero BOLT+", fontsize=15, fontweight="bold")
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # ── ① 3D RMSE surface ─────────────────────────────────────────────────
    if 1 in active:
        ax1 = fig.add_subplot(gs[0, 0], projection="3d")
        plot_rmse_surface(ax1, rmse_grid, max_speed_range, max_accel_range,
                          best_speed, best_accel, rmse_unit=rmse_unit)

    # ── ② 2D RMSE heatmap ────────────────────────────────────────────────
    if 2 in active:
        ax2 = fig.add_subplot(gs[0, 1])
        plot_rmse_heatmap(ax2, rmse_grid, max_speed_range, max_accel_range,
                          best_speed, best_accel, rmse_unit=rmse_unit)

    # ── ③ Real vs Sim final distance ─────────────────────────────────────
    if 3 in active:
        ax3 = fig.add_subplot(gs[0, 2])
        _fill_distance_comparison(ax3, speed_cmds, real_dists, sim_dists)

    # ── ④ Gap % per speed ────────────────────────────────────────────────
    if 4 in active:
        ax4 = fig.add_subplot(gs[1, 0])
        _fill_distance_gap(ax4, speed_cmds, real_dists, sim_dists)

    # ── ⑤ Peak speed ─────────────────────────────────────────────────────
    if 5 in active:
        ax5 = fig.add_subplot(gs[1, 1])
        _fill_peak_speed(ax5, speed_cmds, runs, best_speed, best_accel,
                         simulate_fn=simulate_fn)

    # ── ⑥ Accel / decel ───────────────────────────────────────────────────
    if 6 in active:
        ax6 = fig.add_subplot(gs[1, 2])
        _fill_accel_decel(ax6, speed_cmds, runs, best_accel)

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {out_path}")
    plt.show()


# ── Console table ─────────────────────────────────────────────────────────────

def print_metrics_table(runs: list[RunMetrics]) -> None:
    """Print a formatted per-run metrics table to stdout (real robot only)."""
    header = (
        f"{'Cmd':>5} | {'Dist (m)':>8} | {'MaxSpd (m/s)':>12} | "
        f"{'MaxAcc':>8} | {'MaxDec':>8} | {'Dur (s)':>7}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}")
    print("REAL ROBOT METRICS PER RUN")
    print(sep)
    print(header)
    print("-" * len(header))
    for r in runs:
        print(
            f"{r.speed_cmd:>5} | {r.real_final_dist:>8.3f} | "
            f"{r.real_max_speed:>12.3f} | {r.real_max_accel:>8.3f} | "
            f"{r.real_max_decel:>8.3f} | {r.real_duration:>7.3f}"
        )
    print(sep)


def print_metrics_table_with_sim(
    runs: list[RunMetrics],
    best_speed: float,
    best_accel: float,
    simulate_fn=None,
) -> None:
    """Print a per-run metrics table comparing real robot vs simulation.

    Requires ``best_speed`` and ``best_accel`` from the grid search to simulate
    the expected final distance for each run.

    Args:
        runs:        List of :class:`RunMetrics`.
        best_speed:  Best ``max_speed_ms`` from ``grid_search`` (m/s).
        best_accel:  Best ``max_accel_ms2`` from ``grid_search`` (m/s²).
        simulate_fn: Optional SAPIEN simulate function; defaults to pure-Python
                     ``calib_sim.simulate_run``.

    Example::

        # After grid_search:
        print_metrics_table_with_sim(runs_metrics, best_speed, best_accel)
    """
    # Populate sim_final_dist on each run
    _fill_sim_dists(runs, best_speed, best_accel, simulate_fn=simulate_fn)

    header = (
        f"{'Cmd':>5} | {'Real (m)':>8} | {'Sim (m)':>8} | {'Gap %':>7} | "
        f"{'MaxSpd':>8} | {'MaxAcc':>8} | {'MaxDec':>8} | {'Dur (s)':>7}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(f"REAL vs SIM METRICS  (max_speed={best_speed:.4f} m/s, max_accel={best_accel:.4f} m/s²)")
    print(sep)
    print(header)
    print("-" * len(header))
    for r in runs:
        sim_d = r.sim_final_dist
        gap   = (r.real_final_dist - sim_d) / r.real_final_dist * 100 if r.real_final_dist != 0 else 0.0
        print(
            f"{r.speed_cmd:>5} | {r.real_final_dist:>8.3f} | {sim_d:>8.3f} | "
            f"{gap:>+7.1f}% | {r.real_max_speed:>8.3f} | {r.real_max_accel:>8.3f} | "
            f"{r.real_max_decel:>8.3f} | {r.real_duration:>7.3f}"
        )
    print(sep)


