"""speed_plot.py — Sphero BOLT+ speed-analysis visualisation.

All plot functions consume the ``dict[int, DataFrame]`` produced by
``speed_data.load_all_runs``.  Sim-comparison plots additionally need the
``best_speed`` / ``best_accel`` values exported by ``calib_sim2real.ipynb``.

Public API
----------
Single-run plots
~~~~~~~~~~~~~~~~
``plot_single_speed(speed_cmd, runs, mode, show_components)``
    3-panel: distance / velocity / acceleration for one run.
    - ``mode="raw"``     : raw sensor ``speed_ms`` + ``accel_y_ms2``
    - ``mode="derived"`` : speed/accel derived from distance steps

All-runs plots
~~~~~~~~~~~~~~
``plot_all_speeds(runs, mode, reduction_mode)``
    3-panel overlay of every speed command.

Sim-comparison plots
~~~~~~~~~~~~~~~~~~~~
``plot_single_speed_with_sim(speed_cmd, runs, max_speed_ms, max_accel_ms2)``
    3-panel measured vs simulated for one run.

``plot_all_distance_with_sim_subplots(runs, max_speed_ms, max_accel_ms2, cols)``
    Grid of distance sub-plots — measured vs simulated for every run.

Delay plot
~~~~~~~~~~
``plot_delay_vs_speed(runs)``
    Command-to-motion delay vs speed command.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .speed_data import (
    IDLE_S, STEADY_S, TOTAL_S,
    PHASE_WINDOWS, PHASE_COLORS,
    estimate_motion_delay, _reduce_points,
)
from .calib_sim import simulate_run as _default_simulate_run, SIM_DT


# ── Private: plot utilities ───────────────────────────────────────────────────

def _safe_percentile(series: pd.Series, q: float, default: float = np.nan) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    return float(np.percentile(vals, q)) if vals.size else default


def _shade_phases(ax, x_max: float = TOTAL_S) -> None:
    for phase, (start, end) in PHASE_WINDOWS.items():
        ax.axvspan(start, end, color=PHASE_COLORS[phase], alpha=0.18, lw=0, zorder=0)
        ax.axvline(start, color=PHASE_COLORS[phase], lw=1, ls="--", alpha=0.45, zorder=1)
    ax.axvline(x_max, color="#666666", lw=1, ls=":", alpha=0.5, zorder=1)
    ax.set_xlim(0, x_max)


def _add_phase_labels(ax) -> None:
    for phase, (start, end) in PHASE_WINDOWS.items():
        ax.text(
            (start + end) / 2.0, 0.96, phase,
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, alpha=0.8,
        )


# ── Private: simulation bridge ────────────────────────────────────────────────

def _simulate_profile(
    speed_cmd: int,
    t_arr: np.ndarray,
    max_speed_ms: float,
    max_accel_ms2: float,
    simulate_fn=None,
) -> pd.DataFrame:
    """Run the sim and return a DataFrame aligned to the real time axis."""
    if simulate_fn is None:
        simulate_fn = _default_simulate_run
    sim_total = max(float(t_arr[-1]) - IDLE_S, TOTAL_S - IDLE_S)
    result = simulate_fn(
        speed_255=float(speed_cmd),
        duration_s=STEADY_S,
        max_speed_ms=max_speed_ms,
        max_accel_ms2=max_accel_ms2,
        total_time_s=sim_total,
    )

    # Map real time → sim time:  sim_t = real_t − IDLE_S
    # Negative values (idle window) are clamped to 0 so interp returns
    # the sim's initial state (v=0, d=0) — then we zero them out explicitly.
    t_lookup = np.clip(t_arr - IDLE_S, 0.0, result["time"][-1])
    sim_speed = np.interp(t_lookup, result["time"], result["speed"])
    sim_dist  = np.interp(t_lookup, result["time"], result["distance"])

    # Force idle window to zero (robot hasn't received the command yet)
    idle_mask = t_arr < IDLE_S
    sim_speed[idle_mask] = 0.0
    sim_dist[idle_mask]  = 0.0

    return pd.DataFrame({
        "t_s":                   t_arr,
        "sim_speed_ms":          sim_speed,
        "sim_travel_distance_m": sim_dist,
        "sim_accel_y_ms2":       np.gradient(sim_speed, t_arr),
    })


# ── Public: single-run plot ───────────────────────────────────────────────────

def plot_single_speed(
    speed_cmd: int,
    runs: dict[int, pd.DataFrame],
    mode: str = "derived",
    show_components: bool = True,
) -> None:
    """3-panel plot (distance / velocity / acceleration) for one run.

    Args:
        speed_cmd:       Key into *runs* (e.g. 50 for ``exp_speed_050.csv``).
        runs:            Output of :func:`speed_data.load_all_runs`.
        mode:            ``"raw"`` — sensor ``speed_ms`` + ``accel_y_ms2``.
                         ``"derived"`` — computed from distance steps.
        show_components: Overlay ``vx_ms`` / ``vy_ms`` on the velocity panel.
    """
    if speed_cmd not in runs:
        raise KeyError(
            f"Speed {speed_cmd} not found.  Available: "
            + ", ".join(f"{s:03d}" for s in sorted(runs))
        )
    df    = runs[speed_cmd]
    x_max = max(TOTAL_S, float(df["t_s"].max()))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    fig.suptitle(
        f"Robot experiment — speed {speed_cmd:03d}  ({mode.capitalize()} mode)",
        fontsize=16,
    )

    # ── Distance ─────────────────────────────────────────────────────────────
    ax1.plot(df["t_s"], df["travel_distance_m"],
             color="tab:blue", lw=2, label="Distance")
    ax1.set_ylabel("Distance (m)")
    ax1.set_title("Travel distance")

    # ── Velocity ─────────────────────────────────────────────────────────────
    speed_col   = "derived_speed_ms" if mode == "derived" else "speed_ms"
    speed_label = "Speed (derived from distance)" if mode == "derived" else "Speed magnitude (raw)"
    ax2.plot(df["t_s"], df[speed_col], lw=2.2, color="tab:green", label=speed_label)

    if show_components:
        ax2.plot(df["t_s"], df["vx_ms"], lw=1.4, alpha=0.6, label="Vx")
        ax2.plot(df["t_s"], df["vy_ms"], lw=1.4, alpha=0.6, label="Vy")

    delay = estimate_motion_delay(df)
    if np.isfinite(delay["command_time_s"]):
        ax2.axvline(delay["command_time_s"], color="black", ls=":", lw=1.2, label="Cmd on")
    if np.isfinite(delay["motion_onset_time_s"]):
        ax2.axvline(delay["motion_onset_time_s"], color="purple", ls=":", lw=1.2,
                    label="Motion onset")
    if np.isfinite(delay["motion_delay_s"]):
        ax2.text(0.01, 0.92, f"delay = {delay['motion_delay_s']:.3f} s",
                 transform=ax2.transAxes, ha="left", va="top", fontsize=10,
                 bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"})

    ax2.set_ylabel("Speed (m/s)")
    ax2.set_title("Velocity")
    ax2.legend(loc="upper right")

    # ── Acceleration ──────────────────────────────────────────────────────────
    if mode == "derived":
        ax3.plot(df["t_s"], df["derived_accel_ms2"],
                 color="tab:red", lw=2, label="Accel (derived)")
        if show_components and "accel_y_ms2" in df.columns:
            ax3.plot(df["t_s"], df["accel_y_ms2"], alpha=0.3, label="Raw IMU Y (ref)")
        ax3.set_title("Acceleration from distance")
    else:
        ax3.plot(df["t_s"], df["accel_y_ms2"],
                 color="tab:red", lw=2, label="IMU Accel Y")
        ax3.set_title("IMU acceleration in Y")

    ax3.set_ylabel("Acceleration (m/s²)")
    ax3.set_xlabel("Time (s)")
    ax3.legend(loc="upper right")

    for ax in (ax1, ax2, ax3):
        _shade_phases(ax, x_max)
        _add_phase_labels(ax)
        ax.grid(True, linestyle="--", alpha=0.45)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()


# ── Public: all-runs plot ─────────────────────────────────────────────────────

def plot_all_speeds(
    runs: dict[int, pd.DataFrame],
    mode: str = "derived",
    reduction_mode: str = "none",
) -> None:
    """3-panel overlay of every speed command.

    Args:
        runs:            Output of :func:`speed_data.load_all_runs`.
        mode:            ``"raw"`` or ``"derived"``.
        reduction_mode:  Optional point reduction for cleaner lines.
    """
    speeds = sorted(runs)
    cmap   = plt.get_cmap("viridis", len(speeds))
    x_max  = max(TOTAL_S, max(float(runs[s]["t_s"].max()) for s in speeds))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    fig.suptitle(
        f"Robot experiment — all speeds  ({mode.capitalize()} mode)", fontsize=16
    )

    if mode == "derived":
        v_col, v_title = "derived_speed_ms",  "Velocity (derived from distance)"
        a_col, a_title = "derived_accel_ms2", "Acceleration from distance"
    else:
        v_col, v_title = "speed_ms",    "Velocity magnitude (raw sensor)"
        a_col, a_title = "accel_y_ms2", "IMU acceleration in Y"

    for idx, speed in enumerate(speeds):
        df = runs[speed]
        if reduction_mode != "none":
            df = _reduce_points(df, mode=reduction_mode)
        color, label = cmap(idx), f"{speed:03d}"
        ax1.plot(df["t_s"], df["travel_distance_m"], color=color, lw=1.6, alpha=0.7, label=label)
        ax2.plot(df["t_s"], df[v_col],               color=color, lw=1.6, alpha=0.8, label=label)
        ax3.plot(df["t_s"], df[a_col],               color=color, lw=1.6, alpha=0.8, label=label)

    ax1.set_ylabel("Distance (m)");        ax1.set_title("Travel distance")
    ax2.set_ylabel("Speed (m/s)");         ax2.set_title(v_title)
    ax3.set_ylabel("Acceleration (m/s²)"); ax3.set_xlabel("Time (s)"); ax3.set_title(a_title)

    for ax in (ax1, ax2, ax3):
        _shade_phases(ax, x_max)
        _add_phase_labels(ax)
        ax.grid(True, linestyle="--", alpha=0.3)

    ax2.legend(title="Speed cmd", ncol=2, fontsize=9, title_fontsize=10,
               loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout(rect=(0, 0.03, 0.88, 0.95))
    plt.show()


# ── Public: sim comparison plots ─────────────────────────────────────────────

def plot_single_speed_with_sim(
    speed_cmd: int,
    runs: dict[int, pd.DataFrame],
    max_speed_ms: float,
    max_accel_ms2: float,
    simulate_fn=None,
) -> None:
    """3-panel measured vs simulated comparison for one speed command.

    Args:
        speed_cmd:     Speed command to compare (must be a key in *runs*).
        runs:          Output of :func:`speed_data.load_all_runs`.
        max_speed_ms:  ``SpheroController.max_speed_ms`` — use ``best_speed``
                       from ``calib_sim2real.ipynb`` / ``best_params.json``.
        max_accel_ms2: ``SpheroController.max_accel_ms2`` — use ``best_accel``
                       from ``calib_sim2real.ipynb`` / ``best_params.json``.
    """
    if speed_cmd not in runs:
        raise KeyError(
            f"Speed {speed_cmd} not found.  Available: "
            + ", ".join(f"{s:03d}" for s in sorted(runs))
        )
    real_df = runs[speed_cmd]
    sim_df  = _simulate_profile(
        speed_cmd, real_df["t_s"].to_numpy(), max_speed_ms, max_accel_ms2,
        simulate_fn=simulate_fn,
    )
    x_max = max(TOTAL_S, float(real_df["t_s"].max()))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    fig.suptitle(
        f"Measured vs Simulated — speed {speed_cmd:03d}\n"
        f"(max_speed = {max_speed_ms:.3f} m/s,  max_accel = {max_accel_ms2:.3f} m/s²)",
        fontsize=14,
    )

    # ── Distance ─────────────────────────────────────────────────────────────
    ax1.plot(real_df["t_s"], real_df["travel_distance_m"],
             color="tab:blue", lw=2, label="Measured")
    ax1.plot(sim_df["t_s"],  sim_df["sim_travel_distance_m"],
             color="black", lw=1.6, ls="--", label="Simulated")
    ax1.set_ylabel("Distance (m)"); ax1.set_title("Travel distance")
    ax1.legend(loc="upper left")

    # ── Velocity ─────────────────────────────────────────────────────────────
    ax2.plot(real_df["t_s"], real_df["derived_speed_ms"],
             color="tab:green", lw=2, label="Measured")
    ax2.plot(sim_df["t_s"],  sim_df["sim_speed_ms"],
             color="black", lw=1.6, ls="--", label="Simulated")
    delay = estimate_motion_delay(real_df)
    if np.isfinite(delay["command_time_s"]):
        ax2.axvline(delay["command_time_s"], color="black", ls=":", lw=1.2, alpha=0.8)
    if np.isfinite(delay["motion_onset_time_s"]):
        ax2.axvline(delay["motion_onset_time_s"], color="purple", ls=":", lw=1.2, alpha=0.8)
    if np.isfinite(delay["motion_delay_s"]):
        ax2.text(0.01, 0.92, f"delay = {delay['motion_delay_s']:.3f} s",
                 transform=ax2.transAxes, ha="left", va="top", fontsize=10,
                 bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"})
    ax2.set_ylabel("Speed (m/s)"); ax2.set_title("Speed")
    ax2.legend(loc="upper right")

    # ── Acceleration ──────────────────────────────────────────────────────────
    ax3.plot(real_df["t_s"], real_df["derived_accel_ms2"],
             color="tab:red", lw=2, alpha=0.8, label="Measured")
    ax3.plot(sim_df["t_s"],  sim_df["sim_accel_y_ms2"],
             color="black", lw=1.6, ls="--", label="Simulated dv/dt")
    ax3.set_ylabel("Acceleration (m/s²)"); ax3.set_xlabel("Time (s)")
    ax3.set_title("Acceleration"); ax3.legend(loc="upper right")

    for ax in (ax1, ax2, ax3):
        _shade_phases(ax, x_max)
        _add_phase_labels(ax)
        ax.grid(True, linestyle="--", alpha=0.45)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()


def plot_all_distance_with_sim_subplots(
    runs: dict[int, pd.DataFrame],
    max_speed_ms: float,
    max_accel_ms2: float,
    cols: int = 3,
    share_y: bool = False,
    simulate_fn=None,
) -> None:
    """Grid of distance sub-plots — measured vs simulated for every run.

    Args:
        runs:          Output of :func:`speed_data.load_all_runs`.
        max_speed_ms:  Best ``max_speed_ms`` from ``best_params.json``.
        max_accel_ms2: Best ``max_accel_ms2`` from ``best_params.json``.
        cols:          Number of subplot columns (default 3).
        share_y:        Whether to share the y-axis across subplots (default False).
    """
    speeds = sorted(runs)
    n    = len(speeds)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows, cols, figsize=(5.0 * cols, 3.8 * rows), sharex=True, sharey=share_y
    )
    axes  = np.atleast_1d(axes).flatten()
    x_max = max(TOTAL_S, max(float(runs[s]["t_s"].max()) for s in speeds))
    y_max = 0.0

    for idx, speed in enumerate(speeds):
        ax      = axes[idx]
        real_df = runs[speed]
        sim_df  = _simulate_profile(
            speed, real_df["t_s"].to_numpy(), max_speed_ms, max_accel_ms2,
            simulate_fn=simulate_fn,
        )
        y_max = max(
            y_max,
            float(real_df["travel_distance_m"].max()),
            float(sim_df["sim_travel_distance_m"].max()),
        )
        ax.plot(real_df["t_s"], real_df["travel_distance_m"],
                color="tab:blue", lw=1.8, label="Measured")
        ax.plot(sim_df["t_s"],  sim_df["sim_travel_distance_m"],
                color="black", lw=1.5, ls="--", label="Simulated")
        ax.set_title(f"Speed {speed:03d}")
        _shade_phases(ax, x_max=x_max)
        ax.grid(True, linestyle="--", alpha=0.35)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    for idx, ax in enumerate(axes[:n]):
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max * 1.05 if y_max > 0 else 1.0)
        if idx % cols == 0:
            ax.set_ylabel("Distance (m)")
        if idx >= (rows - 1) * cols:
            ax.set_xlabel("Time (s)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(
        f"Measured vs Simulated distance — all speeds\n"
        f"(max_speed = {max_speed_ms:.3f} m/s,  max_accel = {max_accel_ms2:.3f} m/s²)",
        fontsize=14,
    )
    plt.tight_layout(rect=(0, 0.02, 1, 0.93))
    plt.show()


# ── Public: delay plot ────────────────────────────────────────────────────────

def plot_delay_vs_speed(runs: dict[int, pd.DataFrame]) -> None:
    """Command-to-motion delay vs speed command.

    Computes :func:`speed_data.estimate_motion_delay` for each run and plots
    the results — no pre-built summary DataFrame required.

    Args:
        runs: Output of :func:`speed_data.load_all_runs`.
    """
    speeds = sorted(runs)
    delays = [estimate_motion_delay(runs[s])["motion_delay_s"] for s in speeds]

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    ax.plot(speeds, delays, marker="o", lw=1.8, color="tab:purple")
    for s, d in zip(speeds, delays):
        if np.isfinite(d):
            ax.annotate(f"{d:.3f}", (s, d), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8)
    ax.set_title("Command-to-motion delay by speed command")
    ax.set_xlabel("Speed command (0-255)")
    ax.set_ylabel("Delay (s)")
    ax.grid(True, linestyle="--", alpha=0.45)
    plt.tight_layout()
    plt.show()


