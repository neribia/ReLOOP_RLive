"""speed_data.py — Sphero BOLT+ speed-analysis data loading & preprocessing.

No matplotlib dependency — safe to import anywhere.

Public API
----------
Constants
~~~~~~~~~
``IDLE_S``, ``STEADY_S``, ``RAMP_DOWN_S``, ``TOTAL_S``
``PHASE_WINDOWS``, ``PHASE_COLORS``, ``CSV_COLS``

Data loading
~~~~~~~~~~~~
``prepare_run_df(df, reduction_mode, window)``  → ``DataFrame``
    Process a single raw CSV DataFrame (strip, cast, phase, reduce, derive).

``load_all_runs(data_dir, reduction_mode, window)``  → ``dict[int, DataFrame]``
    Load every ``exp_speed_*.csv`` in a directory.

Analysis
~~~~~~~~
``estimate_motion_delay(df)``  → ``dict``
    Estimate command-on → motion-onset delay for one run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ── Experiment timing (mirrors bolt_log.py) ───────────────────────────────────

IDLE_S      = 0.5
STEADY_S    = 2.0
RAMP_DOWN_S = 5.0
TOTAL_S     = IDLE_S + STEADY_S + RAMP_DOWN_S

PHASE_WINDOWS: dict[str, tuple[float, float]] = {
    "idle":      (0.0,              IDLE_S),
    "steady":    (IDLE_S,           IDLE_S + STEADY_S),
    "ramp-down": (IDLE_S + STEADY_S, TOTAL_S),
}
PHASE_COLORS: dict[str, str] = {
    "idle":      "#d9d9d9",
    "steady":    "#a6cee3",
    "ramp-down": "#fdbf6f",
}
CSV_COLS = [
    "t_s", "cmd", "vx_ms", "vy_ms", "speed_ms",
    "travel_distance_m", "accel_x_ms2", "accel_y_ms2", "accel_z_ms2",
]


# ── Private helpers ───────────────────────────────────────────────────────────

def _phase_from_time(t: float) -> str:
    if t < IDLE_S:
        return "idle"
    if t < IDLE_S + STEADY_S:
        return "steady"
    return "ramp-down"


def _infer_speed_from_path(p: Path) -> int:
    return int(Path(p).stem.split("_")[-1])


def _resolve_data_dir() -> Path:
    """Walk up from cwd until a directory with exp_speed_*.csv is found."""
    for root in [Path.cwd(), *Path.cwd().parents]:
        for candidate in (root / "notebooks" / "data", root / "data"):
            if candidate.exists() and any(candidate.glob("exp_speed_*.csv")):
                return candidate
    raise FileNotFoundError(
        "Could not find a data directory containing exp_speed_*.csv files"
    )


def _reduce_points(df: pd.DataFrame, mode: str = "none", atol: float = 1e-6) -> pd.DataFrame:
    """Collapse repeated sensor blocks — keeps representative rows per block."""
    work = df.sort_values("t_s").copy()
    if mode == "none" or len(work) <= 2:
        return work

    # Exclude time and command columns from change-detection
    value_cols = [c for c in CSV_COLS if c in work.columns and c not in ("t_s", "cmd")]
    if not value_cols:
        return work

    vals = work[value_cols].to_numpy(dtype=float)
    v1, v2 = vals[:-1], vals[1:]
    changed = np.any(
        (np.abs(v1 - v2) > atol) | (np.isnan(v1) != np.isnan(v2)),
        axis=1,
    )
    block_ids = np.zeros(len(work), dtype=int)
    block_ids[1:] = np.cumsum(changed)
    groups = work.groupby(block_ids, sort=False)

    if   mode == "first_last": kept = [g.iloc[[0, -1]] if len(g) > 1 else g.iloc[[0]] for _, g in groups]
    elif mode == "center":     kept = [g.iloc[[len(g) // 2]] for _, g in groups]
    elif mode == "first":      kept = [g.iloc[[0]]  for _, g in groups]
    elif mode == "last":       kept = [g.iloc[[-1]] for _, g in groups]
    else: raise ValueError(f"Unknown reduction mode: {mode!r}")

    return pd.concat(kept).sort_values("t_s")


def _differentiate_and_smooth(
    t: np.ndarray, y: np.ndarray, window: int = 1, fill_staircase: bool = False
) -> np.ndarray:
    """Compute dy/dt with optional staircase-fill and rolling-mean smoothing.

    ``fill_staircase=True`` forward-fills velocity across flat distance
    intervals where the sensor hasn't updated yet (prevents false zero-spikes).
    """
    if len(t) < 2:
        return np.zeros_like(y)

    dt = np.diff(t)
    dy = np.diff(y)
    valid = (dt != 0) & np.isfinite(dt)

    raw = np.zeros_like(y)
    raw[1:] = np.divide(dy, dt, out=np.zeros_like(dy), where=valid)

    if fill_staircase:
        s = pd.Series(raw)
        has_moved = np.abs(dy) > 1e-9
        # Keep index 0, mask everything else that didn't move
        mask = np.concatenate([[True], has_moved])
        s[~mask] = np.nan
        # Forward-fill: holds velocity across flat staircase segments
        raw = s.ffill().fillna(0.0).to_numpy()

    if window > 1:
        return (
            pd.Series(raw)
            .rolling(window=window, min_periods=1, center=True)
            .mean()
            .to_numpy(dtype=float)
        )
    return raw


def _derive_motion_from_distance(
    df: pd.DataFrame,
    time_col: str = "t_s",
    distance_col: str = "travel_distance_m",
    window: int = 1,
    use_fill: bool = False,
) -> pd.DataFrame:
    """Add ``derived_speed_ms`` and ``derived_accel_ms2`` columns to *df*."""
    work = df.copy()
    work[time_col]     = pd.to_numeric(work[time_col],     errors="coerce")
    work[distance_col] = pd.to_numeric(work[distance_col], errors="coerce")
    work = work.sort_values(time_col)

    base = (
        work.dropna(subset=[time_col, distance_col])
        .drop_duplicates(subset=time_col)
        .copy()
    )
    if len(base) < 2:
        work["derived_speed_ms"]  = 0.0
        work["derived_accel_ms2"] = 0.0
        return work

    t_arr = base[time_col].values
    d_arr = base[distance_col].values

    # 1. Speed — sample-and-hold to suppress staircase zero-spikes
    base["derived_speed_ms"] = _differentiate_and_smooth(
        t_arr, d_arr, window=window, fill_staircase=use_fill
    )
    # 2. Acceleration — no staircase fill (zero accel is physically valid)
    base["derived_accel_ms2"] = _differentiate_and_smooth(
        t_arr, base["derived_speed_ms"].values, window=window, fill_staircase=False
    )

    work = work.merge(
        base[[time_col, "derived_speed_ms", "derived_accel_ms2"]],
        on=time_col, how="left",
    )
    work["derived_speed_ms"]  = work["derived_speed_ms"].ffill().bfill().fillna(0.0)
    work["derived_accel_ms2"] = work["derived_accel_ms2"].ffill().bfill().fillna(0.0)
    return work


# ── Public: data loading ──────────────────────────────────────────────────────

def prepare_run_df(
    df: pd.DataFrame,
    reduction_mode: str = "none",
    window: int = 1,
) -> pd.DataFrame:
    """Process a single raw CSV DataFrame into an analysis-ready DataFrame.

    Strips column whitespace, casts all CSV columns to float, assigns phase
    labels, applies point reduction, and derives speed/acceleration from the
    distance signal.

    Args:
        df:             Raw DataFrame from ``pd.read_csv``.
        reduction_mode: ``"none"`` / ``"first_last"`` / ``"center"`` /
                        ``"first"`` / ``"last"``.
        window:         Rolling-mean window for derived-speed smoothing.

    Returns:
        Processed DataFrame with added ``phase``, ``derived_speed_ms``,
        ``derived_accel_ms2`` columns.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df[CSV_COLS] = df[CSV_COLS].apply(pd.to_numeric, errors="coerce")
    df["phase"] = df["t_s"].map(_phase_from_time)
    df = _reduce_points(df, mode=reduction_mode)
    use_fill = reduction_mode in ("none", "first_last")
    df = _derive_motion_from_distance(df, window=window, use_fill=use_fill)
    return df


def load_all_runs(
    data_dir: str | Path | None = None,
    reduction_mode: str = "none",
    window: int = 1,
) -> dict[int, pd.DataFrame]:
    """Load every ``exp_speed_*.csv`` into a dict keyed by speed command.

    Args:
        data_dir:       CSV directory.  ``None`` → auto-detect from cwd.
        reduction_mode: ``"none"`` keeps all rows; ``"center"`` / ``"first"`` /
                        ``"last"`` / ``"first_last"`` collapse repeated blocks.
        window:         Rolling-mean window for derived-speed smoothing.

    Returns:
        ``{speed_cmd: DataFrame}`` with added ``phase``, ``derived_speed_ms``,
        ``derived_accel_ms2`` columns, sorted by speed command.
    """
    resolved = _resolve_data_dir() if data_dir is None else Path(data_dir)
    csv_paths = sorted(resolved.glob("exp_speed_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No exp_speed_*.csv files found in {resolved}")

    runs: dict[int, pd.DataFrame] = {}
    for p in csv_paths:
        speed = _infer_speed_from_path(p)
        raw: pd.DataFrame = pd.read_csv(p)  # type: ignore[assignment]
        runs[speed] = prepare_run_df(
            raw, reduction_mode=reduction_mode, window=window
        )

    print(f"Loaded {len(runs)} runs from {resolved}")
    return dict(sorted(runs.items()))


# ── Public: analysis ──────────────────────────────────────────────────────────

def estimate_motion_delay(df: pd.DataFrame) -> dict:
    """Estimate command-on → motion-onset delay for one run.

    Args:
        df: A single run DataFrame (output of :func:`prepare_run_df`).

    Returns:
        Dict with keys ``command_time_s``, ``motion_onset_time_s``,
        ``motion_delay_s``, ``motion_threshold_ms``  (all floats, NaN on
        failure).
    """
    work = df.sort_values("t_s").drop_duplicates(subset="t_s", keep="first").copy()
    nan_out = {
        "command_time_s": np.nan, "motion_onset_time_s": np.nan,
        "motion_delay_s": np.nan, "motion_threshold_ms":  np.nan,
    }
    if len(work) == 0:
        return nan_out

    work["speed_smooth"] = (
        work["derived_speed_ms"].rolling(window=7, min_periods=1, center=True).mean()
    )
    idle   = work[work["phase"] == "idle"]
    steady = work[work["phase"] == "steady"]

    command_time = float(steady["t_s"].min()) if len(steady) else float(IDLE_S)
    baseline     = float(idle["speed_smooth"].mean()) if len(idle) else 0.0
    noise        = float(idle["speed_smooth"].std())  if len(idle) else 0.0
    steady_ref   = (float(steady["speed_smooth"].median())
                    if len(steady) else float(work["speed_smooth"].max()))
    threshold = max(
        baseline + 3.0 * (0.0 if np.isnan(noise) else noise),
        0.05 * steady_ref,
        0.01,
    )
    post_cmd    = work[work["t_s"] >= command_time]
    onset_cands = post_cmd[post_cmd["speed_smooth"] >= threshold]
    onset_time  = float(onset_cands["t_s"].iloc[0]) if len(onset_cands) else np.nan
    delay       = float(onset_time - command_time)   if np.isfinite(onset_time) else np.nan

    return {
        "command_time_s":     command_time,
        "motion_onset_time_s": onset_time,
        "motion_delay_s":      delay,
        "motion_threshold_ms": float(threshold),
    }


