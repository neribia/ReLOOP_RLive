"""W&B Training Plotter.

Fetches rollout and eval metrics logged by the SB3 training scripts from
Weights & Biases and renders them in a clean 2×2 matplotlib figure.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

# ---------------------------------------------------------------------------
# Metric layout definition — Clean 2x2 Configuration
# ---------------------------------------------------------------------------

ROLLOUT_METRICS: list[tuple[str, str, str]] = [
    ("rollout/success_rate",  "Rollout — Success Rate",   "Success Rate"),
    ("rollout/ep_rew_mean",   "Rollout — Mean Reward",    "Mean Reward"),
]

EVAL_METRICS: list[tuple[str, str, str]] = [
    ("eval/success_rate",    "Eval — Success Rate",    "Success Rate"),
    ("eval/mean_ep_length",  "Eval — Mean Epoch Length", "Mean Epoch Length"),
]

X_KEY = "global_step"
_FALLBACK_X = "_step"


# ---------------------------------------------------------------------------
# Local cache helpers
# ---------------------------------------------------------------------------

_CACHE_DIR = Path.home() / ".cache" / "wandb_plot"


def _cache_path(run_id: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = run_id.replace("/", "_")
    return _CACHE_DIR / f"{safe_id}.pkl"


def _load_cache(run_id: str) -> pd.DataFrame | None:
    p = _cache_path(run_id)
    if p.exists():
        print(f" -> Loading cached history from {p}")
        return pd.read_pickle(p)
    return None


def _save_cache(run_id: str, df: pd.DataFrame) -> None:
    p = _cache_path(run_id)
    df.to_pickle(p)
    print(f" -> Cache saved to {p}")


# ---------------------------------------------------------------------------
# Object-Oriented Data Management
# ---------------------------------------------------------------------------

class WandbRunDataset:
    """Handles connection and isolated timeline extraction for rollout/eval curves."""

    def __init__(self, run_path: str, max_timesteps: int | None = None,
                 timeout: int = 60, no_cache: bool = False):
        self.run_path = run_path
        self.max_timesteps = max_timesteps
        self.timeout = timeout
        self.no_cache = no_cache
        self.run_name: str = ""
        self.rollout_df = pd.DataFrame()
        self.eval_df = pd.DataFrame()
        self._fetch_histories()

    def _fetch_histories(self) -> None:
        api = wandb.Api(timeout=self.timeout)
        run = api.run(self.run_path)

        # Extract meaningful name or fallback to ID
        self.run_name = run.name if (run.name and not run.name.startswith("http")) else run.id

        print(f"[{self.run_name}] Fetching history frames...")

        run_id = self.run_path.split("/")[-1]

        # ── Try local cache first ─────────────────────────────────────────────
        if not self.no_cache:
            cached = _load_cache(run_id)
            if cached is not None:
                full_df = cached
                print(f" -> {len(full_df)} rows loaded from cache (use --no-cache to force re-download).")
            else:
                full_df = self._download(run)
                if full_df is None:
                    return
                _save_cache(run_id, full_df)
        else:
            full_df = self._download(run)
            if full_df is None:
                return
            _save_cache(run_id, full_df)

        if full_df.empty:
            print(f" -> [warning] Run {self.run_name} returned an empty history.")
            return

        # Standardize our timeline index
        if X_KEY not in full_df.columns and _FALLBACK_X in full_df.columns:
            full_df = full_df.rename(columns={_FALLBACK_X: X_KEY})
        elif X_KEY not in full_df.columns:
            full_df[X_KEY] = range(len(full_df))

        # Sort chronologically and forward/backward fill step sequence gaps
        full_df = full_df.sort_values(X_KEY)
        full_df[X_KEY] = full_df[X_KEY].ffill().bfill()

        # Isolate Rollout curves
        r_target_cols = [X_KEY] + [m[0] for m in ROLLOUT_METRICS if m[0] in full_df.columns]
        self.rollout_df = full_df[r_target_cols].dropna(
            subset=[m[0] for m in ROLLOUT_METRICS if m[0] in full_df.columns], how="all"
        )

        # Isolate Eval curves
        e_target_cols = [X_KEY] + [m[0] for m in EVAL_METRICS if m[0] in full_df.columns]
        self.eval_df = full_df[e_target_cols].dropna(
            subset=[m[0] for m in EVAL_METRICS if m[0] in full_df.columns], how="all"
        )

        # Filter timeline cutoffs if applicable
        if self.max_timesteps is not None:
            if not self.rollout_df.empty:
                self.rollout_df = self.rollout_df[self.rollout_df[X_KEY] <= self.max_timesteps]
            if not self.eval_df.empty:
                self.eval_df = self.eval_df[self.eval_df[X_KEY] <= self.max_timesteps]

        print(f" -> Processed {len(self.rollout_df)} rollout entries and {len(self.eval_df)} eval entries.")

    def _download(self, run) -> pd.DataFrame | None:
        """Download history via per-key scan_history calls, then merge on _step.

        Each metric is logged in its own wandb.log() call (separate rows).
        scan_history(keys=[A, B]) returns rows where BOTH A and B are non-null
        (intersection semantics), so combining multiple metrics yields 0 rows.

        Solution:
          1. Fetch each metric key individually as (metric, _step) pairs.
          2. Outer-merge all series on _step to reconstruct the full timeline.
          3. Fetch (global_step, _step) separately and left-merge for real x-axis.
        """
        all_keys = [m[0] for m in ROLLOUT_METRICS] + [m[0] for m in EVAL_METRICS]

        # ── Phase 1: per-key scan_history with _step ──────────────────────────
        per_key_frames: dict[str, pd.DataFrame] = {}
        scan_ok = True
        for key in all_keys:
            rows: list[dict] = []
            try:
                for row in run.scan_history(keys=[key, "_step"], page_size=5000):
                    rows.append(row)
                if rows:
                    df = pd.DataFrame(rows)
                    # keep _step + metric; drop rows where metric is null
                    cols = [c for c in ["_step", key] if c in df.columns]
                    df = df[cols].dropna(subset=[key])
                    per_key_frames[key] = df
                    print(f" -> scan_history '{key}': {len(df)} rows.")
                else:
                    # _step might not be co-logged; try key alone
                    for row in run.scan_history(keys=[key], page_size=5000):
                        rows.append(row)
                    if rows:
                        df = pd.DataFrame(rows)[[key]].dropna()
                        df["_step"] = range(len(df))   # synthetic step index
                        per_key_frames[key] = df
                        print(f" -> scan_history '{key}' (synthetic _step): {len(df)} rows.")
                    else:
                        print(f" -> scan_history '{key}': 0 rows — fallback triggered.")
                        scan_ok = False
                        break
            except Exception as exc:
                print(f" -> scan_history '{key}' failed: {exc}")
                scan_ok = False
                break

        if per_key_frames and scan_ok:
            # ── Phase 2: outer-merge all per-key series on _step ──────────────
            merged: pd.DataFrame | None = None
            for key, df in per_key_frames.items():
                if merged is None:
                    merged = df
                else:
                    merged = merged.merge(df, on="_step", how="outer")
            assert merged is not None

            # ── Phase 3: fetch global_step → merge by _step ───────────────────
            try:
                gstep_rows: list[dict] = []
                for row in run.scan_history(keys=["global_step", "_step"], page_size=5000):
                    gstep_rows.append(row)
                if gstep_rows:
                    gstep_df = pd.DataFrame(gstep_rows)
                    cols = [c for c in ["_step", "global_step"] if c in gstep_df.columns]
                    gstep_df = gstep_df[cols].dropna()
                    merged = merged.merge(gstep_df, on="_step", how="left")
                    print(f" -> global_step merged from {len(gstep_df)} rows.")
                else:
                    print(f" -> global_step: 0 rows — will use _step as x-axis.")
            except Exception as exc:
                print(f" -> global_step fetch failed — will use _step: {exc}")

            print(f" -> Combined: {len(merged)} rows × {len(merged.columns)} cols.")
            return merged

        # ── Phase 4: unfiltered run.history() fallback ────────────────────────
        print(f" -> [retry] Falling back to unfiltered run.history()…")
        for samples in (100_000, 10_000, 5_000, 2_000, 500, 200, 100):
            try:
                df = run.history(samples=samples, pandas=True)
                if not df.empty:
                    print(f" -> Downloaded {len(df)} rows (samples={samples}).")
                    return df
            except Exception as exc:
                print(f" -> [retry] run.history(samples={samples}) failed: {exc}")

        print(f" -> [error] All download attempts exhausted.")
        return None

    def get_metric_timeline(self, metric_key: str) -> pd.DataFrame:
        """Looks up data from the appropriate isolated historical dataframe."""
        target_df = self.eval_df if metric_key.startswith("eval") else self.rollout_df

        if target_df.empty or metric_key not in target_df.columns:
            return pd.DataFrame()

        return target_df[[X_KEY, metric_key]].dropna().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting Engine
# ---------------------------------------------------------------------------

def plot_runs(
    run_ids: Sequence[str] | None = None,
    run_names: Sequence[str] | None = None,
    project: str = "rlive-train",
    entity: str | None = None,
    max_timesteps: Sequence[int] | int | None = None,
    output: str | None = None,
    figsize: tuple[float, float] = (12, 8),
    title: str = "SB3 Training Curves (W&B)",
    timeout: int = 60,
    no_cache: bool = False,
) -> plt.Figure:
    """Plot unified rollout and eval metrics across multiple run datasets."""

    if run_ids:
        if isinstance(max_timesteps, (list, tuple)):
            limits = list(max_timesteps)
            while len(limits) < len(run_ids):
                limits.append(None)
        else:
            limits = [max_timesteps] * len(run_ids)

        # Build display name list — fall back to W&B run name when not provided
        name_overrides: list[str | None] = list(run_names) if run_names else []
        while len(name_overrides) < len(run_ids):
            name_overrides.append(None)

        datasets: list[WandbRunDataset] = []
        for rid, limit, name_override in zip(run_ids, limits, name_overrides):
            run_path = rid if rid.count("/") >= 2 else f"{entity or project}/{project}/{rid}".replace(f"{project}/{project}", project)
            try:
                ds = WandbRunDataset(run_path, limit, timeout=timeout, no_cache=no_cache)
                if name_override:
                    ds.run_name = name_override
                datasets.append(ds)
            except Exception as exc:
                print(f"[warning] Failed to load data slice for {rid}: {exc}")
    else:
        api = wandb.Api(timeout=timeout)
        path = f"{entity}/{project}" if entity else project
        global_limit = max_timesteps[0] if isinstance(max_timesteps, (list, tuple)) and max_timesteps else max_timesteps
        datasets = []
        for run in api.runs(path):
            datasets.append(WandbRunDataset(
                f"{run.entity}/{run.project}/{run.id}", global_limit,
                timeout=timeout, no_cache=no_cache,
            ))

    if not datasets:
        raise RuntimeError("Dataset pool is completely empty. Verify your targets.")

    # constrained_layout handles large row labels and the external legend box
    # automatically — no manual subplots_adjust or tight_layout needed.
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, layout="constrained")
    fig.suptitle(title, fontsize=14, fontweight="bold")

    row_setups = [
        (ROLLOUT_METRICS, "Rollout"),
        (EVAL_METRICS,    "Eval"),
    ]

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colours = [p["color"] for p in prop_cycle]

    legend_tracker: dict[str, plt.Line2D] = {}

    for row_idx, (metrics, _row_label) in enumerate(row_setups):
        for col_idx, (metric_key, subplot_title, y_label) in enumerate(metrics):
            ax: plt.Axes = axes[row_idx, col_idx]

            ax.set_title(subplot_title, fontsize=12, fontweight="semibold")
            ax.set_ylabel(y_label, fontsize=10)
            ax.set_xlabel("Timesteps", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.4)

            for run_idx, dataset in enumerate(datasets):
                colour = colours[run_idx % len(colours)]
                series = dataset.get_metric_timeline(metric_key)

                if series.empty:
                    continue

                raw_y = series[metric_key]

                # Precisely mirror the W&B dashboard smoothing behavior:
                if metric_key.startswith("eval"):
                    # Rolling window average (size 10); min_periods=1 lets the
                    # first point through at its raw value and scales the divisor
                    # up naturally (÷1, ÷2, … ÷10) — matching W&B's sparse eval display.
                    display_y = raw_y.rolling(window=10, min_periods=1).mean()
                else:
                    # De-biased EMA (adjust=True cancels initialisation drag toward
                    # zero) for dense, noisy rollout streams — mirrors W&B smoothing.
                    display_y = raw_y.ewm(alpha=0.15, adjust=True).mean()

                line, = ax.plot(
                    series[X_KEY],
                    display_y,
                    color=colour,
                    linewidth=1.8,
                    alpha=0.95,
                    rasterized=True,
                )

                if dataset.run_name not in legend_tracker:
                    legend_tracker[dataset.run_name] = line

    # Synchronize y-axis limits for both success-rate subplots so they share
    # the same scale and are directly comparable.
    ax_rollout_sr: plt.Axes = axes[0, 0]
    ax_eval_sr:    plt.Axes = axes[1, 0]
    sr_y_min = min(ax_rollout_sr.get_ylim()[0], ax_eval_sr.get_ylim()[0])
    sr_y_max = max(ax_rollout_sr.get_ylim()[1], ax_eval_sr.get_ylim()[1])
    ax_rollout_sr.set_ylim(sr_y_min, sr_y_max)
    ax_eval_sr.set_ylim(sr_y_min, sr_y_max)

    # Global legend — 'outside lower center' is the constrained_layout-native
    # placement: the engine reserves just enough space and centres the box.
    if legend_tracker:
        fig.legend(
            list(legend_tracker.values()),
            list(legend_tracker.keys()),
            loc="outside lower center",
            ncol=min(len(legend_tracker), 4),
            fontsize=10,
            frameon=True,
        )

    if output:
        is_pdf = output.lower().endswith(".pdf")
        save_kwargs: dict = {"bbox_inches": "tight", "dpi": 300}
        if is_pdf:
            save_kwargs["format"] = "pdf"
        fig.savefig(output, **save_kwargs)
        print(f"Figure saved successfully → {output}")
    else:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# CLI Entry
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI execution wrapper."""
    # --- PLOT SETTINGS ---
    DEFAULT_FIGSIZE:       tuple[float, float] = (9, 6)
    DEFAULT_FONTSIZE:      int                 = 11      # base font size; all labels scale from this

    # --- RUN SETTINGS ---
    #DEFAULT_RUN_IDS:       list[str] | None = ["1ny8vtxm", "vaqnq5p8"]
    #DEFAULT_RUN_NAMES:     list[str] | None = ["Sapien", "Simple"]
    DEFAULT_RUN_IDS:       list[str] | None = ["hvjcfp88", "nqhl4ebl", "bf09ibba", "suw1vu0i"]
    DEFAULT_RUN_NAMES:     list[str] | None = ["Sapien_lr3e-04", "Sapien_lr3e-05", "Simple_lr1e-04", "Simple_lr3e-04"]

    DEFAULT_TITLE:         str              = "" #"SB3 Training Curves (W&B)"
    DEFAULT_PROJECT:       str              = "rlive-train"
    DEFAULT_ENTITY:        str | None       = "models-hochschule-kempten"
    #DEFAULT_MAX_TIMESTEPS: list[int] | None = [1_120_000, 1_160_000]
    DEFAULT_MAX_TIMESTEPS: list[int] | None = None# [1_120_000, 1_160_000, 1_120_000, 1_160_000]# [1_120_000, 1_160_000]
    DEFAULT_OUTPUT:        str | None       = "training_runs_fail.pdf"

    parser = argparse.ArgumentParser(description="Plot synchronized SB3 performance metrics.")
    parser.add_argument("--run-ids",       nargs="+", default=DEFAULT_RUN_IDS,       help="W&B target run IDs.")
    parser.add_argument("--run-names",     nargs="+", default=DEFAULT_RUN_NAMES,
                        help="Display names for each run (used in the legend).")
    parser.add_argument("--title",         type=str,  default=DEFAULT_TITLE,         help="Figure suptitle.")
    parser.add_argument("--figsize",       nargs=2,   type=float, default=list(DEFAULT_FIGSIZE),
                        metavar=("W", "H"),           help="Figure width and height in inches.")
    parser.add_argument("--fontsize",      type=int,  default=DEFAULT_FONTSIZE,
                        help="Base font size for all labels.")
    parser.add_argument("--project",       type=str,  default=DEFAULT_PROJECT,       help="W&B Project name.")
    parser.add_argument("--entity",        type=str,  default=DEFAULT_ENTITY,        help="W&B space entity workspace.")
    parser.add_argument("--max-timesteps", nargs="+", type=int, default=DEFAULT_MAX_TIMESTEPS,
                        help="Cutoff timesteps array.")
    parser.add_argument("--output",        type=str,  default=DEFAULT_OUTPUT,
                        help="Optional output file path (.png / .pdf).")
    parser.add_argument("--timeout",       type=int,  default=60,
                        help="W&B API timeout in seconds (default: 60).")
    parser.add_argument("--no-cache",      action="store_true", default=False,
                        help="Force re-download even if a local cache exists.")

    args = parser.parse_args()

    plt.rcParams["font.size"] = args.fontsize

    plot_runs(
        run_ids=args.run_ids,
        run_names=args.run_names,
        title=args.title,
        project=args.project,
        entity=args.entity,
        max_timesteps=args.max_timesteps,
        output=args.output,
        figsize=(float(args.figsize[0]), float(args.figsize[1])),
        timeout=args.timeout,
        no_cache=args.no_cache,
    )


if __name__ == "__main__":
    main()
