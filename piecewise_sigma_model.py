#  piecewise_sigma_model.py
"""
piecewise_sigma_model.py

Compute rolling sigma per axis, calibrate a piecewise-linear mapping
from sigma to InterpSpeed using scenario groups, and write:

- data/modeled/tracking_modelled_sigma_piecewise.csv
- data/config/piecewise_sigma_ranges.json

This script is intentionally separate from linear_sigma_model.py so
models don't interfere with each other.
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd


# ---- Scenario groups for calibration ----------------------------------------

GROUPS = {
    "static": [
        "StillOnTripod",
        "handheld_still",
    ],
    "slow_tripod": [
        "controlled_on_tripod_pan",
        "controlled_on_tripod_tilt",
    ],
    "controlled_handheld": [
        "controlled_handheld_pan",
        "controlled_handheld_tilt",
    ],
    "medium_complex": [
        "slide_handheld",
        "travel_handheld",
    ],
    "fast_aggressive": [
        "fast_pan_tripod",
        "fast_tilt_tripod",
        "handheld_full_nav",
    ],
}

GROUP_ORDER = [
    "static",
    "slow_tripod",
    "controlled_handheld",
    "medium_complex",
    "fast_aggressive",
]


# ---- Helpers ----------------------------------------------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def compute_rolling_sigma(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    For each acceleration column A_*, compute rolling std per label
    and append sigma_* columns (where * matches the pose axis).
    """
    df = df.sort_values(["label", "time"]).copy()

    accel_cols = [c for c in df.columns if c.startswith("A_")]
    if not accel_cols:
        raise ValueError("No acceleration columns (A_*) found in input.")

    for acc_col in accel_cols:
        axis = acc_col[2:]  # strip 'A_'
        sigma_col = f"sigma_{axis}"
        df[sigma_col] = (
            df.groupby("label")[acc_col]
            .rolling(window=window, min_periods=5)
            .std()
            .reset_index(level=0, drop=True)
        )

    return df


def percentile_safe(values: pd.Series, q: float) -> float:
    values = values.dropna()
    if len(values) == 0:
        return np.nan
    return float(np.percentile(values, q))


def calibrate_piecewise_breaks(
    df: pd.DataFrame,
    sigma_col: str,
) -> List[float]:
    """
    Compute group-wise 90th percentile for sigma_col, then enforce
    a monotonically increasing sequence in GROUP_ORDER.
    Returns a list of sigma_breaks (>= 2 elements expected).
    """
    per_group = {}
    for g in GROUP_ORDER:
        scen_names = GROUPS.get(g, [])
        mask = df["scenario"].isin(scen_names)
        vals = df.loc[mask, sigma_col]
        per_group[g] = percentile_safe(vals, 90)

    # Enforce monotonicity and drop NaNs
    breaks: List[float] = []
    current = 0.0
    for g in GROUP_ORDER:
        val = per_group.get(g)
        if val is None or np.isnan(val):
            continue
        if not breaks:
            current = val
        else:
            current = max(current, val)
        breaks.append(float(current))

    # If fewer than 2 breaks, fall back to global range
    if len(breaks) < 2:
        vals = df[sigma_col].dropna()
        if len(vals):
            lo, hi = np.percentile(vals, [10, 90])
            breaks = [float(lo), float(hi)]
        else:
            breaks = [0.0, 1.0]

    return breaks


def load_linear_speeds(
    axis: str,
    linear_config_path: str,
    default_min_speed: float = 20.0,
    default_max_speed: float = 40.0,
) -> (float, float):
    """
    Try to reuse min/max speeds from linear_sigma_ranges.json if available.
    Otherwise fall back to generic defaults.
    """
    if os.path.exists(linear_config_path):
        with open(linear_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        axes_cfg = cfg.get("axes", {})
        axis_cfg = axes_cfg.get(axis, {})
        min_speed = float(axis_cfg.get("min_speed", default_min_speed))
        max_speed = float(axis_cfg.get("max_speed", default_max_speed))
        return min_speed, max_speed
    return default_min_speed, default_max_speed


def piecewise_speed_from_sigma(
    sigma: float,
    breaks: List[float],
    speeds: List[float],
) -> float:
    """
    Scalar piecewise-linear mapping:
    - breaks: [b0, b1, ..., bK-1] (increasing)
    - speeds: [s0, s1, ..., sK-1] (typically decreasing)

    Below b0 → s0, above b_{K-1} → s_{K-1}.
    Between → linear interpolation between segment endpoints.
    """
    if np.isnan(sigma):
        return np.nan

    K = len(breaks)
    if K == 0:
        return np.nan
    if K == 1:
        return float(speeds[0])

    if sigma <= breaks[0]:
        return float(speeds[0])
    if sigma >= breaks[-1]:
        return float(speeds[-1])

    for k in range(K - 1):
        b0, b1 = breaks[k], breaks[k + 1]
        if b1 <= b0:
            continue
        if sigma <= b1:
            t = (sigma - b0) / (b1 - b0)
            return float(speeds[k] + t * (speeds[k + 1] - speeds[k]))

    return float(speeds[-1])


# ---- Main -------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Piecewise σ→InterpSpeed model based on scenario groups."
    )
    parser.add_argument(
        "--input",
        default="data/derived/tracking_derivatives.csv",
        help="Input CSV with dt, V_*, A_* (default: data/derived/tracking_derivatives.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/modeled/tracking_modelled_sigma_piecewise.csv",
        help="Output CSV with sigma_* and InterpSpeed_* (default: data/modeled/tracking_modelled_sigma_piecewise.csv)",
    )
    parser.add_argument(
        "--config-output",
        default="data/config/piecewise_sigma_ranges.json",
        help="Output JSON with per-axis sigma breaks and speed levels.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=25,
        help="Rolling window size in samples for sigma (default: 25).",
    )
    parser.add_argument(
        "--linear-config",
        default="data/config/linear_sigma_ranges.json",
        help="Optional linear model config to reuse min/max speeds.",
    )

    args = parser.parse_args()

    print("[piecewise_sigma_model] Loading derivatives:", args.input)
    df = pd.read_csv(args.input)

    required_cols = {"time", "label", "scenario"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    # 1) Rolling sigma per axis
    print(f"[piecewise_sigma_model] Computing rolling σ with window={args.window}")
    df = compute_rolling_sigma(df, window=args.window)

    # 2) Calibrate breaks and speeds per axis
    accel_cols = [c for c in df.columns if c.startswith("A_")]
    axes = [c[2:] for c in accel_cols]  # 'X_pose', etc.

    axes_cfg: Dict[str, Dict] = {}

    for axis in axes:
        sigma_col = f"sigma_{axis}"
        if sigma_col not in df.columns:
            continue

        print(f"[piecewise_sigma_model] Calibrating axis {axis} ...")
        sigma_breaks = calibrate_piecewise_breaks(df, sigma_col=sigma_col)
        min_speed, max_speed = load_linear_speeds(
            axis, args.linear_config, default_min_speed=20.0, default_max_speed=40.0
        )

        # Speeds: monotonically decreasing from max_speed (stable) to min_speed (aggressive)
        num_levels = len(sigma_breaks)
        speed_levels = np.linspace(max_speed, min_speed, num_levels).tolist()

        axes_cfg[axis] = {
            "sigma_breaks": [float(x) for x in sigma_breaks],
            "speed_levels": [float(x) for x in speed_levels],
            "min_sigma": float(sigma_breaks[0]),
            "max_sigma": float(sigma_breaks[-1]),
            "min_speed": float(min_speed),
            "max_speed": float(max_speed),
        }

        # 3) Apply mapping to get InterpSpeed per frame
        interp_col = f"InterpSpeed_{axis}"

        df[interp_col] = df[f"sigma_{axis}"].apply(
            lambda s: piecewise_speed_from_sigma(
                s, breaks=sigma_breaks, speeds=speed_levels
            )
        )

    # 4) Save extended CSV
    ensure_dir(args.output)
    df.to_csv(args.output, index=False)
    print(
        "[piecewise_sigma_model] Created CSV written to",
        args.output,
    )

    # 5) Save config JSON
    cfg = {
        "window_size": args.window,
        "groups": GROUPS,
        "axes": axes_cfg,
    }

    ensure_dir(args.config_output)
    with open(args.config_output, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(
        "[piecewise_sigma_model] σ breaks and speeds saved to",
        args.config_output,
    )


if __name__ == "__main__":
    main()
