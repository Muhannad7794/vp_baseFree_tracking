# linear_sigma_model.py
import os
import json
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd


POS_AXES: List[str] = ["X_pose", "Y_pose", "Z_pose"]
ROT_AXES: List[str] = ["X_rot", "Y_rot", "Z_rot"]
ALL_AXES: List[str] = POS_AXES + ROT_AXES

# Scenario groupings:
STABLE_SCENARIOS = [
    "StillOnTripod",
    "handheld_still",
]

SLOW_TRIPOD_SCENARIOS = [
    "controlled_on_tripod_pan",
    "controlled_on_tripod_tilt",
]

CONTROLLED_HANDHELD_SCENARIOS = [
    "controlled_handheld_pan",
    "controlled_handheld_tilt",
]

MEDIUM_SCENARIOS = [
    "slide_handheld",
    "travel_handheld",
]

FAST_SCENARIOS = [
    "fast_pan_tripod",
    "fast_tilt_tripod",
    "handheld_full_nav",
]

# For the linear model only global stable vs dynamic split is needed.
# Middle groups are for non-linear models.
LINEAR_STABLE = STABLE_SCENARIOS + SLOW_TRIPOD_SCENARIOS
LINEAR_DYNAMIC = CONTROLLED_HANDHELD_SCENARIOS + MEDIUM_SCENARIOS + FAST_SCENARIOS


def add_rolling_sigma(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute rolling std of per-axis acceleration over 'window' frames,
    grouped by label, and add sigma_* columns.
    """
    if "label" not in df.columns:
        raise ValueError("Expected 'label' column in derivatives CSV")

    for axis in ALL_AXES:
        a_col = f"A_{axis}"
        sigma_col = f"sigma_{axis}"

        if a_col not in df.columns:
            raise ValueError(f"Missing acceleration column '{a_col}'")

        df[sigma_col] = df.groupby("label")[a_col].transform(
            lambda s: s.rolling(window=window, min_periods=window // 2).std()
        )

    return df


def compute_sigma_ranges(
    df: pd.DataFrame,
    window: int,
    stable_scenarios: List[str],
    dynamic_scenarios: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute global MinSigma/MaxSigma per axis using scenario labels.
    """
    if "scenario" not in df.columns:
        raise ValueError("Expected 'scenario' column in derivatives CSV")

    ranges: Dict[str, Dict[str, float]] = {}

    for axis in ALL_AXES:
        sigma_col = f"sigma_{axis}"
        if sigma_col not in df.columns:
            raise ValueError(f"Missing sigma column '{sigma_col}'")

        stable_vals = df.loc[df["scenario"].isin(stable_scenarios), sigma_col].dropna()
        dynamic_vals = df.loc[
            df["scenario"].isin(dynamic_scenarios), sigma_col
        ].dropna()

        if stable_vals.empty or dynamic_vals.empty:
            print(
                f"[linear_sigma_model] Warning: empty σ samples for axis {axis}; "
                f"stable={len(stable_vals)}, dynamic={len(dynamic_vals)}"
            )
            continue

        min_sigma = float(np.percentile(stable_vals, 90))
        max_sigma = float(np.percentile(dynamic_vals, 90))

        # Guard against pathological cases
        if not np.isfinite(min_sigma):
            min_sigma = float(stable_vals.median())
        if not np.isfinite(max_sigma):
            max_sigma = float(dynamic_vals.median())

        if max_sigma <= min_sigma:
            # Fallback: spread them a bit
            eps = abs(min_sigma) * 0.1 if min_sigma != 0 else 1e-4
            max_sigma = min_sigma + eps

        ranges[axis] = {
            "window": window,
            "min_sigma": min_sigma,
            "max_sigma": max_sigma,
        }

    return ranges


def apply_linear_inverse_mapping(
    df: pd.DataFrame,
    sigma_ranges: Dict[str, Dict[str, float]],
    pos_min_speed: float = 3.0,
    pos_max_speed: float = 20.0,
    rot_min_speed: float = 5.0,
    rot_max_speed: float = 40.0,
) -> Dict[str, Dict[str, float]]:
    """
    Add InterpSpeed_* columns based on σ and sigma_ranges.

    Returns a dict mapping axis -> speed config that we can store in JSON.
    """
    speed_cfg: Dict[str, Dict[str, float]] = {}

    for axis, rng in sigma_ranges.items():
        sigma_col = f"sigma_{axis}"
        out_col = f"InterpSpeed_{axis}"

        if sigma_col not in df.columns:
            print(
                f"[linear_sigma_model] Warning: missing σ column {sigma_col}, "
                f"skipping mapping for axis {axis}"
            )
            continue

        min_sigma = rng["min_sigma"]
        max_sigma = rng["max_sigma"]

        if axis in POS_AXES:
            min_speed = pos_min_speed
            max_speed = pos_max_speed
        else:
            min_speed = rot_min_speed
            max_speed = rot_max_speed

        sigma = df[sigma_col].to_numpy(dtype=float)
        sigma_clamped = np.clip(sigma, min_sigma, max_sigma)

        if np.isclose(max_sigma, min_sigma):
            df[out_col] = (min_speed + max_speed) / 2.0
        else:
            # Linear inverse mapping:
            # σ = min_sigma  -> InterpSpeed = max_speed (very responsive)
            # σ = max_sigma  -> InterpSpeed = min_speed (heavy smoothing)
            df[out_col] = max_speed + (min_speed - max_speed) * (
                (sigma_clamped - min_sigma) / (max_sigma - min_sigma)
            )

        speed_cfg[axis] = {
            "min_speed": float(min_speed),
            "max_speed": float(max_speed),
        }

    return speed_cfg


def main():
    parser = argparse.ArgumentParser(
        description="Linear sigma model: rolling σ and inverse mapping."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/derived/tracking_derivatives.csv",
        help="Input derivatives CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/modeled/tracking_modelled_sigma.csv",
        help="Output CSV (extended with σ and InterpSpeed columns)",
    )
    parser.add_argument(
        "--config-output",
        type=str,
        default="data/config/linear_sigma_ranges.json",
        help="Path to JSON config with σ ranges and speed settings",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=25,
        help="Rolling window size (in frames) for σ computation",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input derivatives file not found: {args.input}")

    df = pd.read_csv(args.input)

    # 1) rolling σ per axis
    df = add_rolling_sigma(df, window=args.window)

    # 2) global MinSigma / MaxSigma per axis
    sigma_ranges = compute_sigma_ranges(
        df,
        window=args.window,
        stable_scenarios=LINEAR_STABLE,
        dynamic_scenarios=LINEAR_DYNAMIC,
    )

    # 3) linear inverse mapping σ -> InterpSpeed
    speed_cfg = apply_linear_inverse_mapping(df, sigma_ranges)

    # 4) Write extended CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[linear_sigma_model] Created CSV written to {args.output}")

    # 5) Write JSON config (σ ranges + speed cfg)
    os.makedirs(os.path.dirname(args.config_output), exist_ok=True)
    config = {
        "window": args.window,
        "axes": {},
    }

    for axis, rng in sigma_ranges.items():
        axis_cfg = {
            "min_sigma": rng["min_sigma"],
            "max_sigma": rng["max_sigma"],
        }
        if axis in speed_cfg:
            axis_cfg.update(speed_cfg[axis])
        config["axes"][axis] = axis_cfg

    with open(args.config_output, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"[linear_sigma_model] σ ranges and speeds saved to {args.config_output}")


if __name__ == "__main__":
    main()
