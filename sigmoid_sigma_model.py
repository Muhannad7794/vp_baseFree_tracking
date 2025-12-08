# sigmoid_sigma_model.py

import os
import json
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


SIGMOID_STABLE = STABLE_SCENARIOS + SLOW_TRIPOD_SCENARIOS
SIGMOID_DYNAMIC = CONTROLLED_HANDHELD_SCENARIOS + MEDIUM_SCENARIOS + FAST_SCENARIOS


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
                f"[sigmoid_sigma_model] Warning: empty σ samples for axis {axis}; "
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
            eps = abs(min_sigma) * 0.1 if min_sigma != 0 else 1e-4
            max_sigma = min_sigma + eps

        ranges[axis] = {
            "window": window,
            "min_sigma": min_sigma,
            "max_sigma": max_sigma,
        }

    return ranges


def apply_sigmoid_inverse_mapping(
    df: pd.DataFrame,
    sigma_ranges: Dict[str, Dict[str, float]],
    pos_min_speed: float = 3.0,
    pos_max_speed: float = 20.0,
    rot_min_speed: float = 5.0,
    rot_max_speed: float = 40.0,
    steepness: float = 0.1,  # Controls how "sudden" the switch is
) -> Dict[str, Dict[str, float]]:

    speed_cfg = {}

    for axis, rng in sigma_ranges.items():
        sigma_col = f"sigma_{axis}"
        out_col = f"InterpSpeed_{axis}"

        if sigma_col not in df.columns:
            continue

        # 1. Get Ranges
        min_sigma = rng["min_sigma"]
        max_sigma = rng["max_sigma"]

        # 2. Set Speed Bounds
        if axis in ["X_pose", "Y_pose", "Z_pose"]:
            min_speed = pos_min_speed
            max_speed = pos_max_speed
        else:
            min_speed = rot_min_speed
            max_speed = rot_max_speed

        sigma = df[sigma_col].to_numpy(dtype=float)

        # 3. SIGMOID LOGIC STARTS HERE

        # Calculate the Midpoint (The "Tipping Point")
        midpoint = (min_sigma + max_sigma) / 2.0

        # Calculate the raw sigmoid curve (0.0 to 1.0)
        # We assume High Sigma = High Jitter = Needs Smoothing (Low Speed)
        # So we want the curve to go DOWN.

        # This formula creates a standard logistic curve
        # k controls the slope. A higher k means a sharper switch.
        k = steepness * (10.0 / (max_sigma - min_sigma))
        logistic_curve = 1 / (1 + np.exp(-k * (sigma - midpoint)))

        # Map 0.0-1.0 to MaxSpeed-MinSpeed
        # Note: Logistic goes Low->High. We want Low Sigma -> High Speed.
        # So we invert: Speed = Min + (Max-Min) * (1 - logistic)

        df[out_col] = min_speed + (max_speed - min_speed) * (1 - logistic_curve)

        # Save config
        speed_cfg[axis] = {
            "min_speed": min_speed,
            "max_speed": max_speed,
            "midpoint": midpoint,
            "steepness": steepness,
        }

    return speed_cfg


# -------------------------
# Plotting functions
# -------------------------


def plot_sigma_and_speed_by_scenario(
    scenario: str,
    axis: str,
    modeled_path: str = "data/modeled/tracking_modelled_sigma_sigmoid.csv",
    output_dir: str = "data/sigmoid_plots/sigmoid_model_scenarios",
) -> None:
    """
    For a given scenario and axis, overlay all takes:

    - 2 subplots: σ vs time, InterpSpeed vs time
    - each take (label) is a separate line
    - time is shifted so each take starts at t=0

    Saves a jpg to output_dir.
    """
    if axis not in ALL_AXES:
        raise ValueError(f"axis must be one of {ALL_AXES}, got {axis}")

    sigma_col = f"sigma_{axis}"
    speed_col = f"InterpSpeed_{axis}"

    if not os.path.exists(modeled_path):
        raise FileNotFoundError(f"Modeled CSV not found: {modeled_path}")

    df = pd.read_csv(modeled_path)
    if sigma_col not in df.columns or speed_col not in df.columns:
        raise ValueError(
            f"Expected columns '{sigma_col}' and '{speed_col}' in modeled CSV"
        )

    df_s = df[df["scenario"] == scenario]
    if df_s.empty:
        raise ValueError(f"No data for scenario '{scenario}'")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{scenario}_{axis}.jpg")

    plt.figure(figsize=(12, 7))
    ax_sigma = plt.subplot(2, 1, 1)
    ax_speed = plt.subplot(2, 1, 2)

    for label, g in df_s.groupby("label"):
        g = g.sort_values("time")
        t = g["time"].to_numpy()
        t0 = t - t[0]

        sigma = g[sigma_col].to_numpy()
        speed = g[speed_col].to_numpy()

        ax_sigma.plot(t0, sigma, label=label)
        ax_speed.plot(t0, speed, label=label)

    ax_sigma.set_ylabel("σ")
    ax_speed.set_ylabel("InterpSpeed")
    ax_speed.set_xlabel("time since start (s)")
    ax_sigma.set_title(f"{scenario} – {axis}: σ and InterpSpeed")

    ax_sigma.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[sigmoid_sigma_model] Saved scenario σ/speed plot to {out_path}")


def plot_sigma_bar_by_scenario(
    axis: str,
    modeled_path: str = "data/modeled/tracking_modelled_sigma_sigmoid.csv",
    output_dir: str = "data/sigmoid_plots/sigmoid_model_sigma_bars",
) -> None:
    """
    For a given axis, plot a bar chart of σ (90th percentile) per scenario.

    One figure per axis, one bar per scenario.
    """
    if axis not in ALL_AXES:
        raise ValueError(f"axis must be one of {ALL_AXES}, got {axis}")

    sigma_col = f"sigma_{axis}"

    if not os.path.exists(modeled_path):
        raise FileNotFoundError(f"Modeled CSV not found: {modeled_path}")

    df = pd.read_csv(modeled_path)
    if sigma_col not in df.columns:
        raise ValueError(f"Expected column '{sigma_col}' in modeled CSV")

    df = df.dropna(subset=[sigma_col])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"sigma_bar_{axis}.jpg")

    scenario_order = [
        "StillOnTripod",
        "handheld_still",
        "controlled_on_tripod_pan",
        "controlled_on_tripod_tilt",
        "controlled_handheld_pan",
        "controlled_handheld_tilt",
        "slide_handheld",
        "travel_handheld",
        "fast_pan_tripod",
        "fast_tilt_tripod",
        "handheld_full_nav",
    ]

    vals = []
    labels = []
    for scen in scenario_order:
        s = df.loc[df["scenario"] == scen, sigma_col]
        if s.empty:
            continue
        labels.append(scen)
        vals.append(np.percentile(s, 90))

    if not vals:
        raise ValueError("No σ values found for any scenario")

    x = np.arange(len(labels))

    plt.figure(figsize=(12, 6))

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(labels))]

    plt.bar(x, vals, color=colors)

    plt.xticks(
        x,
        labels,
        rotation=30,
        ha="right",
        fontsize=8,
    )

    plt.ylabel(r"$\sigma$")
    plt.xlabel("scenario")
    plt.title(rf"$\sigma$ (90th percentile) by scenario – {axis}")

    ymax = max(vals)
    plt.ylim(0, ymax * 1.1)

    from matplotlib.ticker import MaxNLocator

    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=False))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[sigmoid_sigma_model] Saved σ bar plot to {out_path}")


# -------------------------
# CLI entry point
# -------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Sigmoid sigma model: rolling σ and inverse mapping."
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
        default="data/modeled/tracking_modelled_sigma_sigmoid.csv",
        help="Output CSV (created with σ and InterpSpeed columns)",
    )
    parser.add_argument(
        "--config-output",
        type=str,
        default="data/config/sigmoid_sigma_ranges.json",
        help="Path to JSON config with σ ranges and speed settings",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=25,
        help="Rolling window size (in frames) for σ computation",
    )
    parser.add_argument(
        "--plot-scenario",
        type=str,
        help="Scenario to plot σ and InterpSpeed for (overlays all takes) – used with --plot-axis",
    )
    parser.add_argument(
        "--plot-axis",
        type=str,
        help="Axis to plot for scenario plots / boxplots (e.g. X_pose, Y_rot)",
    )
    parser.add_argument(
        "--plot-bar-axis",
        type=str,
        help="Axis to plot σ bar chart for across all scenarios (e.g. X_pose, Y_rot)",
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
        stable_scenarios=SIGMOID_STABLE,
        dynamic_scenarios=SIGMOID_DYNAMIC,
    )

    # 3) sigmoid inverse mapping σ -> InterpSpeed
    speed_cfg = apply_sigmoid_inverse_mapping(df, sigma_ranges)

    # 4) Write modeled CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[sigmoid_sigma_model] Created CSV written to {args.output}")

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

    print(f"[sigmoid_sigma_model] σ ranges and speeds saved to {args.config_output}")

    # 6) plotting

    if args.plot_scenario and args.plot_axis:
        plot_sigma_and_speed_by_scenario(
            scenario=args.plot_scenario,
            axis=args.plot_axis,
            modeled_path=args.output,
        )

    if args.plot_bar_axis:
        plot_sigma_bar_by_scenario(
            axis=args.plot_bar_axis,
            modeled_path=args.output,
        )


if __name__ == "__main__":
    main()
