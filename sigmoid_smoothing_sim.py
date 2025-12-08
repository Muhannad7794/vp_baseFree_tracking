# sigmoid_smoothing_sim.py
#
# Offline simulation of the adaptive smoothing algorithm using the SIGMOID model.
# - Reads raw tracking logs + kinematic derivatives
# - Reads sigmoid_sigma_ranges.json for per-axis parameters (including steepness/midpoint)
# - Replays one take (label) and applies the sigmoid logic:
#     * rolling σ on acceleration over N frames
#     * sigmoid inverse mapping σ -> InterpSpeed
#     * FInterpTo-style smoothing
# - Saves JPG plots comparing raw vs smoothed motion.

import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


POS_AXES: List[str] = ["X_pose", "Y_pose", "Z_pose"]
ROT_AXES: List[str] = ["X_rot", "Y_rot", "Z_rot"]
ALL_AXES: List[str] = POS_AXES + ROT_AXES


def load_config(config_path: str) -> Dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def select_label(
    logs_df: pd.DataFrame, label: str = None, scenario: str = None, take: int = None
) -> str:
    """
    Decide which label (take) to simulate.
    Priority:
      1. Explicit --label
      2. (--scenario, --take) pair
    """
    if label:
        if label not in logs_df["label"].unique():
            raise ValueError(f"Label '{label}' not found in logs.")
        return label

    if scenario is not None and take is not None:
        subset = logs_df[(logs_df["scenario"] == scenario) & (logs_df["take"] == take)]
        labels = subset["label"].unique()
        if len(labels) == 0:
            raise ValueError(f"No label found for scenario='{scenario}', take={take}.")
        if len(labels) > 1:
            raise ValueError(
                f"Multiple labels found for scenario='{scenario}', take={take}: {labels}. "
                "Please specify --label explicitly."
            )
        return labels[0]

    raise ValueError(
        "You must provide either --label or (--scenario and --take) to choose a take."
    )


def f_interp_to(current: float, target: float, dt: float, speed: float) -> float:
    """
    Scalar version of Unreal's FMath::FInterpTo:
      Delta = Target - Current
      DeltaMove = DeltaTime * InterpSpeed
      if |Delta| < SMALL: return Target
      return Current + clamp(DeltaMove, 0, 1) * Delta
    """
    delta = target - current
    if abs(delta) < 1e-6:
        return target

    delta_move = dt * speed
    alpha = max(0.0, min(1.0, delta_move))
    return current + alpha * delta


def run_simulation_for_axis(
    logs_df: pd.DataFrame,
    drv_df: pd.DataFrame,
    cfg: Dict,
    label: str,
    axis: str,
) -> Dict[str, np.ndarray]:
    """
    Core simulation loop for a single label & axis (Sigmoid Version).

    Returns a dict with:
      time, raw, smooth, sigma, speed, dt
    """
    if axis not in ALL_AXES:
        raise ValueError(f"axis must be one of {ALL_AXES}, got {axis}")

    if "axes" not in cfg or axis not in cfg["axes"]:
        raise ValueError(f"Axis '{axis}' not found in config JSON.")

    axis_cfg = cfg["axes"][axis]
    window = int(cfg.get("window", 25))

    # Load ranges and sigmoid params
    min_sigma = float(axis_cfg["min_sigma"])
    max_sigma = float(axis_cfg["max_sigma"])
    min_speed = float(axis_cfg["min_speed"])
    max_speed = float(axis_cfg["max_speed"])

    # These should be present if sigmoid_sigma_model.py generated the config
    midpoint = float(axis_cfg.get("midpoint", (min_sigma + max_sigma) / 2.0))
    steepness = float(axis_cfg.get("steepness", 0.1))

    # Extract this label from both raw and derivatives, sort by time
    raw = logs_df[logs_df["label"] == label].sort_values("time")
    drv = drv_df[drv_df["label"] == label].sort_values("time")

    if raw.empty or drv.empty:
        raise ValueError(f"No data found for label '{label}'")

    # Align on time (just in case)
    merged = pd.merge(
        raw[["time", axis]],
        drv[["time", "dt", f"A_{axis}"]],
        on="time",
        how="inner",
    )

    t = merged["time"].to_numpy(dtype=float)
    dt = merged["dt"].to_numpy(dtype=float)
    pose = merged[axis].to_numpy(dtype=float)
    accel = merged[f"A_{axis}"].to_numpy(dtype=float)

    # Replace NaN dt (first sample) with median dt for that take
    median_dt = np.nanmedian(dt) if np.any(~np.isnan(dt)) else 0.0
    dt = np.where(np.isnan(dt), median_dt, dt)

    n = len(t)
    smooth = np.zeros_like(pose)
    sigma_arr = np.full_like(pose, np.nan, dtype=float)
    speed_arr = np.full_like(pose, np.nan, dtype=float)

    # Initial virtual value = first raw value
    current = pose[0]
    smooth[0] = current

    # Acceleration buffer for rolling std
    buffer: List[float] = []

    min_samples_for_sigma = max(1, window // 2)

    for i in range(n):
        a = accel[i]
        if not np.isnan(a):
            buffer.append(a)
            if len(buffer) > window:
                buffer.pop(0)

        # Compute σ over the buffer once we have enough samples
        if len(buffer) >= min_samples_for_sigma:
            sigma = float(np.std(buffer, ddof=0))
        else:
            sigma = np.nan

        # --- SIGMOID MAPPING LOGIC START ---
        if np.isnan(sigma):
            # Before we have enough history, treat as very stable
            speed = max_speed
        else:
            # 1. Calculate standard logistic curve (low to high)
            # Higher sigma -> Higher result (approaching 1.0)
            logistic_curve = 1 / (1 + np.exp(-steepness * (sigma - midpoint)))

            # 2. Invert mapping
            # We want: Low Sigma (stable) -> High Speed (MaxSpeed)
            #          High Sigma (jitter) -> Low Speed (MinSpeed)
            # So: Speed = Min + (Range * (1 - logistic))
            speed = min_speed + (max_speed - min_speed) * (1 - logistic_curve)
        # --- SIGMOID MAPPING LOGIC END ---

        current = f_interp_to(current, pose[i], dt[i], speed)

        smooth[i] = current
        sigma_arr[i] = sigma
        speed_arr[i] = speed

    return {
        "time": t,
        "dt": dt,
        "raw": pose,
        "smooth": smooth,
        "sigma": sigma_arr,
        "speed": speed_arr,
    }


def jitter_metric(series: np.ndarray, dt: np.ndarray) -> float:
    """
    Simple "jitter" / high-frequency energy metric:
      RMS of first difference per unit time.
    """
    if len(series) < 2:
        return 0.0
    diff = np.diff(series)
    # Approximate using mean dt
    mean_dt = np.mean(dt) if np.any(dt > 0) else 1.0
    vel_est = diff / mean_dt
    return float(np.sqrt(np.mean(vel_est**2)))


def lag_estimate(raw: np.ndarray, smooth: np.ndarray, dt: np.ndarray) -> float:
    """
    Estimate lag (in seconds) between raw and smoothed signals using
    cross-correlation. Positive lag means smoothed signal lags behind raw.
    """
    x = raw - np.mean(raw)
    y = smooth - np.mean(smooth)

    corr = np.correlate(x, y, mode="full")
    lags = np.arange(-len(x) + 1, len(x))

    # Lag at maximum correlation (best alignment)
    best_idx = int(np.argmax(corr))
    best_lag_samples = lags[best_idx]

    mean_dt = np.mean(dt) if np.any(dt > 0) else 1.0
    return float(best_lag_samples * mean_dt)


def make_plots(
    label: str,
    axis: str,
    sim: Dict[str, np.ndarray],
    output_dir: str = "data/sigmoid_plots/smoothing",
) -> None:
    """
    Create JPG plots showing:
      - raw vs smoothed motion over time
      - jitter (high-frequency energy) before vs after
      - difference over time, annotated with lag
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{label}_{axis}_sigmoid.jpg")

    t = sim["time"]
    dt = sim["dt"]
    raw = sim["raw"]
    smooth = sim["smooth"]

    # Metrics
    jitter_raw = jitter_metric(raw, dt)
    jitter_smooth = jitter_metric(smooth, dt)
    jitter_reduction = (jitter_raw / jitter_smooth) if jitter_smooth > 0 else np.inf

    lag_sec = lag_estimate(raw, smooth, dt)

    plt.figure(figsize=(12, 9))

    # 1) Raw vs smoothed motion
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t, raw, label="raw", linewidth=1, alpha=0.7)
    ax1.plot(t, smooth, label="smoothed (sigmoid)", linewidth=1)
    ax1.set_ylabel(axis)
    ax1.set_title(f"{label} – {axis}: raw vs smoothed (Sigmoid Model)")
    ax1.legend()

    # 2) Jitter metric bar plot
    ax2 = plt.subplot(3, 1, 2)
    bars_x = np.arange(2)

    bars_vals = [jitter_raw, jitter_smooth]
    bars_labels = ["raw", "smoothed"]

    ax2.bar(bars_x, bars_vals, color=["tab:red", "tab:green"])
    ax2.set_xticks(bars_x)
    ax2.set_xticklabels(bars_labels)
    ymax = max(jitter_raw, jitter_smooth)
    ax2.set_ylim(0, ymax * 1.2 if ymax > 0 else 1.0)
    ax2.set_ylabel("jitter (RMS of diff/dt)")
    ax2.set_title(
        f"Jitter reduction: {jitter_raw:.3f} → {jitter_smooth:.3f} "
        f"({jitter_reduction:.2f}x lower)"
        if jitter_smooth > 0
        else f"Jitter reduction: {jitter_raw:.3f} → {jitter_smooth:.3f}"
    )
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # 3) Difference over time, with lag annotation
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    diff = smooth - raw
    ax3.plot(t, diff, label="smoothed - raw", linewidth=1)
    ax3.axhline(0.0, color="black", linewidth=0.5)
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("difference")
    ax3.set_title(f"Lag estimate ≈ {lag_sec*1000:.1f} ms")
    ax3.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")  # JPG by extension
    plt.close()

    print(
        f"[sigmoid_smoothing_sim] Saved comparison plot to {out_path}\n"
        f"  Jitter raw     : {jitter_raw:.4f}\n"
        f"  Jitter smoothed: {jitter_smooth:.4f}\n"
        f"  Reduction      : {jitter_reduction:.2f}x\n"
        f"  Lag estimate   : {lag_sec*1000:.1f} ms"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Simulate adaptive smoothing using SIGMOID sigma model "
            "and visualise raw vs smoothed motion."
        )
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="data/processed/tracking_logs.csv",
        help="Path to processed tracking_logs.csv",
    )
    parser.add_argument(
        "--derived",
        type=str,
        default="data/derived/tracking_derivatives.csv",
        help="Path to tracking_derivatives.csv (dt + acceleration)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="data/config/sigmoid_sigma_ranges.json",
        help="Path to sigmoid_sigma_ranges.json",
    )
    parser.add_argument(
        "--axis",
        type=str,
        required=True,
        help="Axis to simulate (one of X_pose, Y_pose, Z_pose, X_rot, Y_rot, Z_rot)",
    )
    parser.add_argument(
        "--label",
        type=str,
        help="Exact label (e.g. StillOnTripod_01) to simulate",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="Scenario name (e.g. StillOnTripod). Used with --take if --label not given.",
    )
    parser.add_argument(
        "--take",
        type=int,
        help="Take number (1-4) to use with --scenario if --label not given.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sigmoid_plots/smoothing",
        help="Directory to save JPG plots",
    )

    args = parser.parse_args()

    logs_df = pd.read_csv(args.logs)
    drv_df = pd.read_csv(args.derived)
    cfg = load_config(args.config)

    label = select_label(
        logs_df, label=args.label, scenario=args.scenario, take=args.take
    )
    print(
        f"[sigmoid_smoothing_sim] Running simulation for label='{label}', axis='{args.axis}'"
    )

    sim = run_simulation_for_axis(logs_df, drv_df, cfg, label, args.axis)

    make_plots(label, args.axis, sim, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
