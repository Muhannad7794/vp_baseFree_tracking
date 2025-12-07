# piecewise_smoothing_sim.py
"""
piecewise_smoothing_sim.py

Offline simulation of the piecewise σ→InterpSpeed smoothing model.

For a given label and axis, it:
- Replays the raw pose from tracking_logs.csv,
- Recomputes rolling σ from accelerations,
- Maps σ to InterpSpeed using piecewise sigma breaks & speed levels,
- Applies a scalar FInterpTo step each frame,
- Plots raw vs smoothed, jitter comparison, and difference/lag.

Plots are written under data/piecewise_plots/smoothing/.
"""

import argparse
import json
import math
import os
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---- Helpers ----------------------------------------------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def piecewise_speed_from_sigma(
    sigma: float,
    breaks: List[float],
    speeds: List[float],
) -> float:
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


def interp_to(current: float, target: float, dt: float, speed: float) -> float:
    """
    Discrete FInterpTo-style update:

      Alpha = clamp(dt * Speed, 0, 1)
      New   = Current + (Target - Current) * Alpha
    """
    if dt <= 0.0 or speed <= 0.0:
        return target
    alpha = dt * speed
    if alpha > 1.0:
        alpha = 1.0
    return current + (target - current) * alpha


def compute_jitter(signal: np.ndarray) -> float:
    """
    Simple jitter metric: RMS of frame-to-frame differences
    (approximate high-frequency energy).
    """
    if len(signal) < 2:
        return 0.0
    diff = np.diff(signal)
    return float(np.sqrt(np.mean(diff * diff)))


def estimate_lag(raw: np.ndarray, smoothed: np.ndarray, dt_mean: float) -> float:
    """
    Estimate lag by cross-correlation peak between demeaned signals.
    Returns lag in milliseconds (smoothed vs raw).
    """
    if len(raw) != len(smoothed) or len(raw) < 2:
        return 0.0

    r = raw - np.mean(raw)
    s = smoothed - np.mean(smoothed)
    corr = np.correlate(s, r, mode="full")
    lag_idx = int(np.argmax(corr) - (len(raw) - 1))
    lag_sec = lag_idx * dt_mean
    return float(lag_sec * 1000.0)


# ---- Main -------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Simulate piecewise σ→InterpSpeed smoothing on a single label and axis."
    )
    parser.add_argument(
        "--logs",
        default="data/processed/tracking_logs.csv",
        help="CSV with raw poses (default: data/processed/tracking_logs.csv)",
    )
    parser.add_argument(
        "--derivatives",
        default="data/derived/tracking_derivatives.csv",
        help="CSV with dt and accelerations (default: data/derived/tracking_derivatives.csv)",
    )
    parser.add_argument(
        "--config",
        default="data/config/piecewise_sigma_ranges.json",
        help="Piecewise model config JSON (default: data/config/piecewise_sigma_ranges.json)",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Label (scenario_take) to simulate, e.g. fast_tilt_tripod_02",
    )
    parser.add_argument(
        "--axis",
        required=True,
        help="Axis to simulate, e.g. X_pose, Y_pose, Z_pose, X_rot, Y_rot, Z_rot",
    )
    parser.add_argument(
        "--output-dir",
        default="data/piecewise_plots/smoothing",
        help="Directory to save plots (default: data/piecewise_plots/smoothing)",
    )

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    print("[piecewise_smoothing_sim] Loading logs:", args.logs)
    df_logs = pd.read_csv(args.logs)

    print("[piecewise_smoothing_sim] Loading derivatives:", args.derivatives)
    df_der = pd.read_csv(args.derivatives)

    # Filter by label and sort by time
    df_logs = df_logs[df_logs["label"] == args.label].sort_values("time")
    df_der = df_der[df_der["label"] == args.label].sort_values("time")

    if df_logs.empty or df_der.empty:
        raise ValueError(f"No data found for label {args.label}")

    # Join on time (assume unique times per label)
    df = pd.merge(
        df_logs[["time", "label", args.axis]],
        df_der[["time", "label", "dt", f"A_{args.axis}"]],
        on=["time", "label"],
        how="inner",
    ).copy()

    if df.empty:
        raise ValueError(
            f"After merge, no rows for label={args.label}, axis={args.axis}. "
            f"Check that both CSVs contain matching time/label."
        )

    # Load model config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    axis_cfg = cfg["axes"].get(args.axis)
    if axis_cfg is None:
        raise ValueError(f"Axis {args.axis} not found in config {args.config}")

    sigma_breaks = axis_cfg["sigma_breaks"]
    speed_levels = axis_cfg["speed_levels"]
    window = int(cfg.get("window_size", 25))

    print(
        f"[piecewise_smoothing_sim] Axis={args.axis}, window={window}, "
        f"{len(sigma_breaks)} sigma breaks"
    )

    # Simulation buffers
    accel_buffer: List[float] = []
    raw_vals: List[float] = []
    smooth_vals: List[float] = []
    sigmas: List[float] = []
    speeds: List[float] = []

    pose_col = args.axis
    accel_col = f"A_{args.axis}"

    # Initialise smoothed value at first pose
    current = float(df.iloc[0][pose_col])

    for _, row in df.iterrows():
        target = float(row[pose_col])

        dt = float(row["dt"])
        # Protect against NaN / inf on the first sample
        if not math.isfinite(dt):
            dt = 0.0

        a = float(row[accel_col])
        # Optionally ignore non-finite accelerations (defensive)
        if math.isfinite(a):
            accel_buffer.append(a)
            if len(accel_buffer) > window:
                accel_buffer.pop(0)

        if len(accel_buffer) > 1:
            sigma = float(np.std(accel_buffer))
        else:
            sigma = 0.0

        speed = piecewise_speed_from_sigma(sigma, sigma_breaks, speed_levels)
        current = interp_to(current, target, dt, speed)

        raw_vals.append(target)
        smooth_vals.append(current)
        sigmas.append(sigma)
        speeds.append(speed)

    raw_arr = np.array(raw_vals)
    smooth_arr = np.array(smooth_vals)
    dt_mean = float(df["dt"].mean())

    # Metrics
    jitter_raw = compute_jitter(raw_arr)
    jitter_smooth = compute_jitter(smooth_arr)
    lag_ms = estimate_lag(raw_arr, smooth_arr, dt_mean)
    jitter_reduction = jitter_raw / jitter_smooth if jitter_smooth > 0 else np.inf

    print(
        f"[piecewise_smoothing_sim] Jitter: {jitter_raw:.3f} → {jitter_smooth:.3f} "
        f"({jitter_reduction:.2f}x lower)"
        if jitter_smooth > 0
        else f"[piecewise_smoothing_sim] Jitter: {jitter_raw:.3f} → {jitter_smooth:.3f}"
    )
    print(f"[piecewise_smoothing_sim] Lag estimate ≈ {lag_ms:.1f} ms")

    # Plot
    t = df["time"].to_numpy()

    plt.figure(figsize=(12, 9))

    # 1) Raw vs smoothed
    ax0 = plt.subplot(3, 1, 1)
    ax0.plot(t, raw_arr, label="raw", linewidth=1)
    ax0.plot(t, smooth_arr, label="smoothed", linewidth=1)
    ax0.set_ylabel(args.axis)
    ax0.set_title(f"{args.label} – {args.axis}: raw vs smoothed")
    ax0.legend()

    # 2) Jitter comparison
    ax1 = plt.subplot(3, 1, 2)
    x_pos = [0, 1]
    width = 0.6

    ax1.bar(x_pos[0], jitter_raw, width=width, color="tab:red", label="raw")
    ax1.bar(x_pos[1], jitter_smooth, width=width, color="tab:green", label="smoothed")

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(["raw", "smoothed"])
    ax1.set_xlim(-0.5, 1.5)

    ymax = max(jitter_raw, jitter_smooth)
    # Add some headroom so bars don't touch the top
    ax1.set_ylim(0, ymax * 1.2 if ymax > 0 else 1.0)

    ax1.set_ylabel("jitter (RMS of diff/dt)")
    ax1.set_title(
        f"Jitter reduction: {jitter_raw:.3f} → {jitter_smooth:.3f} "
        f"({jitter_reduction:.2f}x lower)"
        if jitter_smooth > 0
        else f"Jitter reduction: {jitter_raw:.3f} → {jitter_smooth:.3f}"
    )
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # 3) Difference over time
    ax2 = plt.subplot(3, 1, 3)
    diff = smooth_arr - raw_arr
    ax2.plot(t, diff, linewidth=1)
    ax2.axhline(0.0, color="black", linewidth=0.5)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("difference")
    ax2.set_title(f"Lag estimate ≈ {lag_ms:.1f} ms")

    plt.tight_layout()

    filename = f"{args.label}_{args.axis}_piecewise.jpg"
    out_path = os.path.join(args.output_dir, filename)
    plt.savefig(out_path, dpi=150)
    plt.close()

    print("[piecewise_smoothing_sim] Plot saved to", out_path)


if __name__ == "__main__":
    main()
