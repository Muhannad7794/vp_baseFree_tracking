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
from typing import Dict, List
from scipy import signal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POS_AXES: List[str] = ["X_pose", "Y_pose", "Z_pose"]
ROT_AXES: List[str] = ["X_rot", "Y_rot", "Z_rot"]
ALL_AXES: List[str] = POS_AXES + ROT_AXES


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def piecewise_speed_from_sigma(
    sigma: float,
    breaks: List[float],
    speeds: List[float],
) -> float:
    """Piecewise-linear σ→InterpSpeed mapping."""
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


def f_interp_to(current: float, target: float, dt: float, speed: float) -> float:
    """
    Scalar FInterpTo-style update:

      Delta = Target - Current
      DeltaMove = dt * speed
      Alpha = clamp(DeltaMove, 0, 1)
      New = Current + Alpha * Delta
    """
    delta = target - current
    if abs(delta) < 1e-6:
        return target

    if dt <= 0.0 or speed <= 0.0:
        return target

    delta_move = dt * speed
    alpha = max(0.0, min(1.0, delta_move))
    return current + alpha * delta


def jitter_metric(series: np.ndarray, dt: np.ndarray) -> float:
    """
    Jitter / high-frequency energy metric:

      RMS of first difference per unit time.
    """
    if len(series) < 2:
        return 0.0

    diff = np.diff(series)

    # Use mean dt as an approximation for per-step spacing.
    valid_dt = dt[np.isfinite(dt) & (dt > 0)]
    mean_dt = float(np.mean(valid_dt)) if valid_dt.size > 0 else 1.0

    vel_est = diff / mean_dt
    return float(np.sqrt(np.mean(vel_est**2)))


def lag_estimate(raw: np.ndarray, smooth: np.ndarray, dt: np.ndarray) -> float:
    """
    Estimate lag (in seconds) between raw and smoothed signals using
    cross-correlation. Positive lag means smoothed lags behind raw.
    """
    if len(raw) != len(smooth) or len(raw) < 2:
        return 0.0

    # Normalize for correlation calculation
    x = (raw - np.mean(raw)) / (np.std(raw) + 1e-9)
    y = (smooth - np.mean(smooth)) / (np.std(smooth) + 1e-9)

    corr = signal.correlate(y, x, mode="full")
    lags = signal.correlation_lags(len(x), len(y), mode="full")

    # Lag at maximum correlation (for best alignment)
    lag_index = lags[np.argmax(corr)]
    mean_dt = np.mean(dt) if np.any(dt > 0) else 1.0
    return lag_index * mean_dt


def make_piecewise_plots(
    label: str,
    axis: str,
    sim: Dict[str, np.ndarray],
    output_dir: str = "data/piecewise_plots/smoothing",
) -> None:
    """
    Create JPG plots showing:
      - raw vs smoothed motion over time
      - jitter before vs after
      - difference over time, annotated with lag

    The layout and scaling follow the linear model for easier comparison.
    """
    ensure_dir(output_dir)
    out_path = os.path.join(output_dir, f"{label}_{axis}_piecewise.jpg")

    t = sim["time"]
    dt = sim["dt"]
    raw = sim["raw"]
    smooth = sim["smooth"]

    jitter_raw = jitter_metric(raw, dt)
    jitter_smooth = jitter_metric(smooth, dt)
    jitter_reduction = (jitter_raw / jitter_smooth) if jitter_smooth > 0 else np.inf
    lag_sec = lag_estimate(raw, smooth, dt)

    plt.figure(figsize=(12, 9))

    # 1) Raw vs smoothed
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t, raw, label="raw", linewidth=1)
    ax1.plot(t, smooth, label="smoothed", linewidth=1)
    ax1.set_ylabel(axis)
    ax1.set_title(f"{label} – {axis}: raw vs smoothed (piecewise)")
    ax1.legend()

    # 2) Jitter comparison (bar plot)
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
    if jitter_smooth > 0:
        ax2.set_title(
            f"Jitter reduction: {jitter_raw:.3f} → {jitter_smooth:.3f} "
            f"({jitter_reduction:.2f}x lower)"
        )
    else:
        ax2.set_title(f"Jitter reduction: {jitter_raw:.3f} → {jitter_smooth:.3f}")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # 3) Difference over time
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    diff = smooth - raw
    # Plot the delta line
    ax3.plot(t, diff, label="Tracking lag (Delta)", color="#0b9ea8", linewidth=1)
    # Plot the baseLine (original time)
    ax3.axhline(0, label="original time", color="black", linestyle="--", linewidth=1)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("difference")
    ax3.set_title(f"Lag estimate ≈ {lag_sec*1000:.1f} ms")
    ax3.fill_between(t, 0, diff, color="gray", alpha=0.2)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(
        f"  [piecewise_smoothing_sim] Saved comparison plot to {out_path}\n"
        f"  Jitter raw     : {jitter_raw:.4f}\n"
        f"  Jitter smoothed: {jitter_smooth:.4f}\n"
        f"  Reduction      : {jitter_reduction:.2f}x\n"
        f"  Lag estimate   : {lag_sec*1000:.1f} ms"
    )


# ---------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------


def run_piecewise_simulation_for_axis(
    logs_df: pd.DataFrame,
    drv_df: pd.DataFrame,
    cfg: Dict,
    label: str,
    axis: str,
) -> Dict[str, np.ndarray]:
    """
    Run the piecewise smoothing simulation for a single label and axis.

    Returns a dictionary with time, dt, raw, smooth, sigma, speed.
    """
    if axis not in ALL_AXES:
        raise ValueError(f"axis must be one of {ALL_AXES}, got {axis}")

    axes_cfg = cfg.get("axes", {})
    axis_cfg = axes_cfg.get(axis)
    if axis_cfg is None:
        raise ValueError(f"Axis '{axis}' not found in piecewise config JSON.")

    sigma_breaks = axis_cfg["sigma_breaks"]
    speed_levels = axis_cfg["speed_levels"]
    max_speed_fallback = float(axis_cfg.get("max_speed", 40.0))

    window = int(cfg.get("window_size", 25))
    min_samples_for_sigma = max(1, window // 2)

    # Extract this label from both raw and derivatives, sort by time.
    raw = logs_df[logs_df["label"] == label].sort_values("time")
    drv = drv_df[drv_df["label"] == label].sort_values("time")

    if raw.empty or drv.empty:
        raise ValueError(f"No data found for label '{label}'")

    merged = pd.merge(
        raw[["time", "label", axis]],
        drv[["time", "label", "dt", f"A_{axis}"]],
        on=["time", "label"],
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            f"After merge, no rows for label='{label}', axis='{axis}'. "
            "Check that both CSVs contain matching time/label."
        )

    t = merged["time"].to_numpy(dtype=float)
    dt = merged["dt"].to_numpy(dtype=float)
    pose = merged[axis].to_numpy(dtype=float)
    accel = merged[f"A_{axis}"].to_numpy(dtype=float)

    # Replace NaN dt (typically first sample) with median dt.
    valid_dt = dt[np.isfinite(dt)]
    median_dt = float(np.median(valid_dt)) if valid_dt.size > 0 else 0.0
    dt = np.where(np.isfinite(dt), dt, median_dt)

    n = len(t)
    smooth = np.zeros_like(pose)
    sigma_arr = np.full_like(pose, np.nan, dtype=float)
    speed_arr = np.full_like(pose, np.nan, dtype=float)

    # Initial state: start virtual camera at first valid pose sample.
    first_pose = pose[0]
    if not np.isfinite(first_pose):
        first_pose = 0.0
    current = float(first_pose)
    smooth[0] = current
    if not np.isfinite(pose[0]):
        pose[0] = current

    accel_buffer: List[float] = []

    for i in range(n):
        target = pose[i]
        if not np.isfinite(target):
            # Replace invalid pose samples with the last smoothed value.
            target = current

        a = accel[i]
        if math.isfinite(a):
            accel_buffer.append(a)
            if len(accel_buffer) > window:
                accel_buffer.pop(0)

        if len(accel_buffer) >= min_samples_for_sigma:
            sigma = float(np.std(accel_buffer, ddof=0))
        else:
            sigma = np.nan

        speed = piecewise_speed_from_sigma(sigma, sigma_breaks, speed_levels)
        if not np.isfinite(speed):
            speed = max_speed_fallback

        current = f_interp_to(current, target, dt[i], speed)

        smooth[i] = current
        sigma_arr[i] = sigma
        speed_arr[i] = speed

    # Remove any remaining NaNs before metric computation.
    mask = np.isfinite(pose) & np.isfinite(smooth) & np.isfinite(dt)
    t = t[mask]
    dt = dt[mask]
    pose = pose[mask]
    smooth = smooth[mask]
    sigma_arr = sigma_arr[mask]
    speed_arr = speed_arr[mask]

    return {
        "time": t,
        "dt": dt,
        "raw": pose,
        "smooth": smooth,
        "sigma": sigma_arr,
        "speed": speed_arr,
    }


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate piecewise σ→InterpSpeed smoothing on a single label and axis."
        )
    )
    parser.add_argument(
        "--logs",
        default="data/processed/tracking_logs.csv",
        help="CSV with raw poses (default: data/processed/tracking_logs.csv)",
    )
    parser.add_argument(
        "--derived",
        default="data/derived/tracking_derivatives.csv",
        help="CSV with dt and accelerations "
        "(default: data/derived/tracking_derivatives.csv)",
    )
    parser.add_argument(
        "--config",
        default="data/config/piecewise_sigma_ranges.json",
        help="Piecewise model config JSON "
        "(default: data/config/piecewise_sigma_ranges.json)",
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
    logs_df = pd.read_csv(args.logs)

    print("[piecewise_smoothing_sim] Loading derivatives:", args.derived)
    drv_df = pd.read_csv(args.derived)

    if args.label not in logs_df["label"].unique():
        raise ValueError(f"Label '{args.label}' not found in logs CSV.")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    print(
        f"[piecewise_smoothing_sim] Running simulation for "
        f"label='{args.label}', axis='{args.axis}'"
    )

    sim = run_piecewise_simulation_for_axis(
        logs_df=logs_df,
        drv_df=drv_df,
        cfg=cfg,
        label=args.label,
        axis=args.axis,
    )

    make_piecewise_plots(args.label, args.axis, sim, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
