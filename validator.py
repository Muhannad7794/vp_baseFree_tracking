# validator.py
"""Validation script over Unreal Engine runtime sessions.
It validates the smoothing effect on recorded motion data by comparing
the raw and smoothed signals. It generates plots for visual inspection
and computes metrics like jitter reduction and lag estimation.
Additionally, the script allows for triggering the simualtion scripts
of other models to compare their results to the result of the chosen
model during runtime.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal

# --- CONFIGURATION ---
INPUT_DIR = Path("data/validation/processed")
OUTPUT_PLOT_DIR = Path("data/validation/plots")

AXES_TO_PLOT = ["X_pose", "Y_pose", "Z_pose", "X_rot", "Y_rot", "Z_rot"]


def calculate_jitter(series: np.ndarray, dt: np.ndarray) -> float:
    """RMS of velocity (first difference)"""
    if len(series) < 2:
        return 0.0
    velocity = np.diff(series) / dt
    return float(np.sqrt(np.mean(velocity**2)))


def calculate_lag(t: np.ndarray, raw: np.ndarray, smooth: np.ndarray) -> float:
    """Estimates time lag using Cross-Correlation."""
    if len(raw) != len(smooth):
        return 0.0

    # Normalize for correlation calculation
    raw_norm = (raw - np.mean(raw)) / (np.std(raw) + 1e-9)
    smooth_norm = (smooth - np.mean(smooth)) / (np.std(smooth) + 1e-9)

    correlation = signal.correlate(smooth_norm, raw_norm, mode="full")
    lags = signal.correlation_lags(raw_norm.size, smooth_norm.size, mode="full")

    lag_index = lags[np.argmax(correlation)]
    dt = np.mean(np.diff(t)) if np.any(np.diff(t)) else 1.0
    return lag_index * dt


def process_file(csv_path):
    print(f"--> Validating: {csv_path.name}")
    try:
        df = pd.read_csv(csv_path).sort_values("Time")
    except Exception as e:
        print(f"    Error reading file: {e}")
        return

    # Normalize Time
    t = df["Time"].values
    t = t - t[0]
    dt = np.diff(t)

    # Safety: Fix zero or negative time deltas (logging artifacts)
    dt = np.maximum(dt, 1e-3)

    # Create subfolder
    session_plot_dir = OUTPUT_PLOT_DIR / csv_path.stem
    session_plot_dir.mkdir(parents=True, exist_ok=True)

    for axis in AXES_TO_PLOT:
        col_raw = f"{axis}_raw"
        col_smooth = f"{axis}_smoothed"

        if col_raw not in df.columns:
            continue

        # Extract Data
        y_raw = df[col_raw].values
        y_smooth = df[col_smooth].values

        # Align Smoothed signal to account for UE offset function and start difference
        start_offset = y_raw[0] - y_smooth[0]
        y_smooth_aligned = y_smooth + start_offset

        # Metrics
        j_raw = calculate_jitter(y_raw, dt)
        j_smooth = calculate_jitter(y_smooth_aligned, dt)
        j_reduction = j_raw / j_smooth if j_smooth > 0 else 0.0
        lag_seconds = calculate_lag(t, y_raw, y_smooth_aligned)

        # --- PLOTTING ---
        plt.figure(figsize=(12, 9))

        # 1. Raw vs smoothed motion
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(t, y_raw, label="Raw", color="tab:red", linewidth=1)
        ax1.plot(t, y_smooth_aligned, label="Smoothed", color="tab:green", linewidth=1)
        ax1.set_title(f"{csv_path.stem} - {axis}: Raw vs Smoothed")
        ax1.set_ylabel("Amplitude")
        ax1.legend()

        # 2. Jitter Metric bar plot
        ax2 = plt.subplot(3, 1, 2)
        bars_x = np.arange(2)

        bar_vals = [j_raw, j_smooth]
        bar_labels = ["Raw", "Smoothed"]

        ax2.bar(bars_x, bar_vals, color=["tab:red", "tab:green"])
        ax2.set_xticks(bars_x)
        ax2.set_xticklabels(bar_labels)

        ymax = max(j_raw, j_smooth)
        ax2.set_ylim(0, ymax * 1.2 if ymax > 0 else 1.0)
        ax2.set_ylabel("jitter (RMS of diff/dt)")
        ax2.set_title(
            f"Jitter Reduction: {j_raw:.3f} → {j_smooth:.3f}"
            f"({j_reduction:.2f}x lower)"
            if j_smooth > 0
            else f"Jitter Reduction: {j_raw:.3f} → {j_smooth:.3f}"
        )
        ax2.grid(axis="y", linestyle="--", alpha=0.4)

        # 3. Lag / Delta
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        delta = y_smooth_aligned - y_raw
        # Plot the delta line
        ax3.plot(t, delta, label="Tracking lag (Delta)", color="#0b9ea8", linewidth=1)
        # Plot the baseLine (original time)
        ax3.axhline(
            0, label="original time", color="black", linestyle="--", linewidth=1
        )
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("difference")
        ax3.set_title(f"Lag estimate ≈ {lag_seconds*1000:.1f} ms")
        ax3.fill_between(t, 0, delta, color="gray", alpha=0.2)
        ax3.legend()

        plt.tight_layout()
        save_path = session_plot_dir / f"{axis}_validation.jpg"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  [OK] Plots saved to {session_plot_dir}")


def main():
    # CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Specific file to process")
    args = parser.parse_args()

    if args.file:
        # Process single file requested by CLI
        files = [Path(args.file)]
    else:
        # Auto-discover all CSVs
        files = list(INPUT_DIR.glob("*_parsed.csv"))

    if not files:
        print(f"No parsed CSV files found in {INPUT_DIR}. Run parser first.")
        return

    for f in files:
        process_file(f)


if __name__ == "__main__":
    main()
