import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURATION ---
INPUT_DIR = Path("data/validation/processed")
OUTPUT_PLOT_DIR = Path("data/validation/plots")

AXES_TO_PLOT = ["X_pose", "Y_pose", "Z_pose", "X_rot", "Y_rot", "Z_rot"]


def calculate_jitter(series: np.ndarray, dt_mean: float) -> float:
    """RMS of velocity (first difference)"""
    if len(series) < 2:
        return 0.0
    velocity = np.diff(series) / dt_mean
    return float(np.sqrt(np.mean(velocity**2)))


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
    dt = np.mean(np.diff(t)) if len(t) > 1 else 0.016

    # Create subfolder for this session
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

        # 1. Normalize Offsets (Center them) so we can compare motion
        y_raw_centered = y_raw - np.mean(y_raw)
        y_smooth_centered = y_smooth - np.mean(y_smooth)

        # 2. Metrics
        j_raw = calculate_jitter(y_raw, dt)
        j_smooth = calculate_jitter(y_smooth, dt)
        reduction = j_raw / j_smooth if j_smooth > 0 else 0.0

        # 3. Plotting
        plt.figure(figsize=(10, 8))

        # Top: Path Overlay
        plt.subplot(2, 1, 1)
        plt.plot(
            t, y_raw_centered, label="Raw (Centered)", color="silver", linewidth=1.5
        )
        plt.plot(
            t,
            y_smooth_centered,
            label="Smoothed (Centered)",
            color="#2ecc71",
            linewidth=2.0,
        )
        plt.title(f"{csv_path.stem} | {axis} | Jitter Red: {reduction:.2f}x")
        plt.ylabel("Relative Units")
        plt.legend()
        plt.grid(alpha=0.3)

        # Bottom: Jitter Bar
        plt.subplot(2, 1, 2)
        bars = [j_raw, j_smooth]
        colors = ["#e74c3c", "#27ae60"]
        plt.bar(["Raw Input", "Smoothed"], bars, color=colors, width=0.5)
        plt.ylabel("Noise Level (RMS Velocity)")
        plt.title("Noise Comparison (Lower is Better)")
        plt.grid(axis="y", alpha=0.3)

        # Save
        save_path = session_plot_dir / f"{axis}_validation.jpg"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    print(f"    [OK] Plots saved to {session_plot_dir}")


def main():
    # CLI args (optional)
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
