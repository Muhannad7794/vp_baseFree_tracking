# kinematics.py
import os
import argparse
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Columns in the raw tracking data
POS_AXES: List[str] = ["X_pose", "Y_pose", "Z_pose"]
ROT_AXES: List[str] = ["X_rot", "Y_rot", "Z_rot"]
ALL_AXES: List[str] = POS_AXES + ROT_AXES


def compute_kinematics(
    input_path: str = "data/processed/tracking_logs.csv",
    output_path: str = "data/derived/tracking_derivatives.csv",
) -> pd.DataFrame:
    """
    Compute dt, per-axis velocity and acceleration for each sample.

    Output columns:
      time, label, scenario, take, dt,
      V_X_pose, ..., V_Z_rot,
      A_X_pose, ..., A_Z_rot
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    required_cols = ["time", "label", "scenario", "take"] + ALL_AXES
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in input: {missing}")

    # Ensure deterministic ordering
    df = df.sort_values(["label", "time"]).reset_index(drop=True)

    out_rows = []

    for label, g in df.groupby("label", sort=False):
        g = g.sort_values("time").copy()

        t = g["time"].to_numpy(dtype=float)

        # dt: first sample in each label will be NaN
        dt = np.empty_like(t)
        dt[0] = np.nan
        if len(t) > 1:
            dt[1:] = np.diff(t)

        out = pd.DataFrame(
            {
                "time": g["time"].to_numpy(),
                "label": g["label"].to_numpy(),
                "scenario": g["scenario"].to_numpy(),
                "take": g["take"].to_numpy(),
                "dt": dt,
            }
        )

        # Velocity and acceleration per axis using numerical gradient over time
        for axis in ALL_AXES:
            x = g[axis].to_numpy(dtype=float)

            # Velocity: dx/dt
            v = np.gradient(x, t)
            # Acceleration: dv/dt
            a = np.gradient(v, t)

            out[f"V_{axis}"] = v
            out[f"A_{axis}"] = a

        out_rows.append(out)

    result = pd.concat(out_rows, ignore_index=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"[kinematics] Saved derivatives to {output_path}")
    return result


def plot_axis_timeseries(
    label: str,
    axis: str,
    raw_path: str = "data/processed/tracking_logs.csv",
    derived_path: str = "data/derived/tracking_derivatives.csv",
) -> None:
    """
    Visualise position, velocity and acceleration for a given label & axis.

    axis must be one of: X_pose, Y_pose, Z_pose, X_rot, Y_rot, Z_rot
    """
    if axis not in ALL_AXES:
        raise ValueError(f"axis must be one of {ALL_AXES}, got {axis}")

    raw = pd.read_csv(raw_path)
    drv = pd.read_csv(derived_path)

    raw_l = raw[raw["label"] == label].sort_values("time")
    drv_l = drv[drv["label"] == label].sort_values("time")

    if raw_l.empty or drv_l.empty:
        raise ValueError(f"No data found for label '{label}'")

    # Merge on time to keep alignment robust
    merged = pd.merge(
        raw_l[["time", axis]],
        drv_l[["time", f"V_{axis}", f"A_{axis}"]],
        on="time",
        how="inner",
    )

    t = merged["time"].to_numpy()
    pos = merged[axis].to_numpy()
    vel = merged[f"V_{axis}"].to_numpy()
    acc = merged[f"A_{axis}"].to_numpy()

    plt.figure()
    plt.plot(t, pos, label="position")
    plt.plot(t, vel, label="velocity")
    plt.plot(t, acc, label="acceleration")
    plt.xlabel("time (s)")
    plt.ylabel("value")
    plt.title(f"{label} – {axis}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compute kinematic derivatives.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/tracking_logs.csv",
        help="Path to tracking_logs.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/derived/tracking_derivatives.csv",
        help="Output CSV for derivatives",
    )
    parser.add_argument(
        "--plot-label",
        type=str,
        help="Optional: label to plot (uses data from input & output paths)",
    )
    parser.add_argument(
        "--plot-axis",
        type=str,
        help="Optional: axis to plot (e.g. X_pose, Y_rot). Requires --plot-label.",
    )

    args = parser.parse_args()

    # Always compute derivatives when the script runs
    compute_kinematics(args.input, args.output)

    if args.plot_label and args.plot_axis:
        plot_axis_timeseries(args.plot_label, args.plot_axis, args.input, args.output)


if __name__ == "__main__":
    main()
