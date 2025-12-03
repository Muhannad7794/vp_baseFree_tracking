# kinematics.py
import os
import argparse
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

        for axis in ALL_AXES:
            x = g[axis].to_numpy(dtype=float)

            v = np.gradient(x, t)
            a = np.gradient(v, t)

            out[f"V_{axis}"] = v
            out[f"A_{axis}"] = a

        out_rows.append(out)

    result = pd.concat(out_rows, ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"[kinematics] Saved derivatives to {output_path}")
    return result


def plot_axis_timeseries(
    label: str,
    axis: str,
    raw_path: str = "data/processed/tracking_logs.csv",
    derived_path: str = "data/derived/tracking_derivatives.csv",
    output_dir: str = "data/plots/kinematics",
) -> None:
    """
    Visualise position, velocity and acceleration for a given label & axis.
    """
    if axis not in ALL_AXES:
        raise ValueError(f"axis must be one of {ALL_AXES}, got {axis}")

    raw = pd.read_csv(raw_path)
    drv = pd.read_csv(derived_path)

    raw_l = raw[raw["label"] == label].sort_values("time")
    drv_l = drv[drv["label"] == label].sort_values("time")

    if raw_l.empty or drv_l.empty:
        raise ValueError(f"No data found for label '{label}'")

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

    plt.figure(figsize=(10, 6))
    plt.plot(t, pos, label="position")
    plt.plot(t, vel, label="velocity")
    plt.plot(t, acc, label="acceleration")
    plt.xlabel("time (s)")
    plt.ylabel("value")
    plt.title(f"{label} – {axis}")
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fname = f"{label}_{axis}.jpg"
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[kinematics] Saved plot to {out_path}")


def plot_axis_by_scenario(
    scenario: str,
    axis: str,
    raw_path: str = "data/processed/tracking_logs.csv",
    derived_path: str = "data/derived/tracking_derivatives.csv",
    output_dir: str = "data/plots/kinematics_scenarios",
) -> None:
    """
    For a given scenario and axis, overlay all takes:

    - 3 subplots: position, velocity, acceleration vs time
    - each take (label) is a separate line
    - time is shifted so each take starts at t=0

    Saves a jpg to output_dir.
    """
    if axis not in ALL_AXES:
        raise ValueError(f"axis must be one of {ALL_AXES}, got {axis}")

    raw = pd.read_csv(raw_path)
    drv = pd.read_csv(derived_path)

    raw_s = raw[raw["scenario"] == scenario]
    drv_s = drv[drv["scenario"] == scenario]

    if raw_s.empty or drv_s.empty:
        raise ValueError(f"No data found for scenario '{scenario}'")

    os.makedirs(output_dir, exist_ok=True)
    fname = f"{scenario}_{axis}.jpg"
    out_path = os.path.join(output_dir, fname)

    plt.figure(figsize=(12, 9))

    ax_pos = plt.subplot(3, 1, 1)
    ax_vel = plt.subplot(3, 1, 2)
    ax_acc = plt.subplot(3, 1, 3)

    for label, g_raw in raw_s.groupby("label"):
        g_raw = g_raw.sort_values("time")
        g_drv = drv_s[drv_s["label"] == label].sort_values("time")

        merged = pd.merge(
            g_raw[["time", axis]],
            g_drv[["time", f"V_{axis}", f"A_{axis}"]],
            on="time",
            how="inner",
        )
        if merged.empty:
            continue

        t = merged["time"].to_numpy()
        t0 = t - t[0]  # align each take to t=0
        pos = merged[axis].to_numpy()
        vel = merged[f"V_{axis}"].to_numpy()
        acc = merged[f"A_{axis}"].to_numpy()

        ax_pos.plot(t0, pos, label=label)
        ax_vel.plot(t0, vel, label=label)
        ax_acc.plot(t0, acc, label=label)

    ax_pos.set_title(f"{scenario} – {axis}")
    ax_acc.set_xlabel("time since start of take (s)")
    ax_pos.set_ylabel("position")
    ax_vel.set_ylabel("velocity")
    ax_acc.set_ylabel("acceleration")

    ax_pos.legend(fontsize="small", ncol=2)
    plt.tight_layout()

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[kinematics] Saved scenario plot to {out_path}")


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
        help="Label to plot (position/velocity/acceleration over time)",
    )
    parser.add_argument(
        "--plot-scenario",
        type=str,
        help="Scenario to plot (overlays all takes) – used with --plot-axis",
    )
    parser.add_argument(
        "--plot-axis",
        type=str,
        help="Axis to plot (e.g. X_pose, Y_rot)",
    )

    args = parser.parse_args()

    compute_kinematics(args.input, args.output)

    if args.plot_label and args.plot_axis:
        plot_axis_timeseries(
            label=args.plot_label,
            axis=args.plot_axis,
            raw_path=args.input,
            derived_path=args.output,
        )

    if args.plot_scenario and args.plot_axis:
        plot_axis_by_scenario(
            scenario=args.plot_scenario,
            axis=args.plot_axis,
            raw_path=args.input,
            derived_path=args.output,
        )


if __name__ == "__main__":
    main()
