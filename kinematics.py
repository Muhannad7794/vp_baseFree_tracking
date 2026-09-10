# kinematics.py
"""
Kinematic derivative computation for the vp_baseFree_tracking pipeline.

Reads tracking_logs.csv and produces tracking_derivatives.csv with per-axis
velocity (V_*) and acceleration (A_*) columns, computed using a causal
backward-difference scheme.

Derivative formulae
--------------------
For a signal p at time index t with time step dt_t = t_t - t_(t-1):

    V_i(t) = ( p_i(t) - p_i(t-1) ) / dt_t
    A_i(t) = ( V_i(t) - V_i(t-1) ) / dt_t

The first sample of each label group has no predecessor within that group,
so V and A are undefined and stored as NaN. The second sample has a defined
velocity but no previous velocity, so A is also NaN there. All downstream
model scripts handle leading NaN values via dropna() and rolling windows
with min_periods, so this propagates cleanly through the pipeline.

Causality requirement
----------------------
The backward-difference scheme is causal: the estimate at time t depends
only on the current and previous sample. This is a hard requirement for any
derivative formula used in this pipeline, because the downstream runtime
models — both the FInterpTo-based sigma models and the One Euro filter —
compute velocity and acceleration the same way, one sample at a time, with
no access to future data. Offline and runtime computations must use the same
formula to produce numerically consistent results. A non-causal scheme would
produce offline validation metrics that cannot be reproduced at runtime.

Duplicate timestamp guard
--------------------------
Raw log files occasionally contain consecutive samples with identical or
near-identical timestamps (logging artifacts). A time step of zero produces
a division-by-zero in both the velocity and acceleration expressions. All
computed dt values are clamped to a minimum of MIN_DT (1e-4 s) before any
division is performed.
"""

import os
import argparse
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

POS_AXES: List[str] = ["X_pose", "Y_pose", "Z_pose"]
ROT_AXES: List[str] = ["X_rot", "Y_rot", "Z_rot"]
ALL_AXES: List[str] = POS_AXES + ROT_AXES

# Minimum time step guard — prevents division by zero on duplicate timestamps.
# Matches the guard used in parse_validation.py for consistency across the pipeline.
MIN_DT: float = 1e-4  # seconds


def compute_kinematics(
    input_path: str = "data/processed/tracking_logs.csv",
    output_path: str = "data/derived/tracking_derivatives.csv",
) -> pd.DataFrame:
    """
    Compute dt, per-axis velocity and acceleration for each label group.

    Each label (recording take) is processed independently so that the
    first-sample NaN boundary falls at the correct position and state from
    one take never bleeds into the next.

    Output columns
    --------------
    time, label, scenario, take, dt,
    V_X_pose, V_Y_pose, V_Z_pose, V_X_rot, V_Y_rot, V_Z_rot,
    A_X_pose, A_Y_pose, A_Z_pose, A_X_rot, A_Y_rot, A_Z_rot

    The first row of every label group has dt = NaN, V_* = NaN, A_* = NaN.
    The second row has V_* defined but A_* = NaN (no previous velocity).
    All subsequent rows have all columns defined.

    Parameters
    ----------
    input_path : str
        Path to tracking_logs.csv.
    output_path : str
        Destination path for the output CSV.

    Returns
    -------
    pd.DataFrame
        The computed derivatives DataFrame (also written to output_path).
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
        n = len(g)

        t = g["time"].to_numpy(dtype=float)

        # dt[0] is undefined — no previous sample exists within this label.
        dt = np.empty(n)
        dt[0] = np.nan
        if n > 1:
            raw_dt = np.diff(t)
            # Guard against duplicate or reversed timestamps.
            raw_dt = np.where(raw_dt < MIN_DT, MIN_DT, raw_dt)
            dt[1:] = raw_dt

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

            # --- Velocity: causal backward difference ---
            # V[0] is undefined; V[t] = (x[t] - x[t-1]) / dt[t]
            v = np.empty(n)
            v[0] = np.nan
            if n > 1:
                v[1:] = np.diff(x) / dt[1:]

            # --- Acceleration: causal backward difference on velocity ---
            # A[0] and A[1] are undefined; A[t] = (V[t] - V[t-1]) / dt[t]
            a = np.empty(n)
            a[0] = np.nan
            a[1] = np.nan
            if n > 2:
                a[2:] = np.diff(v[1:]) / dt[2:]

            out[f"V_{axis}"] = v
            out[f"A_{axis}"] = a

        out_rows.append(out)

    result = pd.concat(out_rows, ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"[kinematics] Saved derivatives to {output_path}")
    return result


# ---------------------------------------------------------------------------
# Plotting helpers — unchanged in interface from the previous version.
# The plots now reflect backward-difference velocities and accelerations.
# ---------------------------------------------------------------------------


def plot_axis_timeseries(
    label: str,
    axis: str,
    raw_path: str = "data/processed/tracking_logs.csv",
    derived_path: str = "data/derived/tracking_derivatives.csv",
    output_dir: str = "data/plots/kinematics",
) -> None:
    """
    Plot position, velocity, and acceleration over time for a single label and axis.

    Produces a three-panel figure saved as a JPG under output_dir.

    Parameters
    ----------
    label : str
        Recording label to plot (e.g. "handheld_full_nav_01").
    axis : str
        Pose axis to plot (one of ALL_AXES).
    raw_path : str
        Path to tracking_logs.csv.
    derived_path : str
        Path to tracking_derivatives.csv.
    output_dir : str
        Directory where the JPG is written.
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

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t, pos, linewidth=1)
    axes[0].set_ylabel("position")
    axes[0].set_title(f"{label} – {axis}")

    axes[1].plot(t, vel, linewidth=1, color="steelblue")
    axes[1].set_ylabel("velocity (backward diff)")

    axes[2].plot(t, acc, linewidth=1, color="tomato")
    axes[2].set_ylabel("acceleration (backward diff)")
    axes[2].set_xlabel("time (s)")

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{label}_{axis}.jpg")
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
    For a given scenario and axis, overlay all takes in a three-panel figure:
    position, velocity, and acceleration vs time-since-start-of-take.

    Each label (take) is drawn as a separate line. Time is shifted so all
    takes start at t=0, enabling direct comparison of motion profiles.

    Parameters
    ----------
    scenario : str
        Scenario name (e.g. "controlled_handheld_pan").
    axis : str
        Pose axis to plot (one of ALL_AXES).
    raw_path : str
        Path to tracking_logs.csv.
    derived_path : str
        Path to tracking_derivatives.csv.
    output_dir : str
        Directory where the JPG is written.
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
    out_path = os.path.join(output_dir, f"{scenario}_{axis}.jpg")

    fig, axes_panels = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax_pos, ax_vel, ax_acc = axes_panels

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
        t0 = t - t[0]
        pos = merged[axis].to_numpy()
        vel = merged[f"V_{axis}"].to_numpy()
        acc = merged[f"A_{axis}"].to_numpy()

        ax_pos.plot(t0, pos, label=label)
        ax_vel.plot(t0, vel, label=label)
        ax_acc.plot(t0, acc, label=label)

    ax_pos.set_title(f"{scenario} – {axis}")
    ax_pos.set_ylabel("position")
    ax_vel.set_ylabel("velocity (backward diff)")
    ax_acc.set_ylabel("acceleration (backward diff)")
    ax_acc.set_xlabel("time since start of take (s)")

    ax_pos.legend(fontsize="small", ncol=2)
    plt.tight_layout()

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[kinematics] Saved scenario plot to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute causal backward-difference kinematic derivatives "
            "(velocity and acceleration) from tracking_logs.csv."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/tracking_logs.csv",
        help="Path to tracking_logs.csv.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/derived/tracking_derivatives.csv",
        help="Destination path for tracking_derivatives.csv.",
    )
    parser.add_argument(
        "--plot-label",
        type=str,
        help="Label to plot (position / velocity / acceleration over time).",
    )
    parser.add_argument(
        "--plot-scenario",
        type=str,
        help="Scenario to plot (overlays all takes). Used with --plot-axis.",
    )
    parser.add_argument(
        "--plot-axis",
        type=str,
        help="Axis to plot (e.g. X_pose, Y_rot).",
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
