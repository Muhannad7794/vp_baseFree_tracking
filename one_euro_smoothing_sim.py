# one_euro_smoothing_sim.py
"""
Offline simulation of the One Euro filter camera smoothing model.

For a given label and axis, this script:
  - Replays the raw 6-DoF pose from tracking_logs.csv frame by frame,
  - Applies the One Euro filter using parameters from one_euro_params.json,
  - Plots raw vs filtered signal, jitter comparison, and lag estimate.

Plots are written under data/one_euro_plots/smoothing/.

Filter behaviour
-----------------
The One Euro filter is a velocity-adaptive first-order low-pass filter. Its
cutoff frequency is recomputed every frame based on the instantaneous speed
of the signal. At low speed the cutoff stays near fc_min, producing maximum
smoothing. At high speed the cutoff rises proportionally to beta * |velocity|,
allowing the filter to become transparent and track fast motion without lag.

Two parameters govern the filter per axis:
  fc_min  — minimum cutoff frequency in Hz. Controls jitter suppression at rest.
  beta    — speed coefficient. Controls how quickly smoothing backs off as speed
            increases.
A third parameter, d_cutoff, sets the cutoff of the derivative smoother and
rarely requires adjustment from its default.

Rotation channel handling
--------------------------
Euler angles are cyclic. A camera rotating from 179 deg to -177 deg has moved
4 degrees, but the raw difference reads as -356 deg. Without correction, this
produces a large false velocity spike that incorrectly opens the filter cutoff.
Per-axis angle unwrapping converts the rotation signal into a continuous
real-valued domain before any derivative is computed. The filtered output is
wrapped back to (-180, 180] degrees before being stored and plotted.

Causal derivative
------------------
The filter estimates velocity using a causal backward difference on the
filtered signal: dx = (x_current - x_prev_filtered) / dt. This matches the
runtime implementation exactly, ensuring simulation metrics are reproducible
in the deployed C++ filter.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sp_signal

POS_AXES: List[str] = ["X_pose", "Y_pose", "Z_pose"]
ROT_AXES: List[str] = ["X_rot", "Y_rot", "Z_rot"]
ALL_AXES: List[str] = POS_AXES + ROT_AXES

MIN_DT: float = 1e-4  # seconds — matches kinematics.py guard


# ---------------------------------------------------------------------------
# Config loading and label selection — mirrors the pattern in other sim scripts
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> Dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_label(
    logs_df: pd.DataFrame,
    label: str = None,
    scenario: str = None,
    take: int = None,
) -> str:
    """
    Resolve which recording take to simulate.

    Priority: explicit --label over (--scenario, --take) pair.
    Raises ValueError if the requested label or scenario/take combination
    does not exist in the dataset.
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
                f"Multiple labels for scenario='{scenario}', take={take}: {labels}. "
                "Specify --label explicitly."
            )
        return labels[0]

    raise ValueError("Provide either --label or both --scenario and --take.")


# ---------------------------------------------------------------------------
# One Euro filter primitives
# ---------------------------------------------------------------------------


def _smoothing_alpha(dt: float, cutoff_hz: float) -> float:
    """
    Compute the EMA alpha coefficient for a given time step and cutoff frequency.

    Derived from matching the discrete EMA transfer function to an analogue
    RC low-pass filter with time constant tau = 1 / (2 * pi * fc):

        r     = 2 * pi * cutoff_hz * dt
        alpha = r / (r + 1)

    At cutoff -> 0: alpha -> 0 (output barely moves — maximum smoothing).
    At cutoff -> inf: alpha -> 1 (output equals input — no smoothing).
    """
    r = 2.0 * np.pi * cutoff_hz * dt
    return r / (r + 1.0)


def _unwrap_step(angle: float, prev_unwrapped: float) -> float:
    """
    Return the unwrapped continuation of angle given the previous unwrapped value.

    Adds or subtracts 360 deg to keep the step within (-180, 180], converting
    a cyclic Euler angle into a value on the same continuous branch as
    prev_unwrapped.
    """
    diff = angle - prev_unwrapped
    if diff > 180.0:
        diff -= 360.0
    elif diff < -180.0:
        diff += 360.0
    return prev_unwrapped + diff


def _wrap_angle(angle: float) -> float:
    """Map any angle in degrees back to the range (-180, 180]."""
    return (angle + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Core simulation loop
# ---------------------------------------------------------------------------


def run_simulation_for_axis(
    logs_df: pd.DataFrame,
    axis: str,
    label: str,
    fc_min: float,
    beta: float,
    d_cutoff: float,
) -> Dict[str, np.ndarray]:
    """
    Replay the One Euro filter on a single label and axis.

    The filter state is initialised on the first valid sample and advanced
    frame by frame in chronological order. The simulation uses only past and
    present data at each step, matching the causal behaviour of the C++ runtime.

    Parameters
    ----------
    logs_df : pd.DataFrame
        tracking_logs.csv loaded as a DataFrame.
    axis : str
        Pose axis to filter (one of ALL_AXES).
    label : str
        Recording take to simulate.
    fc_min : float
        Minimum cutoff frequency in Hz.
    beta : float
        Speed coefficient.
    d_cutoff : float
        Cutoff frequency for the derivative smoother in Hz.

    Returns
    -------
    dict with keys:
        time     — sample timestamps (s)
        dt       — per-frame time steps (s)
        raw      — raw pose values
        filtered — One Euro filtered values
    """
    is_rot = axis in ROT_AXES

    subset = logs_df[logs_df["label"] == label].sort_values("time").copy()
    if subset.empty:
        raise ValueError(f"No data for label '{label}'.")

    t_arr = subset["time"].to_numpy(dtype=float)
    raw_arr = subset[axis].to_numpy(dtype=float)
    n = len(t_arr)

    # Compute dt using the same backward-difference and MIN_DT guard as
    # kinematics.py, so the time steps used here are numerically consistent
    # with those stored in tracking_derivatives.csv.
    dt_arr = np.empty(n)
    dt_arr[0] = np.nan
    if n > 1:
        raw_dt = np.diff(t_arr)
        raw_dt = np.where(raw_dt < MIN_DT, MIN_DT, raw_dt)
        dt_arr[1:] = raw_dt

    filtered_arr = np.full(n, np.nan)

    # Filter state
    x_hat: float = np.nan  # previous filtered value (continuous domain)
    dx_hat: float = 0.0  # previous filtered derivative
    t_prev: float = np.nan
    x_prev_unwrapped: float = np.nan

    for i in range(n):
        x_raw = float(raw_arr[i])
        t_now = float(t_arr[i])

        if np.isnan(x_raw):
            continue

        # Initialise filter state on first valid sample.
        if np.isnan(x_hat):
            x_hat = x_raw
            x_prev_unwrapped = x_raw
            t_prev = t_now
            filtered_arr[i] = _wrap_angle(x_hat) if is_rot else x_hat
            continue

        dt = t_now - t_prev
        if dt < MIN_DT:
            filtered_arr[i] = _wrap_angle(x_hat) if is_rot else x_hat
            continue

        # Rotation: convert raw angle to continuous domain before differentiation.
        if is_rot:
            x_in = _unwrap_step(x_raw, x_prev_unwrapped)
            x_prev_unwrapped = x_in
        else:
            x_in = x_raw

        # Causal backward-difference derivative computed from the filtered
        # previous value, reducing derivative noise.
        dx_raw = (x_in - x_hat) / dt

        # Smooth the derivative with a fixed-cutoff EMA.
        alpha_d = _smoothing_alpha(dt, d_cutoff)
        dx_hat = alpha_d * dx_raw + (1.0 - alpha_d) * dx_hat

        # Adaptive cutoff: rises linearly with derivative magnitude.
        fc = fc_min + beta * abs(dx_hat)

        # Filter the primary signal.
        alpha = _smoothing_alpha(dt, fc)
        x_hat = alpha * x_in + (1.0 - alpha) * x_hat

        t_prev = t_now
        filtered_arr[i] = _wrap_angle(x_hat) if is_rot else x_hat

    return {
        "time": t_arr,
        "dt": dt_arr,
        "raw": raw_arr,
        "filtered": filtered_arr,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def jitter_metric(series: np.ndarray, dt: np.ndarray) -> float:
    """
    RMS of frame-to-frame first differences, normalised by mean dt.

    Quantifies high-frequency energy in the signal. A lower value after
    filtering indicates effective jitter suppression. Uses mean dt as an
    approximation for per-step spacing, consistent with the other sim scripts.
    """
    if len(series) < 2:
        return 0.0
    diff = np.diff(series)
    valid_dt = dt[np.isfinite(dt) & (dt > 0)]
    mean_dt = float(np.mean(valid_dt)) if valid_dt.size > 0 else 1.0
    return float(np.sqrt(np.mean((diff / mean_dt) ** 2)))


def lag_estimate(
    raw: np.ndarray,
    filtered: np.ndarray,
    dt: np.ndarray,
) -> float:
    """
    Estimate the time lag introduced by the filter using normalised
    cross-correlation.

    Cross-correlation measures the similarity between two signals as one is
    shifted in time relative to the other. The shift at which similarity is
    maximised is the estimated lag. A result near zero indicates the filter is
    tracking the raw signal without temporal offset — the ideal outcome for a
    speed-adaptive filter on fast motion.

    Returns lag in seconds. Positive values mean the filtered signal lags
    behind the raw signal.
    """
    if len(raw) != len(filtered) or len(raw) < 2:
        return 0.0

    raw_norm = (raw - np.mean(raw)) / (np.std(raw) + 1e-9)
    filt_norm = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-9)

    corr = sp_signal.correlate(filt_norm, raw_norm, mode="full")
    lags = sp_signal.correlation_lags(len(raw_norm), len(filt_norm), mode="full")

    lag_samples = int(lags[np.argmax(corr)])
    valid_dt = dt[np.isfinite(dt) & (dt > 0)]
    mean_dt = float(np.mean(valid_dt)) if valid_dt.size > 0 else 1.0
    return lag_samples * mean_dt


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_plots(
    label: str,
    axis: str,
    sim: Dict[str, np.ndarray],
    output_dir: str,
    settle_seconds: float = 3.0,
) -> None:
    """
    Produce a three-panel JPG comparison plot for one label and axis.

    Panel 1 — raw vs filtered signal over the full take duration.
    Panel 2 — jitter bar chart: raw vs filtered RMS jitter, computed over
              the settled region only (after settle_seconds).
    Panel 3 — difference (raw - filtered) over the settled region, with
              lag estimate. The settle period is excluded because the filter
              state is still converging during that window, and any motion
              event at take start dominates the lag cross-correlation and
              inflates the jitter metric without representing steady-state
              filter behaviour.

    A vertical dashed line at t = settle_seconds in panels 1 and 3 marks
    where metric computation begins.

    The layout mirrors the other model sim scripts to allow direct visual
    comparison of filtering behaviour across models on the same take.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{label}_{axis}_one_euro.jpg")

    t = sim["time"]
    dt = sim["dt"]
    raw = sim["raw"]
    filtered = sim["filtered"]

    valid = ~(np.isnan(raw) | np.isnan(filtered))

    # Settled region: frames after the settle window, measured from the
    # first valid timestamp in the take.
    t_start = t[valid][0] if valid.any() else 0.0
    settled = valid & (t >= t_start + settle_seconds)

    # Fall back to the full valid region if settle window covers the whole take.
    if settled.sum() < 10:
        settled = valid
        settle_label = "(full take)"
    else:
        settle_label = f"(after {settle_seconds:.0f}s settle)"

    j_raw = jitter_metric(raw[settled], dt[settled])
    j_filt = jitter_metric(filtered[settled], dt[settled])
    j_reduction = (j_raw / j_filt) if j_filt > 0 else float("inf")
    lag_sec = lag_estimate(raw[settled], filtered[settled], dt[settled])

    plt.figure(figsize=(12, 9))

    # Panel 1: raw vs filtered — full take
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t[valid], raw[valid], linewidth=1, label="raw", color="gray")
    ax1.plot(
        t[valid],
        filtered[valid],
        linewidth=1.2,
        label="filtered (One Euro)",
        color="royalblue",
    )
    ax1.axvline(
        t_start + settle_seconds,
        color="orange",
        linestyle="--",
        linewidth=0.9,
        label=f"settle boundary ({settle_seconds:.0f}s)",
    )
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"{label} – {axis}: raw vs filtered (One Euro)")
    ax1.legend(fontsize="small")

    # Panel 2: jitter bar chart — settled region only
    ax2 = plt.subplot(3, 1, 2)
    bars_x = np.arange(2)
    bars_vals = [j_raw, j_filt]
    bars_labels = ["raw", "filtered"]
    ax2.bar(bars_x, bars_vals, color=["tab:red", "tab:green"])
    ax2.set_xticks(bars_x)
    ax2.set_xticklabels(bars_labels)
    ymax = max(j_raw, j_filt)
    ax2.set_ylim(0, ymax * 1.2 if ymax > 0 else 1.0)
    ax2.set_ylabel("jitter (RMS of diff/dt)")
    title_jitter = (
        f"Jitter {settle_label}: {j_raw:.3f} → {j_filt:.3f}  ({j_reduction:.2f}x reduction)"
        if j_filt > 0
        else f"Jitter {settle_label}: {j_raw:.3f} → {j_filt:.3f}"
    )
    ax2.set_title(title_jitter)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 3: difference over settled region
    ax3 = plt.subplot(3, 1, 3)
    diff_settled = raw[settled] - filtered[settled]
    ax3.plot(
        t[settled], diff_settled, linewidth=1, color="#0b9ea8", label="raw − filtered"
    )
    ax3.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax3.fill_between(t[settled], 0, diff_settled, color="gray", alpha=0.2)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Difference")
    ax3.set_title(f"Lag estimate {settle_label} ≈ {lag_sec * 1000:.1f} ms")
    ax3.legend(fontsize="small")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(
        f"[one_euro_smoothing_sim] Saved plot to {out_path}\n"
        f"  Jitter raw      : {j_raw:.4f}  {settle_label}\n"
        f"  Jitter filtered : {j_filt:.4f}\n"
        f"  Reduction       : {j_reduction:.2f}x\n"
        f"  Lag estimate    : {lag_sec * 1000:.1f} ms"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate the One Euro filter on a single recording take and axis, "
            "and produce comparison plots under data/one_euro_plots/smoothing/."
        )
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="data/processed/tracking_logs.csv",
        help="Path to tracking_logs.csv.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="data/config/one_euro_params.json",
        help="Path to one_euro_params.json.",
    )
    parser.add_argument(
        "--axis",
        type=str,
        required=True,
        help="Axis to simulate (one of X_pose, Y_pose, Z_pose, X_rot, Y_rot, Z_rot).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Exact label to simulate (e.g. still_on_tripod_01).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Scenario name. Used with --take when --label is not provided.",
    )
    parser.add_argument(
        "--take",
        type=int,
        default=None,
        help="Take number. Used with --scenario when --label is not provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/one_euro_plots/smoothing",
        help="Directory to save JPG plots.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=3.0,
        help=(
            "Seconds to exclude from the start of each take when computing "
            "jitter and lag metrics. The filter state converges during this "
            "window and any motion event at take start would otherwise dominate "
            "both metrics. Default: 3.0. Set to 0 to disable."
        ),
    )

    args = parser.parse_args()

    if args.axis not in ALL_AXES:
        raise ValueError(f"--axis must be one of {ALL_AXES}, got '{args.axis}'.")

    logs_df = pd.read_csv(args.logs)
    cfg = load_config(args.config)

    label = select_label(
        logs_df,
        label=args.label,
        scenario=args.scenario,
        take=args.take,
    )

    if "axes" not in cfg or args.axis not in cfg["axes"]:
        raise ValueError(
            f"Axis '{args.axis}' not found in config. "
            f"Available axes: {list(cfg.get('axes', {}).keys())}"
        )

    axis_cfg = cfg["axes"][args.axis]
    fc_min = float(axis_cfg["fc_min"])
    beta = float(axis_cfg["beta"])
    d_cutoff = float(axis_cfg.get("d_cutoff", 1.0))

    print(
        f"[one_euro_smoothing_sim] label='{label}'  axis='{args.axis}'  "
        f"fc_min={fc_min}  beta={beta}  d_cutoff={d_cutoff}"
    )

    sim = run_simulation_for_axis(
        logs_df=logs_df,
        axis=args.axis,
        label=label,
        fc_min=fc_min,
        beta=beta,
        d_cutoff=d_cutoff,
    )

    make_plots(
        label,
        args.axis,
        sim,
        output_dir=args.output_dir,
        settle_seconds=args.settle_seconds,
    )


if __name__ == "__main__":
    main()
