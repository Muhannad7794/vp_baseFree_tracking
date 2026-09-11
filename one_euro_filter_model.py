# one_euro_filter_model.py
"""
Offline analysis and configuration script for the One Euro Filter camera smoothing model.

Purpose
-------
This script does two things:

1. ANALYSIS — reads tracking_derivatives.csv and characterises the per-axis
   velocity distribution across all recorded scenarios. This gives quantitative
   context for setting fc_min and beta: knowing the typical resting velocity
   (stable shot) and the peak velocity (fast pan) tells the operator what speed
   range the filter will encounter at runtime.

2. VALIDATION — runs the One Euro filter forward across the full dataset using
   the configured fc_min and beta values, then computes per-axis jitter and lag
   metrics. These metrics confirm that the chosen parameters produce the intended
   filtering behaviour on real recorded data before the values are carried into C++.

The script writes:
  - data/config/one_euro_params.json    : filter parameters per axis
  - data/modeled/one_euro_modeled.csv   : per-frame filtered values + metadata
  - data/one_euro_plots/analysis/       : per-axis velocity distribution plots
  - data/one_euro_plots/modeled/        : per-axis raw vs filtered overview plots

Causality and numerical consistency
-------------------------------------
The One Euro filter estimates the derivative of the input signal using a causal
backward difference: dx = (x_current - x_previous_filtered) / dt. This is the
only formula compatible with real-time execution, where future samples are not
available. The V_* columns in tracking_derivatives.csv are computed by the same
backward-difference scheme, so the velocity distributions analysed in step 1
reflect the same numerical quantities the filter operates on at runtime. Offline
validation metrics are therefore directly comparable to runtime behaviour without
a hidden discretisation discrepancy.

Rotation unwrapping note
------------------------
Raw Euler angles (X_rot, Y_rot, Z_rot) are cyclic: a camera at 179 deg that
rotates 4 deg further produces a raw jump from 179 to -177 deg. Without
correction, the backward-difference derivative reads this as -356 deg/s,
triggering a massive false velocity spike that incorrectly opens the filter
cutoff. Per-axis angle unwrapping converts each rotation channel into a
continuous real-valued signal before any derivative is computed. The filtered
output is then wrapped back to [-180, 180] deg for storage and downstream use.
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

# ---------------------------------------------------------------------------
# Axis definitions — must match kinematics.py and the rest of the pipeline
# ---------------------------------------------------------------------------

POS_AXES: List[str] = ["X_pose", "Y_pose", "Z_pose"]
ROT_AXES: List[str] = ["X_rot", "Y_rot", "Z_rot"]
ALL_AXES: List[str] = POS_AXES + ROT_AXES

# Scenario groupings — identical to those used by the sigma models so that
# distribution comparisons across models are made on the same data splits.
STABLE_SCENARIOS: List[str] = ["still_on_tripod", "handheld_still"]
SLOW_SCENARIOS: List[str] = ["controlled_on_tripod_pan", "controlled_on_tripod_tilt"]
MEDIUM_SCENARIOS: List[str] = [
    "controlled_handheld_pan",
    "controlled_handheld_tilt",
    "slide_handheld",
    "travel_handheld",
]
FAST_SCENARIOS: List[str] = [
    "fast_pan_tripod",
    "fast_tilt_tripod",
    "handheld_full_nav",
]

# Default filter parameters — starting point before visual tuning.
# These are derived from the validated skeleton-tracking parameters in the
# Signal Processing course report (min_cutoff=1.0 Hz, beta=0.05) adjusted
# upward for camera motion, which has lower-frequency true motion content
# than skeletal joints but requires tighter jitter suppression on held shots.
# Tuning recipe:
#   Pass 1 — set beta=0, lower fc_min until held-shot jitter disappears,
#             stop at the lowest value where slow moves do not feel rubbery.
#   Pass 2 — raise beta from 0, perform a fast whip pan, raise beta until
#             the pan tracks crisply without lag. Pass 1 is not disturbed.
DEFAULT_FC_MIN: float = 0.5  # Hz — minimum cutoff (max smoothing at rest)
DEFAULT_BETA: float = 0.05  # speed coefficient (cutoff rise rate with velocity)
DEFAULT_D_CUTOFF: float = 1.0  # Hz — derivative filter cutoff (rarely needs tuning)

# Minimum time step guard — prevents division by zero on duplicate timestamps.
MIN_DT: float = 1e-4  # seconds


# ---------------------------------------------------------------------------
# One Euro Filter core mathematics
# ---------------------------------------------------------------------------


def _smoothing_alpha(dt: float, cutoff_hz: float) -> float:
    """
    Compute the EMA smoothing factor alpha for a given time step and cutoff.

    The One Euro filter is a first-order low-pass filter (exponential moving
    average) whose cutoff frequency is adapted per frame. The relationship
    between cutoff frequency (Hz) and the EMA coefficient alpha is:

        r     = 2 * pi * cutoff_hz * dt
        alpha = r / (r + 1)

    Background: this derivation comes from matching the discrete EMA transfer
    function to an analogue RC low-pass filter with time constant tau = 1/(2*pi*fc).
    At fc -> 0, alpha -> 0 (maximum smoothing, output barely moves).
    At fc -> inf, alpha -> 1 (no smoothing, output equals input).

    Parameters
    ----------
    dt : float
        Elapsed time since the previous sample, in seconds.
    cutoff_hz : float
        Target cutoff frequency in Hz.

    Returns
    -------
    float
        Alpha coefficient in [0, 1].
    """
    r = 2.0 * np.pi * cutoff_hz * dt
    return r / (r + 1.0)


def _unwrap_angle_step(angle: float, prev_angle: float) -> float:
    """
    Return the unwrapped version of angle given the previous unwrapped value.

    Adds or subtracts 360 deg to minimise the absolute difference between
    the current raw angle and the previous unwrapped angle. This converts
    cyclic Euler angles into a continuous real-valued signal suitable for
    derivative estimation.

    Parameters
    ----------
    angle : float
        Current raw angle in degrees, assumed in [-180, 180].
    prev_angle : float
        Previous unwrapped angle in degrees (continuous, may exceed [-180, 180]).

    Returns
    -------
    float
        Unwrapped angle in degrees, on the same continuous branch as prev_angle.
    """
    diff = angle - prev_angle
    # Shortest path on the circle: keep diff in (-180, 180]
    if diff > 180.0:
        diff -= 360.0
    elif diff < -180.0:
        diff += 360.0
    return prev_angle + diff


def run_one_euro_on_series(
    times: np.ndarray,
    values: np.ndarray,
    fc_min: float,
    beta: float,
    d_cutoff: float,
    is_rotation: bool,
) -> np.ndarray:
    """
    Apply the One Euro filter to a single 1-D time series.

    The filter state is initialised on the first valid (non-NaN) sample.
    NaN values in the input are passed through as NaN in the output without
    corrupting the filter state, so that brief tracking gaps do not cause
    permanent drift in the filtered signal.

    Algorithm per frame
    -------------------
    1. If input is NaN, output NaN and advance without updating state.
    2. Compute dt from timestamps.
    3. If rotation axis: unwrap the raw angle relative to the previous
       unwrapped value to eliminate boundary discontinuities.
    4. Compute raw backward-difference derivative:
           dx_raw = (x_current - x_prev_filtered) / dt
       Note: derivative is computed from the *filtered* previous value, not
       the raw previous value. This matches the original One Euro paper
       (Casiez et al., 2012) and reduces derivative noise.
    5. Smooth the derivative with a fixed-cutoff EMA (d_cutoff):
           dx_hat = alpha_d * dx_raw + (1 - alpha_d) * dx_hat_prev
    6. Compute adaptive cutoff from smoothed derivative magnitude:
           fc = fc_min + beta * |dx_hat|
    7. Filter the primary signal:
           x_hat = alpha * x_current + (1 - alpha) * x_hat_prev
    8. Store state and return x_hat. For rotation axes, wrap output to
       [-180, 180] deg before returning.

    Parameters
    ----------
    times : np.ndarray
        Sample timestamps in seconds, shape (N,).
    values : np.ndarray
        Raw signal values, shape (N,). May contain NaN.
    fc_min : float
        Minimum cutoff frequency in Hz.
    beta : float
        Speed coefficient. Higher values reduce lag during fast motion.
    d_cutoff : float
        Fixed cutoff frequency for the derivative smoother in Hz.
    is_rotation : bool
        If True, apply angle unwrapping before filtering and wrap output
        back to [-180, 180] deg.

    Returns
    -------
    np.ndarray
        Filtered signal, shape (N,). NaN where input was NaN.
    """
    N = len(values)
    output = np.full(N, np.nan)

    # Filter state — initialised on the first valid sample.
    x_hat: float = np.nan  # previous filtered position
    dx_hat: float = 0.0  # previous filtered derivative
    t_prev: float = np.nan  # previous timestamp
    x_prev_unwrapped: float = np.nan  # previous unwrapped value (rotation only)

    for i in range(N):
        x_raw = float(values[i])
        t_now = float(times[i])

        # --- Pass NaN through without touching filter state ---
        if np.isnan(x_raw):
            continue

        # --- Initialise on first valid sample ---
        if np.isnan(x_hat):
            x_hat = x_raw
            x_prev_unwrapped = x_raw
            t_prev = t_now
            output[i] = x_raw
            continue

        # --- Time step ---
        dt = t_now - t_prev
        if dt < MIN_DT:
            # Duplicate or reversed timestamp — output previous filtered value.
            output[i] = x_hat if not is_rotation else _wrap_angle(x_hat)
            continue

        # --- Rotation: unwrap angle to continuous domain ---
        if is_rotation:
            x_unwrapped = _unwrap_angle_step(x_raw, x_prev_unwrapped)
            x_prev_unwrapped = x_unwrapped
            x_in = x_unwrapped
        else:
            x_in = x_raw

        # --- Step 4: backward-difference derivative from filtered previous value ---
        dx_raw = (x_in - x_hat) / dt

        # --- Step 5: smooth the derivative ---
        alpha_d = _smoothing_alpha(dt, d_cutoff)
        dx_hat = alpha_d * dx_raw + (1.0 - alpha_d) * dx_hat

        # --- Step 6: adaptive cutoff ---
        fc = fc_min + beta * abs(dx_hat)

        # --- Step 7: filter primary signal ---
        alpha = _smoothing_alpha(dt, fc)
        x_hat = alpha * x_in + (1.0 - alpha) * x_hat

        # --- Step 8: store state and write output ---
        t_prev = t_now

        if is_rotation:
            output[i] = _wrap_angle(x_hat)
        else:
            output[i] = x_hat

    return output


def _wrap_angle(angle: float) -> float:
    """Wrap an angle in degrees to the range (-180, 180]."""
    return (angle + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Velocity distribution analysis
# ---------------------------------------------------------------------------


def analyse_velocity_distributions(
    deriv_df: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """
    Characterise the per-axis velocity distribution across scenario groups.

    Uses V_* columns from tracking_derivatives.csv, which kinematics.py now
    computes via the same causal backward-difference formula used by the filter
    at runtime. These distributions quantify the speed range the filter will
    encounter, which informs the choice of fc_min and beta:

      - stable p90 indicates the highest speed expected during a held shot.
        fc_min should be set so the filter is still heavily smoothing at this
        speed (i.e. the stable p90 speed should not strongly open the cutoff).
      - fast p90 indicates the highest speed expected during an aggressive pan.
        beta should be set so the filter is nearly transparent at this speed.

    For each axis, reports:
      - p50_stable  : median |velocity| during stable scenarios
      - p90_stable  : 90th-percentile |velocity| during stable scenarios
      - p50_fast    : median |velocity| during fast scenarios
      - p90_fast    : 90th-percentile |velocity| during fast scenarios

    Parameters
    ----------
    deriv_df : pd.DataFrame
        tracking_derivatives.csv loaded as a DataFrame.

    Returns
    -------
    dict
        Nested dict: axis -> {p50_stable, p90_stable, p50_fast, p90_fast}.
    """
    result: Dict[str, Dict[str, float]] = {}

    for axis in ALL_AXES:
        v_col = f"V_{axis}"
        if v_col not in deriv_df.columns:
            print(f"  [analyse] Warning: column {v_col} not found, skipping axis.")
            continue

        stable_mask = deriv_df["scenario"].isin(STABLE_SCENARIOS)
        fast_mask = deriv_df["scenario"].isin(FAST_SCENARIOS)

        stable_speeds = deriv_df.loc[stable_mask, v_col].abs().dropna()
        fast_speeds = deriv_df.loc[fast_mask, v_col].abs().dropna()

        result[axis] = {
            "p50_stable": (
                float(np.percentile(stable_speeds, 50)) if len(stable_speeds) else 0.0
            ),
            "p90_stable": (
                float(np.percentile(stable_speeds, 90)) if len(stable_speeds) else 0.0
            ),
            "p50_fast": (
                float(np.percentile(fast_speeds, 50)) if len(fast_speeds) else 0.0
            ),
            "p90_fast": (
                float(np.percentile(fast_speeds, 90)) if len(fast_speeds) else 0.0
            ),
        }

    return result


# ---------------------------------------------------------------------------
# Full-dataset filter run and metric computation
# ---------------------------------------------------------------------------


def run_filter_on_dataset(
    logs_df: pd.DataFrame,
    axis_params: Dict[str, Tuple[float, float, float]],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Run the One Euro filter across every label and axis in tracking_logs.csv.

    For each label (a named recording take), the filter state is reset so
    takes do not bleed into each other. Within each take, samples are
    processed in chronological order.

    The velocity derivative inside the filter uses a backward difference
    on the filtered signal, matching the C++ runtime implementation.

    Parameters
    ----------
    logs_df : pd.DataFrame
        tracking_logs.csv loaded as a DataFrame. Must contain columns:
        time, label, scenario, take, X_pose, Y_pose, Z_pose,
        X_rot, Y_rot, Z_rot.
    axis_params : dict
        Mapping of axis name -> (fc_min, beta, d_cutoff).

    Returns
    -------
    result_df : pd.DataFrame
        Input DataFrame with additional columns filtered_<axis> for each axis.
    metrics : dict
        Per-axis dict containing jitter_raw, jitter_filtered, jitter_reduction,
        lag_seconds computed over the full dataset.
    """
    logs_df = logs_df.sort_values(["label", "time"]).copy()
    result_df = logs_df.copy()

    all_metrics: Dict[str, Dict[str, float]] = {}

    for axis in ALL_AXES:
        if axis not in logs_df.columns:
            print(f"  [run_filter] Warning: axis {axis} not found in logs, skipping.")
            continue

        fc_min, beta, d_cutoff = axis_params[axis]
        is_rot = axis in ROT_AXES

        filtered_values = np.full(len(logs_df), np.nan)

        # Process each label independently to prevent state from bleeding
        # across separate recording takes.
        for label, group in logs_df.groupby("label", sort=False):
            group = group.sort_values("time")
            idx = group.index
            times = group["time"].to_numpy(dtype=float)
            raw = group[axis].to_numpy(dtype=float)

            filtered = run_one_euro_on_series(
                times=times,
                values=raw,
                fc_min=fc_min,
                beta=beta,
                d_cutoff=d_cutoff,
                is_rotation=is_rot,
            )
            filtered_values[idx] = filtered

        result_df[f"filtered_{axis}"] = filtered_values

        # Compute aggregate metrics over all valid (non-NaN) frames.
        raw_all = logs_df[axis].to_numpy(dtype=float)
        filt_all = filtered_values

        valid = ~(np.isnan(raw_all) | np.isnan(filt_all))
        raw_valid = raw_all[valid]
        filt_valid = filt_all[valid]

        jitter_raw = _rms_jitter(raw_valid)
        jitter_filt = _rms_jitter(filt_valid)
        reduction = jitter_raw / jitter_filt if jitter_filt > 0.0 else float("inf")
        lag = _cross_correlation_lag(raw_valid, filt_valid)

        all_metrics[axis] = {
            "jitter_raw": jitter_raw,
            "jitter_filtered": jitter_filt,
            "jitter_reduction_x": reduction,
            "lag_seconds": lag,
        }

        print(
            f"  [{axis}] jitter raw={jitter_raw:.4f}  filtered={jitter_filt:.4f}"
            f"  reduction={reduction:.2f}x  lag={lag*1000:.1f}ms"
        )

    return result_df, all_metrics


def _rms_jitter(signal: np.ndarray) -> float:
    """
    RMS of frame-to-frame first differences — measures high-frequency energy.

    This is the same metric used by the sigma-model simulation scripts,
    enabling direct comparison across models on identical data.
    """
    if len(signal) < 2:
        return 0.0
    diffs = np.diff(signal)
    return float(np.sqrt(np.mean(diffs**2)))


def _cross_correlation_lag(raw: np.ndarray, filtered: np.ndarray) -> float:
    """
    Estimate the time lag (in samples) introduced by the filter using
    normalised cross-correlation.

    Cross-Correlation
    -----------------
    Background: Cross-correlation measures how similar two signals are as one
    is shifted in time relative to the other. For two signals x and y, the
    cross-correlation R(tau) is the inner product of x with a time-shifted
    copy of y. The shift tau at which R is maximised is the estimated lag.

    In this context: if the filtered signal consistently lags the raw signal
    by k samples, the cross-correlation will peak at lag = k. Converting
    sample lag to time requires the mean sample interval, which is approximated
    from the index since timestamps are not passed into this helper.

    A returned value of 0.0 indicates no detectable lag — the ideal result
    for a filter with good speed-adaptation.

    Note: this function returns lag in samples (not seconds) because timestamps
    are not available here. The caller in run_filter_on_dataset stores lag
    in samples directly; the simulation script converts to milliseconds using
    per-take timestamps.
    """
    from scipy import signal as sp_signal

    if len(raw) != len(filtered) or len(raw) < 2:
        return 0.0

    raw_norm = (raw - np.mean(raw)) / (np.std(raw) + 1e-9)
    filt_norm = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-9)

    corr = sp_signal.correlate(filt_norm, raw_norm, mode="full")
    lags = sp_signal.correlation_lags(len(raw_norm), len(filt_norm), mode="full")

    lag_samples = int(lags[np.argmax(corr)])
    return float(lag_samples)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_velocity_distributions(
    deriv_df: pd.DataFrame,
    dist_stats: Dict[str, Dict[str, float]],
    output_dir: str,
    axis_filter: str = None,
    scenario_filter: str = None,
) -> None:
    """
    Plot the |velocity| distribution for the specified axis and scenario.

    Two modes:
      - scenario_filter given: plots a single histogram for that scenario only.
        Useful for examining the speed range of one specific motion type.
      - scenario_filter omitted: overlays stable vs fast scenario groups.
        Useful as a global reference for setting fc_min and beta.

    In both modes, axis_filter restricts plotting to a single axis.
    When axis_filter is omitted, all axes are plotted.
    """
    os.makedirs(output_dir, exist_ok=True)

    axes_to_plot = [axis_filter] if axis_filter else ALL_AXES

    for axis in axes_to_plot:
        v_col = f"V_{axis}"
        if v_col not in deriv_df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))

        if scenario_filter:
            # Single-scenario mode.
            scen_mask = deriv_df["scenario"] == scenario_filter
            speeds = deriv_df.loc[scen_mask, v_col].abs().dropna().to_numpy()

            if len(speeds) == 0:
                print(
                    f"  [plot] No data for scenario '{scenario_filter}' on axis {axis}. Skipping."
                )
                plt.close()
                continue

            clip = np.percentile(speeds, 99)
            bins = np.linspace(0, clip, 60)
            p90 = float(np.percentile(speeds, 90))

            ax.hist(
                speeds.clip(0, clip),
                bins=bins,
                alpha=0.7,
                label=scenario_filter,
                color="steelblue",
            )
            ax.axvline(
                p90,
                color="steelblue",
                linestyle="--",
                linewidth=1.2,
                label=f"p90 = {p90:.3f}",
            )
            ax.set_title(f"{axis} — velocity distribution: {scenario_filter}")
            filename = f"{axis}_{scenario_filter}_velocity_distribution.jpg"

        else:
            # Stable vs fast group overview mode.
            stable_mask = deriv_df["scenario"].isin(STABLE_SCENARIOS)
            fast_mask = deriv_df["scenario"].isin(FAST_SCENARIOS)

            stable_speeds = deriv_df.loc[stable_mask, v_col].abs().dropna().to_numpy()
            fast_speeds = deriv_df.loc[fast_mask, v_col].abs().dropna().to_numpy()

            clip = np.percentile(fast_speeds, 99) if len(fast_speeds) else 1.0
            bins = np.linspace(0, clip, 60)

            ax.hist(
                stable_speeds.clip(0, clip),
                bins=bins,
                alpha=0.6,
                label="stable scenarios",
                color="steelblue",
            )
            ax.hist(
                fast_speeds.clip(0, clip),
                bins=bins,
                alpha=0.6,
                label="fast scenarios",
                color="tomato",
            )

            if axis in dist_stats:
                s = dist_stats[axis]
                ax.axvline(
                    s["p90_stable"],
                    color="steelblue",
                    linestyle="--",
                    linewidth=1.2,
                    label=f"stable p90 = {s['p90_stable']:.3f}",
                )
                ax.axvline(
                    s["p90_fast"],
                    color="tomato",
                    linestyle="--",
                    linewidth=1.2,
                    label=f"fast p90 = {s['p90_fast']:.3f}",
                )

            ax.set_title(f"{axis} — velocity distribution by scenario group")
            filename = f"{axis}_velocity_distribution.jpg"

        ax.set_xlabel("|velocity| (units/s)")
        ax.set_ylabel("frame count")
        ax.legend(fontsize="small")

        plt.tight_layout()
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [plot] Saved {out_path}")


# ---------------------------------------------------------------------------
# Config JSON construction
# ---------------------------------------------------------------------------


def build_config(
    axis_params: Dict[str, Tuple[float, float, float]],
    dist_stats: Dict[str, Dict[str, float]],
    metrics: Dict[str, Dict[str, float]],
) -> dict:
    """
    Assemble the one_euro_params.json structure.

    The JSON is the authoritative parameter source consumed by both the
    simulation script and the C++ runtime. Structure:

    {
      "model": "one_euro",
      "axes": {
        "X_pose": {
          "fc_min": float,        // minimum cutoff frequency (Hz)
          "beta": float,          // speed coefficient
          "d_cutoff": float,      // derivative smoother cutoff (Hz)
          "velocity_p90_stable":  // informational: stable-scenario speed ceiling
          "velocity_p90_fast":    // informational: fast-scenario speed ceiling
          "jitter_reduction_x":   // validation: factor by which filter reduces jitter
          "lag_seconds": float    // validation: estimated lag in seconds
        },
        ...
      }
    }
    """
    axes_out = {}
    for axis in ALL_AXES:
        fc_min, beta, d_cutoff = axis_params.get(
            axis, (DEFAULT_FC_MIN, DEFAULT_BETA, DEFAULT_D_CUTOFF)
        )
        entry = {
            "fc_min": fc_min,
            "beta": beta,
            "d_cutoff": d_cutoff,
        }
        if axis in dist_stats:
            entry["velocity_p90_stable"] = dist_stats[axis]["p90_stable"]
            entry["velocity_p90_fast"] = dist_stats[axis]["p90_fast"]
        if axis in metrics:
            entry["jitter_raw"] = metrics[axis]["jitter_raw"]
            entry["jitter_filtered"] = metrics[axis]["jitter_filtered"]
            entry["jitter_reduction_x"] = metrics[axis]["jitter_reduction_x"]
            entry["lag_samples"] = metrics[axis][
                "lag_seconds"
            ]  # stored as sample count
        axes_out[axis] = entry

    return {"model": "one_euro", "axes": axes_out}


# ---------------------------------------------------------------------------
# CLI and main entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One Euro filter analysis and configuration for camera tracking."
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="data/processed/tracking_logs.csv",
        help="Path to tracking_logs.csv (raw 6-DoF pose data).",
    )
    parser.add_argument(
        "--derived",
        type=str,
        default="data/derived/tracking_derivatives.csv",
        help="Path to tracking_derivatives.csv (kinematics output, used for analysis only).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/modeled/one_euro_modeled.csv",
        help="Output CSV with per-frame filtered values appended.",
    )
    parser.add_argument(
        "--config-output",
        type=str,
        default="data/config/one_euro_params.json",
        help="Output path for one_euro_params.json.",
    )
    parser.add_argument(
        "--plot-axis",
        type=str,
        default=None,
        help=(
            "Axis for which to generate a velocity distribution plot "
            "(e.g. Y_rot, X_pose). Use alone for a stable vs fast group "
            "overview, or combine with --plot-scenario for a single scenario."
        ),
    )
    parser.add_argument(
        "--plot-scenario",
        type=str,
        default=None,
        help=(
            "Scenario to plot (e.g. fast_pan_tripod). Used with --plot-axis. "
            "When given, the velocity distribution is computed for that scenario "
            "only. When omitted, stable and fast scenario groups are overlaid."
        ),
    )
    parser.add_argument(
        "--plot-analysis-dir",
        type=str,
        default="data/one_euro_plots/analysis",
        help="Directory where velocity distribution plots are written.",
    )

    # Per-axis parameter overrides — allow individual axis tuning from the CLI
    # without editing the script. Format: --fc-min-X-pose 0.3
    for axis in ALL_AXES:
        safe = axis.replace("_", "-").lower()
        parser.add_argument(
            f"--fc-min-{safe}",
            type=float,
            default=None,
            help=f"fc_min override for {axis} (Hz).",
        )
        parser.add_argument(
            f"--beta-{safe}",
            type=float,
            default=None,
            help=f"beta override for {axis}.",
        )
        parser.add_argument(
            f"--d-cutoff-{safe}",
            type=float,
            default=None,
            help=f"d_cutoff override for {axis} (Hz).",
        )

    # Global defaults (applied to all axes unless overridden per-axis)
    parser.add_argument(
        "--fc-min",
        type=float,
        default=DEFAULT_FC_MIN,
        help="Global default fc_min (Hz).",
    )
    parser.add_argument(
        "--beta", type=float, default=DEFAULT_BETA, help="Global default beta."
    )
    parser.add_argument(
        "--d-cutoff",
        type=float,
        default=DEFAULT_D_CUTOFF,
        help="Global default d_cutoff (Hz).",
    )

    return parser.parse_args()


def resolve_axis_params(
    args: argparse.Namespace,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Build per-axis (fc_min, beta, d_cutoff) tuples from CLI arguments.

    Per-axis overrides take precedence over the global defaults.
    """
    params: Dict[str, Tuple[float, float, float]] = {}
    for axis in ALL_AXES:
        safe = axis.replace("_", "-").lower()
        fc_min = getattr(args, f"fc_min_{safe.replace('-', '_')}", None) or args.fc_min
        beta = getattr(args, f"beta_{safe.replace('-', '_')}", None) or args.beta
        d_cutoff = (
            getattr(args, f"d_cutoff_{safe.replace('-', '_')}", None) or args.d_cutoff
        )
        params[axis] = (fc_min, beta, d_cutoff)
    return params


def main() -> None:
    args = parse_args()
    axis_params = resolve_axis_params(args)

    # --- Load inputs ---
    if not os.path.exists(args.logs):
        raise FileNotFoundError(f"tracking_logs.csv not found: {args.logs}")
    if not os.path.exists(args.derived):
        raise FileNotFoundError(f"tracking_derivatives.csv not found: {args.derived}")

    print("[one_euro_filter_model] Loading input files...")
    logs_df = pd.read_csv(args.logs)
    deriv_df = pd.read_csv(args.derived)

    required_log_cols = ["time", "label", "scenario", "take"] + ALL_AXES
    missing = set(required_log_cols) - set(logs_df.columns)
    if missing:
        raise ValueError(f"tracking_logs.csv is missing columns: {missing}")

    # --- Step 1: Velocity distribution analysis ---
    print("\n[one_euro_filter_model] Step 1/3: Analysing velocity distributions...")
    dist_stats = analyse_velocity_distributions(deriv_df)

    print("\n  Per-axis velocity summary (|V| in signal units/s):")
    for axis, s in dist_stats.items():
        print(
            f"  {axis:10s}  stable p90={s['p90_stable']:8.3f}  fast p90={s['p90_fast']:8.3f}"
        )

    # --- Step 2: Run filter across full dataset ---
    print(
        "\n[one_euro_filter_model] Step 2/3: Running One Euro filter across full dataset..."
    )
    result_df, metrics = run_filter_on_dataset(logs_df, axis_params)

    # --- Step 3: Write outputs ---
    print("\n[one_euro_filter_model] Step 3/3: Writing output files...")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result_df.to_csv(args.output, index=False)
    print(f"  Modeled CSV written to {args.output}")

    config = build_config(axis_params, dist_stats, metrics)
    os.makedirs(os.path.dirname(args.config_output), exist_ok=True)
    with open(args.config_output, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"  Config JSON written to {args.config_output}")

    # --- Optional: velocity distribution plot for one axis ---
    if args.plot_axis:
        if args.plot_axis not in ALL_AXES:
            print(
                f"  [warning] --plot-axis '{args.plot_axis}' is not a recognised axis. "
                f"Valid axes: {ALL_AXES}"
            )
        else:
            print(
                f"\n[one_euro_filter_model] Generating velocity distribution plot "
                f"for axis={args.plot_axis}"
                + (f", scenario={args.plot_scenario}" if args.plot_scenario else "")
                + "..."
            )
            plot_velocity_distributions(
                deriv_df,
                dist_stats,
                args.plot_analysis_dir,
                axis_filter=args.plot_axis,
                scenario_filter=args.plot_scenario,
            )

    print("\n[one_euro_filter_model] Done.")


if __name__ == "__main__":
    main()
