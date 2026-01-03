# parse_validation.py
import re
from pathlib import Path
import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_DIR = Path("data/validation/raw")
OUTPUT_DIR_VALIDATION = Path("data/validation/processed/")
OUTPUT_DIR_SIMULATION = Path("data/validation/simulate/")

# Math Constants for Derivatives
WINDOW_SIZE = 8  # Matches C++ Ring Buffer size
DT_DEFAULT = 0.011  # ~90Hz fallback


def parse_log_file(file_path):
    print(f"--> Scanning: {file_path.name}")
    if not file_path.exists():
        print(f"    Error: File not found!")
        return None

    parsed_rows = []

    # Robust Regex to capture the CSV part after TRACKDATA
    regex_pattern = r"TRACKDATA.*?,(.*)"
    match_count = 0

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "TRACKDATA" in line:
                match = re.search(regex_pattern, line)
                if match:
                    try:
                        csv_content = match.group(1)
                        # Cleanups
                        csv_content = csv_content.replace(r"\n", "")
                        csv_content = csv_content.replace("\n", "").replace("\r", "")
                        csv_content = csv_content.replace('"', "").strip()

                        parts = [p.strip() for p in csv_content.split(",")]
                        parts = [p for p in parts if p]

                        if len(parts) >= 14:
                            row = {
                                "Time": float(parts[0]),
                                "Label": parts[1],
                                # Raw
                                "X_pose_raw": float(parts[2]),
                                "Y_pose_raw": float(parts[3]),
                                "Z_pose_raw": float(parts[4]),
                                "X_rot_raw": float(parts[5]),
                                "Y_rot_raw": float(parts[6]),
                                "Z_rot_raw": float(parts[7]),
                                # Smooth
                                "X_pose_smoothed": float(parts[8]),
                                "Y_pose_smoothed": float(parts[9]),
                                "Z_pose_smoothed": float(parts[10]),
                                "X_rot_smoothed": float(parts[11]),
                                "Y_rot_smoothed": float(parts[12]),
                                "Z_rot_smoothed": float(parts[13]),
                            }
                            parsed_rows.append(row)
                            match_count += 1
                    except ValueError:
                        continue

    print(f"    Found {match_count} valid frames.")
    return pd.DataFrame(parsed_rows)


def convert_to_simulation_format(df, original_filename):
    """
    Converts Validation DF -> Simulation Standard Format.
    """
    sim_data = []

    for _, row in df.iterrows():
        label = row["Label"]
        sim_row = {
            "time": row["Time"],
            "label": label,
            "scenario": label,  # Use label as scenario
            "take": 1,  # Default to 1 as we split by file
            # Map Raw values to standard names
            "X_pose": row["X_pose_raw"],
            "Y_pose": row["Y_pose_raw"],
            "Z_pose": row["Z_pose_raw"],
            "X_rot": row["X_rot_raw"],
            "Y_rot": row["Y_rot_raw"],
            "Z_rot": row["Z_rot_raw"],
            "file": original_filename,
        }
        sim_data.append(sim_row)

    return pd.DataFrame(sim_data)


def calculate_derivatives_formatted(df):
    """
    Calculates Velocity and Acceleration matching the naming convention.
    """
    df = df.sort_values("time").reset_index(drop=True)

    # Calculate dt
    df["dt"] = df["time"].diff().fillna(DT_DEFAULT)
    df["dt"] = df["dt"].replace(0, DT_DEFAULT)

    axes = ["X_pose", "Y_pose", "Z_pose", "X_rot", "Y_rot", "Z_rot"]

    # Initialize result with metadata
    results = df[["time", "label", "scenario", "take", "dt"]].copy()

    for axis in axes:
        # 1. Velocity (V_)
        vel = df[axis].diff() / df["dt"]
        vel = vel.fillna(0)

        # 2. Acceleration (A_)
        acc = vel.diff() / df["dt"]
        acc = acc.fillna(0)

        # 3. Add to results
        results[f"V_{axis}"] = vel
        results[f"A_{axis}"] = acc

    return results


def process_dataframe(df, base_filename, label_name):
    """
    Helper function to process and save a single dataframe group.
    """
    # Sanitize label for filename (remove spaces, slashes)
    safe_label = "".join(c for c in label_name if c.isalnum() or c in ("_", "-"))

    # Construct distinct filename: e.g. Node_0_Run01Linear
    file_stem = f"{base_filename}_{safe_label}"

    # 2. Save Validation CSV (Raw vs Smooth)
    val_out_path = OUTPUT_DIR_VALIDATION / f"{file_stem}_parsed.csv"
    df.to_csv(val_out_path, index=False)
    print(f"    [VAL] Saved: {val_out_path.name}")

    # 3. Convert to Sim Format (Raw only)
    df_sim = convert_to_simulation_format(df, file_stem)
    sim_out_path = OUTPUT_DIR_SIMULATION / f"{file_stem}.csv"
    df_sim.to_csv(sim_out_path, index=False)
    print(f"    [SIM] Saved: {sim_out_path.name}")

    # 4. Calculate Derivatives (V_, A_)
    df_derivs = calculate_derivatives_formatted(df_sim)
    deriv_out_path = OUTPUT_DIR_SIMULATION / f"{file_stem}_derivatives.csv"
    df_derivs.to_csv(deriv_out_path, index=False)
    print(f"    [DRV] Saved: {deriv_out_path.name}")


def main():
    if not OUTPUT_DIR_VALIDATION.exists():
        os.makedirs(OUTPUT_DIR_VALIDATION)
    if not OUTPUT_DIR_SIMULATION.exists():
        os.makedirs(OUTPUT_DIR_SIMULATION)

    if not INPUT_DIR.exists():
        print(f"Error: Input directory '{INPUT_DIR}' does not exist.")
        return

    log_files = list(INPUT_DIR.glob("*.log"))

    if not log_files:
        print(f"No .log files found in {INPUT_DIR}")
        return

    for log_file in log_files:
        # 1. Parse the full monolithic log
        df_full = parse_log_file(log_file)

        if df_full is not None and not df_full.empty:

            # --- NEW LOGIC: Split by Label ---
            # Group the dataframe by the 'Label' column
            grouped = df_full.groupby("Label")

            print(
                f"    -> splitting '{log_file.name}' into {len(grouped)} separate datasets..."
            )

            for label_name, df_group in grouped:
                # Process each group as if it were a separate file
                process_dataframe(df_group, log_file.stem, str(label_name))

            print("-" * 30)
        else:
            print(f"    [WARNING] No data extracted from {log_file.name}")


if __name__ == "__main__":
    main()
