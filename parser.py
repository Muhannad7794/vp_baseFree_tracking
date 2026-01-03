# parser.py
from pathlib import Path
import re
import pandas as pd

# Directory with your per-scenario log/text files
LOG_DIR = Path("data/raw")

# Output CSV to data/processed/tracking_logs.csv
OUTPUT_CSV = LOG_DIR.parent / "processed" / "tracking_logs.csv"

# Valid Training Scenarios
# We use this to filter the *content* of the logs, not the filenames.
VALID_SCENARIO_PREFIXES = [
    "controlled_handheld_pan",
    "controlled_handheld_tilt",
    "controlled_on_tripod_pan",
    "controlled_on_tripod_tilt",
    "fast_pan_tripod",
    "fast_tilt_tripod",
    "handheld_full_nav",
    "handheld_still",
    "slide_handheld",
    "StillOnTripod",
    "travel_handheld",
]


def parse_track_line(line: str):
    """
    Parse each UE log line that contains TRACKDATA.
    """
    if "TRACKDATA" not in line:
        return None

    parts = line.split('"')

    # Find the TRACKDATA token
    try:
        idx_track = parts.index("TRACKDATA")
    except ValueError:
        return None

    time_val = None
    label = None
    numeric_vals = []

    # Walk all segments AFTER TRACKDATA and classify them
    for seg in parts[idx_track + 1 :]:
        s = seg.strip()
        if not s or s == "," or s == r"\n":
            continue

        # First token after TRACKDATA must be time (numeric)
        if time_val is None:
            try:
                time_val = float(s)
                continue
            except ValueError:
                return None

        # Next token is label (string)
        if label is None:
            label = s
            continue

        # Remaining tokens should be numeric values (6 DOF)
        try:
            numeric_vals.append(float(s))
        except ValueError:
            continue

    # Expect exactly 6 numeric values for 6 DOF
    if time_val is None or label is None or len(numeric_vals) != 6:
        return None

    X_pose, Y_pose, Z_pose, X_rot, Y_rot, Z_rot = numeric_vals

    # Derive scenario + take from label, e.g. "fast_pan_tripod_01"
    # Logic: Strip the trailing digits to find the scenario name
    m = re.match(r"(.+?)_?0*(\d+)$", label)
    if m:
        scenario = m.group(1)
        take = int(m.group(2))
    else:
        scenario = label
        take = 1  # Default to take 1 if no number found

    return {
        "time": time_val,
        "label": label,
        "scenario": scenario,
        "take": take,
        "X_pose": X_pose,
        "Y_pose": Y_pose,
        "Z_pose": Z_pose,
        "X_rot": X_rot,
        "Y_rot": Y_rot,
        "Z_rot": Z_rot,
    }


def main():
    rows = []

    # Process all .log and .txt files under ./data/raw
    # We no longer filter files by name, so we can support monolithic session logs.
    files = list(LOG_DIR.glob("*.log")) + list(LOG_DIR.glob("*.txt"))

    if not files:
        print(f"No log files found in {LOG_DIR}")
        return

    print(f"Scanning {len(files)} files in {LOG_DIR}...")

    for file_path in files:
        file_row_count = 0
        stem = file_path.stem

        with file_path.open(encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "TRACKDATA" not in line:
                    continue

                row = parse_track_line(line)
                if row is None:
                    continue

                # Filter: Only keep rows where the internal label matches our training set
                # This prevents random test labels (like "Test_Run_01") from polluting the training data
                if not any(
                    row["scenario"].startswith(base) for base in VALID_SCENARIO_PREFIXES
                ):
                    continue

                row["file"] = stem
                rows.append(row)
                file_row_count += 1

        if file_row_count > 0:
            print(
                f"  -> Extracted {file_row_count} valid training samples from {file_path.name}"
            )

    if not rows:
        raise RuntimeError(
            "No valid TRACKDATA rows found. Check your log files and label names."
        )

    df = pd.DataFrame(rows)

    # Sort for sanity: by scenario, then take, then time
    df.sort_values(["scenario", "take", "time"], inplace=True)

    # Save combined dataset
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSuccess! Saved {len(df)} total rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
