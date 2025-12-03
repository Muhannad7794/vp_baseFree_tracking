# parse_logs.py
from pathlib import Path
import re
import pandas as pd

# Directory with your per-scenario log/text files
LOG_DIR = Path("data")

# Output CSV to data/processed/tracking_logs.csv
OUTPUT_CSV = LOG_DIR / "processed" / "tracking_logs.csv"

# Scenario base names (file stems and label prefixes)
SCENARIO_BASES = [
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

    to match the blueprint structure:
      "TRACKDATA","<time>","<label>",
      "<X_pose>","<Y_pose>","<Z_pose>",
      "<X_rot>","<Y_rot>","<Z_rot>"\n

    i.e. one time value, then label, then 6 DOF.
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
                # If time doesn't parse, skip line
                return None

        # Next token is label (string)
        if label is None:
            label = s
            continue

        # all remaining tokens should be numeric values (6 DOF)
        try:
            numeric_vals.append(float(s))
        except ValueError:
            # Ignore any stray non-numeric junk
            continue

    # Expect exactly 6 numeric values for 6 DOF
    if time_val is None or label is None or len(numeric_vals) != 6:
        return None

    X_pose, Y_pose, Z_pose, X_rot, Y_rot, Z_rot = numeric_vals

    # Derive scenario + take from label, e.g. "fast_pan_tripod_01"
    m = re.match(r"(.+)_0*(\d+)$", label)
    if m:
        scenario = m.group(1)
        take = int(m.group(2))
    else:
        scenario = label
        take = None

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

    # Process both .log and .txt files under ./data
    for pattern in ("*.log", "*.txt"):
        for file_path in LOG_DIR.glob(pattern):
            stem = file_path.stem

            # Only take files that match the scenario bases
            if not any(stem.startswith(base) for base in SCENARIO_BASES):
                continue

            print(f"Parsing {file_path.name} ...")

            with file_path.open(encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "TRACKDATA" not in line:
                        continue
                    row = parse_track_line(line)
                    if row is None:
                        continue
                    row["file"] = stem
                    rows.append(row)

    if not rows:
        raise RuntimeError(
            "No TRACKDATA rows found. Check data/ and scenario file names."
        )

    df = pd.DataFrame(rows)

    # Sort for sanity: by scenario, then take, then time
    df.sort_values(["scenario", "take", "time"], inplace=True)

    # Save combined dataset
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved {len(df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
