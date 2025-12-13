import re
from pathlib import Path
import pandas as pd
import os

# --- CONFIGURATION ---
INPUT_DIR = Path("data/validation/raw")
OUTPUT_DIR = Path("data/validation/processed/")


def parse_log_file(file_path):
    print(f"--> Scanning: {file_path.name}")
    if not file_path.exists():
        print(f"    Error: File not found!")
        return None

    parsed_rows = []

    # Regex finds the line, ignoring the prefix (timestamp, LogBlueprintUserMessages, etc.)
    regex_pattern = r"TRACKDATA.*?,(.*)"

    match_count = 0
    error_count = 0

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "TRACKDATA" in line:
                match = re.search(regex_pattern, line)
                if match:
                    try:
                        # 1. Get the raw CSV string part
                        csv_content = match.group(1)

                        # 2. Aggressive Cleaning
                        # Remove literal "\n" (backslash n) often found in UE logs
                        csv_content = csv_content.replace(r"\n", "")
                        # Remove normal newlines
                        csv_content = csv_content.replace("\n", "").replace("\r", "")
                        # Remove quotes
                        csv_content = csv_content.replace('"', "").strip()

                        # 3. Split and Clean individual items
                        # We use a list comprehension to strip every single item
                        parts = [p.strip() for p in csv_content.split(",")]

                        # 4. Filter empty strings (caused by trailing commas)
                        parts = [p for p in parts if p]

                        # We need at least 14 columns
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
                        else:
                            if error_count < 1:
                                print(
                                    f"    Skipping malformed line (len={len(parts)}). Content sample: {csv_content[:20]}..."
                                )
                            error_count += 1

                    except ValueError as e:
                        if error_count < 3:
                            print(
                                f"    Value Error parsing line: {e} | Content: {csv_content}"
                            )
                        error_count += 1

    print(f"    Found {match_count} valid frames.")
    return pd.DataFrame(parsed_rows)


def main():
    if not OUTPUT_DIR.exists():
        os.makedirs(OUTPUT_DIR)

    if not INPUT_DIR.exists():
        print(f"Error: Input directory '{INPUT_DIR}' does not exist.")
        return

    log_files = list(INPUT_DIR.glob("*.log"))

    if not log_files:
        print(f"No .log files found in {INPUT_DIR}")
        return

    for log_file in log_files:
        df = parse_log_file(log_file)

        if df is not None and not df.empty:
            output_path = OUTPUT_DIR / f"{log_file.stem}_parsed.csv"
            df.to_csv(output_path, index=False)
            print(f"    [OK] Saved to {output_path}")
        else:
            print(f"    [WARNING] No data extracted from {log_file.name}")


if __name__ == "__main__":
    main()
