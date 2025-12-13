#!/usr/bin/env bash
set -euo pipefail


echo "[run_validation] Step 1/2: Parsing validation logs..."
python parse_validation.py

echo "[run_validation] Step 2/2: Generating comparison plots"
python validator.py

echo "[run_validation] All steps finished."