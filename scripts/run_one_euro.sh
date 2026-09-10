#!/usr/bin/env bash
set -euo pipefail

echo "[run_one_euro] Step 1/3: parsing raw logs -> tracking_logs.csv"
python parser.py

echo "[run_one_euro] Step 2/3: computing kinematics -> tracking_derivatives.csv"
python kinematics.py

echo "[run_one_euro] Step 3/3: One Euro filter model -> filtered CSV + config JSON"
python one_euro_filter_model.py

echo "[run_one_euro] All steps finished."