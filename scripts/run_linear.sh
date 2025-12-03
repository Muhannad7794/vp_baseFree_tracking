#!/usr/bin/env bash
set -euo pipefail

echo "[run_linear] Step 1/3: parsing raw logs -> tracking_logs.csv"
python parser.py

echo "[run_linear] Step 2/3: computing kinematics -> tracking_derivatives.csv"
python kinematics.py

echo "[run_linear] Step 3/3: linear sigma model -> extend derivatives + config JSON"
python linear_sigma_model.py

echo "[run_linear] All steps finished."
