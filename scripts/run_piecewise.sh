#!/usr/bin/env bash
set -e

echo "=== [run_piecewise] Step 1/3: parsing raw logs -> tracking_logs.csv"
python parser.py

echo "=== [run_piecewise] Step 2/3: computing kinematics -> tracking_derivatives.csv"
python kinematics.py

echo "=== [run_piecewise] Step 3/3: piecewise sigma model -> extend derivatives + config JSON"
python piecewise_sigma_model.py

echo "[run_piecewise] All steps finished."