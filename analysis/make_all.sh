#!/usr/bin/env bash
# Reproduce every table, figure and number in the paper from the saved results (CPU only, about 2 minutes).
set -e
cd "$(dirname "$0")"
python audit_table.py
python design_checks.py
python item_analysis.py
python power_table.py
python figures.py
echo "Outputs written to ../outputs/tables and ../outputs/figures"
