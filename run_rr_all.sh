#!/bin/sh
# Sequential RetailRocket pipeline (CPU is shared, so no parallel jobs).
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe
$PY experiments/run_baselines.py retailrocket --trials 25 > logs/baselines_rr.log 2>&1; echo EXIT $? >> logs/baselines_rr.log
$PY experiments/run_autofe.py retailrocket --grid full --trials 25 > logs/autofe_full_rr.log 2>&1; echo EXIT $? >> logs/autofe_full_rr.log
$PY experiments/run_autofe.py retailrocket --grid coarse --trials 25 > logs/autofe_coarse_rr.log 2>&1; echo EXIT $? >> logs/autofe_coarse_rr.log
