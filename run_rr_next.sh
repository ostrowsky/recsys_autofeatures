#!/bin/sh
# Queue after run_rr_all.sh: diversity-capped E2, screening comparison, neural E4/E5.
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe
until grep -q EXIT logs/autofe_coarse_rr.log 2>/dev/null; do sleep 20; done
$PY experiments/run_autofe.py retailrocket --grid full --cap 2 --tag full_div --trials 25 > logs/autofe_div_rr.log 2>&1; echo EXIT $? >> logs/autofe_div_rr.log
$PY experiments/screen_compare.py retailrocket --tag full > logs/screen_compare_rr.log 2>&1; echo EXIT $? >> logs/screen_compare_rr.log
$PY experiments/run_neural.py retailrocket --epochs 4 --trials 25 > logs/neural_rr.log 2>&1; echo EXIT $? >> logs/neural_rr.log
