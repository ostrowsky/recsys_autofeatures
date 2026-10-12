#!/bin/sh
# Rolling-origin check on Yambda (pre-registered in reports/prereg_rolling.md).
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe
for w in yambda_w1 yambda_w2 yambda_w3; do
  $PY experiments/run_baselines.py $w --trials 25 > logs/roll_base_$w.log 2>&1; echo EXIT $? >> logs/roll_base_$w.log
  $PY experiments/run_hand_plus.py $w > logs/roll_plus_$w.log 2>&1; echo EXIT $? >> logs/roll_plus_$w.log
  $PY experiments/run_autofe.py $w --grid full --trials 25 > logs/roll_e2_$w.log 2>&1; echo EXIT $? >> logs/roll_e2_$w.log
done
$PY experiments/rolling_summary.py > logs/rolling_summary.log 2>&1; echo EXIT $? >> logs/rolling_summary.log
