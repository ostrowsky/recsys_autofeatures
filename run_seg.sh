#!/bin/sh
# Segment-aware selection (designed 2026-10-11 after the RetailRocket new-item slice).
# Yambda: clean test (no Yambda test result seen when the design was fixed).
# RetailRocket: post-hoc -- motivated by one look at the RR test slice; reported as such.
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe
until grep -q EXIT logs/collect_ya.log 2>/dev/null; do sleep 30; done
$PY experiments/run_autofe.py yambda --grid full --select seg --tag full_seg --trials 25 > logs/autofe_seg_ya.log 2>&1; echo EXIT $? >> logs/autofe_seg_ya.log
$PY experiments/run_autofe.py retailrocket --grid full --select seg --tag full_seg --trials 25 > logs/autofe_seg_rr.log 2>&1; echo EXIT $? >> logs/autofe_seg_rr.log
for d in yambda retailrocket; do $PY experiments/slices.py $d > logs/slices2_$d.log 2>&1; $PY experiments/collect_results.py $d > logs/collect2_$d.log 2>&1; done
echo ALLDONE >> logs/collect2_retailrocket.log
