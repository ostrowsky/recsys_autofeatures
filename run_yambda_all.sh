#!/bin/sh
# Yambda pipeline, starts after the RetailRocket queue has finished (CPU is shared).
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe
until grep -q EXIT logs/collect_rr.log 2>/dev/null; do sleep 30; done
$PY experiments/run_baselines.py yambda --trials 25 > logs/baselines_ya.log 2>&1; echo EXIT $? >> logs/baselines_ya.log
$PY experiments/run_autofe.py yambda --grid full --trials 25 > logs/autofe_full_ya.log 2>&1; echo EXIT $? >> logs/autofe_full_ya.log
$PY experiments/run_autofe.py yambda --grid full --cap 2 --tag full_div --trials 25 > logs/autofe_div_ya.log 2>&1; echo EXIT $? >> logs/autofe_div_ya.log
$PY experiments/run_autofe.py yambda --grid coarse --trials 25 > logs/autofe_coarse_ya.log 2>&1; echo EXIT $? >> logs/autofe_coarse_ya.log
$PY experiments/run_neural.py yambda --epochs 4 --trials 25 > logs/neural_ya.log 2>&1; echo EXIT $? >> logs/neural_ya.log
$PY experiments/slices.py yambda > logs/slices_ya.log 2>&1; echo EXIT $? >> logs/slices_ya.log
$PY experiments/collect_results.py yambda > logs/collect_ya.log 2>&1; echo EXIT $? >> logs/collect_ya.log
