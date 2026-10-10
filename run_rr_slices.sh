#!/bin/sh
# After run_rr_next.sh: re-create test scores of early runs, then report-only slices.
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe
until grep -q EXIT logs/neural_rr.log 2>/dev/null; do sleep 30; done
$PY experiments/rescore.py retailrocket E0_history_recency E1_hand_lgb E2_full E2_coarse E2_full_div > logs/rescore_rr.log 2>&1; echo EXIT $? >> logs/rescore_rr.log
$PY experiments/slices.py retailrocket > logs/slices_rr.log 2>&1; echo EXIT $? >> logs/slices_rr.log
$PY experiments/collect_results.py retailrocket > logs/collect_rr.log 2>&1; echo EXIT $? >> logs/collect_rr.log
