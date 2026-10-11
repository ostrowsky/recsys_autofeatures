#!/bin/sh
# E2-split (one model per segment). Design and selection rule committed before any split result.
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe
until grep -q ALLDONE logs/collect2_retailrocket.log 2>/dev/null; do sleep 30; done
$PY experiments/run_split.py yambda --trials 25 > logs/split_ya.log 2>&1; echo EXIT $? >> logs/split_ya.log
$PY experiments/run_split.py retailrocket --trials 25 > logs/split_rr.log 2>&1; echo EXIT $? >> logs/split_rr.log
for d in yambda retailrocket; do $PY experiments/slices.py $d > logs/slices3_$d.log 2>&1; $PY experiments/collect_results.py $d > logs/collect3_$d.log 2>&1; done
echo ALLDONE >> logs/collect3_retailrocket.log
