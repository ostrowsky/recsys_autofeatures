#!/bin/sh
# Resume the pre-registered rolling-origin runs; skips any step whose result file already exists.
cd /d/Projects/recsys_autofeatures
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe
R=experiments/results
echo "START $(date)" >> logs/rolling_resume.log
for w in yambda_w1 yambda_w2 yambda_w3; do
  [ -f $R/$w/baselines.json ] || { $PY experiments/run_baselines.py $w --trials 25 > logs/roll_base_$w.log 2>&1; echo EXIT $? >> logs/roll_base_$w.log; }
  [ -f $R/$w/E1plus_hand_lgb.json ] || { $PY experiments/run_hand_plus.py $w > logs/roll_plus_$w.log 2>&1; echo EXIT $? >> logs/roll_plus_$w.log; }
  [ -f $R/$w/E2_full.json ] || { $PY experiments/run_autofe.py $w --grid full --trials 25 > logs/roll_e2_$w.log 2>&1; echo EXIT $? >> logs/roll_e2_$w.log; }
  echo "$w done $(date)" >> logs/rolling_resume.log
done
$PY experiments/rolling_summary.py > logs/rolling_summary.log 2>&1; echo EXIT $? >> logs/rolling_summary.log
echo "ALLDONE $(date)" >> logs/rolling_resume.log
