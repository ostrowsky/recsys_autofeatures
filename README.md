# Automatic feature generation for RecSys ranking

> Status: work in progress (deadline 2026-10-25). Results tables are filled from
> `experiments/results/*/*.json`; every number in this file is produced by a script in this repo.

## Idea in one paragraph

Almost every hand-crafted RecSys feature has the same shape:
**aggregate( events of type S, inside a time window or with time decay, grouped by a path from the
candidate row to a set of past events )** — e.g. "number of purchases (S) in the last 7 days (window)
by this user on items of this category (path user→item→category)". We write that shape down as a
small **grammar**, enumerate it (≈1–2k features per dataset), compute every feature leak-free with a
single as-of pass per (path, event-type) group, and let a model-based search pick the subset. The only
dataset-specific input is the schema (which columns are item attributes, which label codes exist).
"7 days or 10 days?" stops being a human decision: windows form a dense grid and time decay has a
continuous half-life.

Second line: a neural ranker that **learns** the same aggregates (learnable exponential time kernels
on exact user–item / user–attribute matches + target attention over the user's history), and a hybrid
that feeds its score to the booster.

## Protocol (fixed before any model was fitted)

* Query = (user, day d) where the user is active on d and has history before d (warm users).
* Cutoff = start of day d; all features and candidates use only events with `ts < cutoff`.
* Targets = items the user interacts with on day d; graded relevance (view 1 / cart 2 / buy 3).
* Split by day: train 28 d | val 14 d (feature selection, tuning, early stopping) | test 14 d (looked at once).
* Same candidate rows for every method (history, co-visitation, category popularity, global popularity).
  Metrics' ideal DCG / recall count **all** targets, including those the candidates missed.
* Leakage tests (`tests/test_leakage.py`): engine == brute force; rewriting the future changes nothing;
  a deliberately leaky canary is caught by the same check.

## Experiments

| ID | Method |
|----|--------|
| E0 | popularity; user's own history by recency |
| E1 | 34 hand-crafted features (windows 1/7/30 d) + LightGBM, Optuna-tuned |
| E2 | grammar (full grid) + model-based selection, same tuning budget |
| E2-coarse | ablation: same grammar with only 1/7/30-day windows, no decay |
| E4 | neural ranker with learned temporal kernels + target attention |
| E5 | hybrid: E2 features + out-of-fold E4 score in LightGBM |

## Reproduce

```
python -m venv .venv && .venv\Scripts\pip install -r requirements.lock.txt
python src/rr_prep.py && python src/yambda_prep.py
python -m pytest tests
python experiments/run_baselines.py retailrocket
python experiments/run_autofe.py retailrocket --grid full
python experiments/run_autofe.py retailrocket --grid coarse
python experiments/run_neural.py retailrocket
```
