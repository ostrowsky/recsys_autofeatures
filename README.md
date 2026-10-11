# Automatic feature generation for RecSys ranking

**Full report (Russian): [reports/REPORT.md](reports/REPORT.md).** Every number below comes from a
script in this repository; per-dataset tables are in `experiments/results/<dataset>/`.

## Idea

Almost every hand-crafted RecSys feature has one shape:
**aggregate( events of type S, in a time window or with time decay, grouped by a path from the
candidate row to past events )** — "how many times did *this user* *buy* items of *this category* in
the last *7 days*". We write that shape as a **grammar**, enumerate it (~1.7–1.9k features per dataset),
compute every feature leak-free with one as-of pass per (path, event-set) group, and let model-based
search choose the subset. The only dataset-specific input is the schema. Windows form a dense grid and
decay has a continuous half-life, so "7 days or 10 days?" stops being a human decision.

## Results (test NDCG@10, paired bootstrap 95% CI vs the tuned hand-crafted baseline E1)

| | RetailRocket (e-commerce) | Yambda (music) |
|---|---|---|
| E0 user history by recency | 0.3134 | 0.1742 |
| **E1 38/36 hand-crafted features + Optuna-tuned LightGBM** | **0.3281** | **0.2772** |
| E2 grammar, windows 1/7/30 only | 0.3277 (−0.0004 [−0.0011, +0.0005]) | 0.2805 (+0.0033 [−0.0003, +0.0066]) |
| **E2 grammar, full time grid** | 0.3281 (−0.0000 [−0.0010, +0.0009]) | **0.2885 (+0.0113 [+0.0069, +0.0157])** |
| E2-split (primary by pre-declared "best val" rule) | 0.3279 (−0.0002 [−0.0013, +0.0010]) | 0.2853 (+0.0080 [+0.0035, +0.0122]) |
| E4 neural ranker with learned time kernels | 0.3237 (−0.0044 [−0.0058, −0.0031]) | 0.2306 (−0.0466 [−0.0541, −0.0395]) |

* Yambda: every full-grid variant of the automatic pipeline beats the tuned hand-crafted booster;
  most of the gain is the automatic choice of time scale (full grid vs 1/7/30: +0.0081 [+0.0041, +0.0122]).
* RetailRocket: a tie, bounded by the candidate generator (repeats are solved by recency, new items have
  10% candidate recall).
* Weak spot found on both datasets: selection by overall NDCG favours repeat-item features and slightly
  loses on items new to the user; segment-aware selection removes that loss at a cost on repeats.

![Difference to E1](reports/figures/fig_delta.png)

## Protocol

Query = (user, day d) with history before d; cutoff = start of d; features and candidates use only
`ts < cutoff`; split by day (train | val for all selection and tuning | test looked at once); identical
candidate rows for every method; ideal DCG counts all targets. Leakage tests in `tests/test_leakage.py`
(engine vs brute force, future-mutation invariance, a canary leak that must be caught).

## Data (not included in this repository)

* **RetailRocket** — original source: Kaggle `retailrocket/ecommerce-dataset`. Put `events.csv`,
  `category_tree.csv`, `item_properties_part1.csv`, `item_properties_part2.csv` into
  `data/raw/retailrocket/`. (Our runs used an identical-size mirror on Hugging Face.)
* **Yambda-50M** — Hugging Face `yandex/yambda`: `flat/50m/multi_event.parquet`,
  `artist_item_mapping.parquet`, `album_item_mapping.parquet` into `data/raw/yambda/`.

The datasets are distributed by their owners under their own terms; this repository contains only code
and derived metrics.

## Reproduce

Python 3.14, CPU only. Raw test-score arrays (`scores_*.npy`) are not versioned; `experiments/rescore.py`
and the runners regenerate them.

```
python -m venv .venv
.venv\Scripts\pip install numpy pandas pyarrow polars duckdb lightgbm catboost optuna scikit-learn scipy matplotlib tqdm pytest
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
# exact versions used: requirements.lock.txt
python src/rr_prep.py && python src/yambda_prep.py
python -m pytest tests
sh run_rr_all.sh && sh run_rr_next.sh && sh run_rr_slices.sh
sh run_yambda_all.sh && sh run_seg.sh && sh run_split.sh
python experiments/make_figures.py
```

| path | what |
|---|---|
| `src/fe_engine.py` | grammar, as-of computation, `to_sql()` |
| `src/grammars.py` | per-dataset grammar from the schema only |
| `src/protocol.py` | queries, split, candidate generator |
| `src/hand_features.py` | E1 hand-crafted sets |
| `src/neural.py` | E4 model |
| `experiments/run_*.py` | E0–E5 runners; `slices.py`, `collect_results.py`, `make_figures.py` |
