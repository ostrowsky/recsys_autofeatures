"""E2: automatic feature generation = grammar enumeration + model-based selection.

Search (uses train + val only; test is touched once at the end):
  1. enumerate the grammar (~1-2k features) and compute it on a SCREENING sample of train/val
     queries;
  2. importance screen: LightGBM with strong column subsampling, 3 seeds, mean gain;
  3. nested top-K sets, K in {16, 32, 64, 128, 256}: val NDCG@10 picks K (smallest K within
     0.002 of the best, to prefer the cheaper model);
  4. compute the chosen features on the full train/val/test, same Optuna budget as E1,
     3-seed final evaluation.

usage: python experiments/run_autofe.py retailrocket [--grid full|coarse] [--trials 25]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_dataset  # noqa: E402
from fe_engine import compute  # noqa: E402
from grammars import auto_grammar  # noqa: E402
from metrics import evaluate  # noqa: E402
from pipeline import compute_splits, fit_lgb, final_eval, prepare, tune_lgb  # noqa: E402

KS = (16, 32, 64, 128, 256)


def subsample(c: pl.DataFrame, q: pl.DataFrame, n: int, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    if q.height <= n:
        return c, q
    qs = q.sample(n, seed=seed).sort("qid")
    return c.filter(pl.col("qid").is_in(qs["qid"].implode())), qs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--grid", default="full")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--screen_train", type=int, default=4000)
    ap.add_argument("--screen_val", type=int, default=2000)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tag = a.tag or a.grid
    ds, sp, cc = load_dataset(a.dataset)
    out_dir = ROOT / "experiments" / "results" / ds.name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    P = prepare(ds, sp, cc)
    feats = auto_grammar(ds.name, a.grid)
    by_name = {f.name: f for f in feats}
    print(f"[E2-{tag}] grammar size {len(feats)}")

    # 1. screening sample
    c_tr, q_tr = subsample(P["c_train"], P["q_train"], a.screen_train, 1)
    c_va, q_va = subsample(P["c_val"], P["q_val"], a.screen_val, 2)
    S = compute_splits(feats, ds, {"c_train": c_tr, "c_val": c_va}, ("train", "val"))
    Ftr, Fva = S["train"], S["val"]
    del S
    names = Ftr.columns
    Xtr, Xva = Ftr.to_numpy(), Fva.to_numpy()
    del Ftr, Fva
    ytr, yva = c_tr["y"].to_numpy(), c_va["y"].to_numpy()
    print(f"[E2-{tag}] screening matrix train {Xtr.shape} val {Xva.shape}  ({time.time() - t0:.0f}s)")

    # 2. importance screen
    gain = np.zeros(len(names))
    for s in range(3):
        m = fit_lgb(Xtr, ytr, Xva, yva, {"feature_fraction": 0.3, "seed": s, "learning_rate": 0.08}, names)
        gain += m.feature_importance("gain", iteration=m.best_iteration)
    rank = np.argsort(-gain)
    m_all = evaluate(c_va, m.predict(Xva, num_iteration=m.best_iteration), q_va, ks=(10,))["ndcg@10"]
    print(f"[E2-{tag}] all-features val ndcg@10 {m_all:.4f}  ({time.time() - t0:.0f}s)")

    # 3. nested top-K
    curve = {}
    for k in KS:
        idx = rank[:k]
        vals = []
        for s in range(2):
            m = fit_lgb(Xtr[:, idx], ytr, Xva[:, idx], yva, {"seed": s}, [names[i] for i in idx])
            vals.append(evaluate(c_va, m.predict(Xva[:, idx], num_iteration=m.best_iteration), q_va,
                                 ks=(10,))["ndcg@10"])
        curve[k] = float(np.mean(vals))
        print(f"[E2-{tag}] top-{k}: val ndcg@10 {curve[k]:.4f}")
    best = max(curve.values())
    k_star = min(k for k, v in curve.items() if v >= best - 0.002)
    chosen = [names[i] for i in rank[:k_star]]
    print(f"[E2-{tag}] chose K={k_star}  ({time.time() - t0:.0f}s)")
    del Xtr, Xva

    # 4. full computation of the chosen features + same tuning budget as E1
    gram = [by_name[n] for n in chosen if n in by_name]
    X = compute_splits(gram, ds, P)
    fn = X["train"].columns
    Xtr, Xva, Xte = (X[s].to_numpy() for s in ("train", "val", "test"))
    ytr, yva = P["c_train"]["y"].to_numpy(), P["c_val"]["y"].to_numpy()
    params, val_ndcg = tune_lgb(Xtr, ytr, Xva, yva, P["c_val"], P["q_val"], n_trials=a.trials, names=fn)
    met, pq, _ = final_eval(Xtr, ytr, Xva, yva, Xte, P["c_test"], P["q_test"], params, fn)
    pq.write_parquet(out_dir / f"pq_E2_{tag}.parquet")
    res = {**met, "val_ndcg@10": val_ndcg, "n_features": len(fn), "grammar_size": len(feats), "k_curve": curve,
           "k_star": k_star, "params": params, "chosen": chosen, "top_features": chosen[:40],
           "top_gain": [float(gain[i]) for i in rank[:40]], "sql_examples": [by_name[n].to_sql() for n in chosen[:5]
                                                                             if n in by_name and hasattr(by_name[n], "to_sql")],
           "seconds": time.time() - t0}
    (out_dir / f"E2_{tag}.json").write_text(json.dumps(res, indent=2))
    print(f"[E2-{tag}]", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in met.items()})
    print(f"[E2-{tag}] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
