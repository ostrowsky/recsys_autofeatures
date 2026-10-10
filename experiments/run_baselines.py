"""Stage 1: queries + candidates, E0 (non-learned baselines), E1 (hand features + tuned LightGBM).

usage: python experiments/run_baselines.py retailrocket [--trials 25]
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
from hand_features import hand_features  # noqa: E402
from metrics import evaluate, per_query_ndcg  # noqa: E402
from pipeline import compute_splits, final_eval, prepare, tune_lgb  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--trials", type=int, default=25)
    a = ap.parse_args()
    ds, sp, cc = load_dataset(a.dataset)
    out_dir = ROOT / "experiments" / "results" / ds.name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    P = prepare(ds, sp, cc)
    print(json.dumps(P["meta"], indent=1))

    res: dict = {}
    # ---------------- E0: non-learned baselines on the same candidates
    c_te, q_te = P["c_test"], P["q_test"]
    from fe_engine import D, Feat
    simple = compute([Feat(("item",), (0, 1, 2), "cnt", 7 * D), Feat(("user", "item"), (0, 1, 2), "tsl")],
                     ds.events, c_te)
    pop = simple[simple.columns[0]].to_numpy()
    tsl = simple[simple.columns[1]].to_numpy()
    rec = np.where(np.isnan(tsl), -1e9 + pop, -tsl)            # own history by recency, then popularity
    for name, s in (("E0_popularity_7d", pop), ("E0_history_recency", rec)):
        res[name] = evaluate(c_te, s, q_te)
        per_query_ndcg(c_te, s, q_te).write_parquet(out_dir / f"pq_{name}.parquet")
        np.save(out_dir / f"scores_{name}.npy", s)
        print(name, {k: round(v, 4) for k, v in res[name].items()})

    # ---------------- E1: hand features + tuned LightGBM
    feats = hand_features(ds.name)
    X = compute_splits(feats, ds, P)
    names = X["train"].columns
    Xtr, Xva, Xte = (X[s].to_numpy() for s in ("train", "val", "test"))
    ytr, yva = P["c_train"]["y"].to_numpy(), P["c_val"]["y"].to_numpy()
    print(f"E1 features: {len(names)}  rows train {len(ytr):,} val {len(yva):,} test {Xte.shape[0]:,}  "
          f"({time.time() - t0:.0f}s)")
    best, val_ndcg = tune_lgb(Xtr, ytr, Xva, yva, P["c_val"], P["q_val"], n_trials=a.trials, names=names)
    m, pq, sc = final_eval(Xtr, ytr, Xva, yva, Xte, c_te, q_te, best, names)
    np.save(out_dir / "scores_E1_hand_lgb.npy", sc)
    res["E1_hand_lgb"] = {**m, "val_ndcg@10": val_ndcg, "n_features": len(names), "params": best}
    pq.write_parquet(out_dir / "pq_E1_hand_lgb.parquet")
    print("E1_hand_lgb", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["E1_hand_lgb"].items()})
    (out_dir / "baselines.json").write_text(json.dumps({"meta": P["meta"], "results": res}, indent=2))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
