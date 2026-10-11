"""E1+: strengthened hand-crafted set (see hand_features.hand_features_yambda_plus), same tuning budget
and evaluation as E1. usage: python experiments/run_hand_plus.py yambda"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_dataset  # noqa: E402
from hand_features import hand_features_yambda_plus  # noqa: E402
from pipeline import compute_splits, final_eval, prepare, tune_lgb  # noqa: E402


def main(name: str) -> None:
    ds, sp, cc = load_dataset(name)
    out_dir = ROOT / "experiments" / "results" / ds.name
    t0 = time.time()
    P = prepare(ds, sp, cc)
    X = compute_splits(hand_features_yambda_plus(), ds, P)
    fn = X["train"].columns
    Xtr, Xva, Xte = (X[s].to_numpy() for s in ("train", "val", "test"))
    ytr, yva = P["c_train"]["y"].to_numpy(), P["c_val"]["y"].to_numpy()
    params, vnd = tune_lgb(Xtr, ytr, Xva, yva, P["c_val"], P["q_val"], n_trials=25, names=fn)
    met, pq, sc = final_eval(Xtr, ytr, Xva, yva, Xte, P["c_test"], P["q_test"], params, fn)
    pq.write_parquet(out_dir / "pq_E1plus_hand_lgb.parquet")
    np.save(out_dir / "scores_E1plus_hand_lgb.npy", sc)
    res = {**met, "val_ndcg@10": vnd, "n_features": len(fn), "params": params, "seconds": time.time() - t0}
    (out_dir / "E1plus_hand_lgb.json").write_text(json.dumps(res, indent=2))
    print("E1plus", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in met.items()}, "val", round(vnd, 4))


if __name__ == "__main__":
    main(sys.argv[1])
