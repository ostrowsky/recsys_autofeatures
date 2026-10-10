"""Apples-to-apples check on the SCREENING subsample used by E2: E1 hand features vs
E2-chosen features, identical rows, identical untuned LightGBM, 3 seeds.
Answers: is E2's screening-stage number higher because of the features, or because of the subset?

usage: python experiments/screen_compare.py retailrocket [--tag full]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from datasets import load_dataset  # noqa: E402
from grammars import auto_grammar  # noqa: E402
from hand_features import hand_features  # noqa: E402
from metrics import evaluate  # noqa: E402
from pipeline import compute_splits, fit_lgb, prepare  # noqa: E402
from run_autofe import subsample  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--screen_train", type=int, default=2500)
    ap.add_argument("--screen_val", type=int, default=1500)
    a = ap.parse_args()
    ds, sp, cc = load_dataset(a.dataset)
    out_dir = ROOT / "experiments" / "results" / ds.name
    P = prepare(ds, sp, cc)
    c_tr, q_tr = subsample(P["c_train"], P["q_train"], a.screen_train, 1)
    c_va, q_va = subsample(P["c_val"], P["q_val"], a.screen_val, 2)
    sub = {"c_train": c_tr, "c_val": c_va}
    e2 = json.loads((out_dir / f"E2_{a.tag}.json").read_text())
    by_name = {f.name: f for f in auto_grammar(ds.name, a.tag if a.tag in ("full", "coarse") else "full")}
    sets = {"E1_hand": hand_features(ds.name), f"E2_{a.tag}": [by_name[n] for n in e2["chosen"] if n in by_name]}
    res = {}
    for name, feats in sets.items():
        S = compute_splits(feats, ds, sub, ("train", "val"))
        Xtr, Xva = S["train"].to_numpy(), S["val"].to_numpy()
        vals = []
        for s in range(3):
            m = fit_lgb(Xtr, c_tr["y"].to_numpy(), Xva, c_va["y"].to_numpy(), {"seed": s}, S["train"].columns)
            vals.append(evaluate(c_va, m.predict(Xva, num_iteration=m.best_iteration), q_va, ks=(10,))["ndcg@10"])
        res[name] = {"mean": float(np.mean(vals)), "seeds": vals, "n_features": Xtr.shape[1]}
        print(name, res[name])
    (out_dir / f"screen_compare_{a.tag}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
