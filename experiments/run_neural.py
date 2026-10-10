"""E4: neural ranker with learned aggregates.  E5: hybrid = E2 features + E4 score in LightGBM.

E5 stacking without leakage: the E4 score used as a TRAIN feature is out-of-fold — train queries
are split into 3 contiguous day blocks, each block is scored by a model fitted on the other two.
Val/test scores come from the model fitted on all train queries.

usage: python experiments/run_neural.py retailrocket [--epochs 4] [--trials 25] [--no_hybrid]
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
from grammars import auto_grammar, schema  # noqa: E402
from metrics import evaluate, per_query_ndcg  # noqa: E402
from neural import HistoryBuilder, NeuralRanker, make_part  # noqa: E402
from pipeline import compute_splits, final_eval, prepare, src_features, tune_lgb  # noqa: E402


def sub(part: dict, mask: np.ndarray) -> dict:
    return {**part, "qidx": part["qidx"][mask], "rows": {k: v[mask] for k, v in part["rows"].items()},
            "bank": part["bank"][mask], "src": part["src"][mask], "y": part["y"][mask]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--no_hybrid", action="store_true")
    a = ap.parse_args()
    ds, sp, cc = load_dataset(a.dataset)
    out_dir = ROOT / "experiments" / "results" / ds.name
    t0 = time.time()
    P = prepare(ds, sp, cc)
    labels = schema(ds.name)["labels"]
    attrs = ds.item_attrs
    hb = HistoryBuilder(ds.events, attrs)
    parts = {s: make_part(hb, ds.events, P[f"q_{s}"], P[f"c_{s}"], attrs, labels,
                          src_features(P[f"c_{s}"]).to_numpy()) for s in ("train", "val", "test")}
    print(f"[E4] tensors ready ({time.time() - t0:.0f}s)")
    n_bank, n_src = parts["train"]["bank"].shape[1], parts["train"]["src"].shape[1]
    n_lab = int(ds.events["label"].max()) + 1
    eval_val = lambda p: evaluate(P["c_val"], p, P["q_val"], ks=(10,))["ndcg@10"]  # noqa: E731

    preds = []
    for seed in range(2):
        nr = NeuralRanker(hb, n_lab, n_bank, n_src, seed=seed)
        nr.fit(parts["train"], parts["val"], epochs=a.epochs, eval_fn=eval_val)
        preds.append((nr.predict(parts["val"]), nr.predict(parts["test"])))
    p_va = np.mean([p[0] for p in preds], 0)
    p_te = np.mean([p[1] for p in preds], 0)
    res = {"E4_neural": {**evaluate(P["c_test"], p_te, P["q_test"], proba=p_te), "val_ndcg@10": eval_val(p_va),
                         "learned_taus_hours": nr.model.log_tau.exp().detach().numpy().tolist(),
                         "seconds": time.time() - t0}}
    per_query_ndcg(P["c_test"], p_te, P["q_test"]).write_parquet(out_dir / "pq_E4_neural.parquet")
    print("E4_neural", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["E4_neural"].items()})

    if not a.no_hybrid:
        # out-of-fold E4 score for train rows, 3 contiguous day blocks
        days = P["c_train"]["day"].to_numpy()
        cuts = np.quantile(np.unique(days), [1 / 3, 2 / 3])
        fold = np.digitize(days, cuts)
        oof = np.zeros(len(days))
        for f in range(3):
            nr = NeuralRanker(hb, n_lab, n_bank, n_src, seed=10 + f)
            nr.fit(sub(parts["train"], fold != f), parts["val"], epochs=a.epochs, eval_fn=eval_val)
            oof[fold == f] = nr.predict(sub(parts["train"], fold == f))
        e2 = json.loads((out_dir / "E2_full.json").read_text())
        by_name = {f.name: f for f in auto_grammar(ds.name, "full")}
        gram = [by_name[n] for n in e2["chosen"] if n in by_name]
        X = compute_splits(gram, ds, P)
        for s, v in (("train", oof), ("val", p_va), ("test", p_te)):
            X[s] = X[s].with_columns(pl.Series("e4_score", v.astype(np.float32)))
        fn = X["train"].columns
        Xtr, Xva, Xte = (X[s].to_numpy() for s in ("train", "val", "test"))
        ytr, yva = P["c_train"]["y"].to_numpy(), P["c_val"]["y"].to_numpy()
        params, vnd = tune_lgb(Xtr, ytr, Xva, yva, P["c_val"], P["q_val"], n_trials=a.trials, names=fn)
        met, pq, _ = final_eval(Xtr, ytr, Xva, yva, Xte, P["c_test"], P["q_test"], params, fn)
        pq.write_parquet(out_dir / "pq_E5_hybrid.parquet")
        res["E5_hybrid"] = {**met, "val_ndcg@10": vnd, "n_features": len(fn), "params": params,
                            "seconds": time.time() - t0}
        print("E5_hybrid", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in met.items()})
    (out_dir / "neural.json").write_text(json.dumps(res, indent=2))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
