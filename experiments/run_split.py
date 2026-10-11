"""E2-split: one model per segment instead of one shared feature set.

Motivation (stated before running): E2-seg showed the two segments want different features
(repeat: long user x item memory; new: hours-scale user x attribute activity), and forcing them into
one set trades one segment for the other. Here:

  * segment = whether the user touched the item before the cutoff (repeat) or not (new);
  * features are ranked per segment on the screening sample (same screen as E2-seg);
  * K_repeat, K_new in {32, 64, 128} are chosen jointly on VAL NDCG@10 of the combined ranking
    (each segment model is a binary classifier on its own rows, so the two probability scales are
    both "P(positive | row)" and can be merged in one list);
  * tuning budget equals E1's: 25 Optuna trials in total (13 repeat + 12 new); for each segment the
    objective is the combined val NDCG with the other segment at default params;
  * final scores = 3-seed average per segment. Test is touched once.

usage: python experiments/run_split.py yambda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from datasets import load_dataset  # noqa: E402
from fe_engine import Feat, compute  # noqa: E402
from grammars import auto_grammar  # noqa: E402
from metrics import evaluate, per_query_ndcg  # noqa: E402
from pipeline import BASE_PARAMS, compute_splits, fit_lgb, prepare  # noqa: E402
from run_autofe import subsample  # noqa: E402

KS = (32, 64, 128)


def seen_flag(ds, c: pl.DataFrame) -> np.ndarray:
    labs = tuple(sorted(ds.events["label"].unique().to_list()))
    f = Feat(("user", "item"), labs, "age")
    return ~np.isnan(compute([f], ds.events, c)[f.name].to_numpy())


def gain_rank(Xtr, ytr, Xva, yva, names) -> np.ndarray:
    g = np.zeros(len(names))
    for s in range(3):
        m = fit_lgb(Xtr, ytr, Xva, yva, {"feature_fraction": 0.3, "seed": s, "learning_rate": 0.08}, names)
        g += m.feature_importance("gain", iteration=m.best_iteration)
    return np.argsort(-g)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--screen_train", type=int, default=2500)
    ap.add_argument("--screen_val", type=int, default=1500)
    a = ap.parse_args()
    ds, sp, cc = load_dataset(a.dataset)
    out_dir = ROOT / "experiments" / "results" / ds.name
    t0 = time.time()
    P = prepare(ds, sp, cc)
    feats = auto_grammar(ds.name, "full")
    by_name = {f.name: f for f in feats}

    # ---- screening: per-segment rankings, joint (K_repeat, K_new) on combined val NDCG
    c_tr, _ = subsample(P["c_train"], P["q_train"], a.screen_train, 1)
    c_va, q_va = subsample(P["c_val"], P["q_val"], a.screen_val, 2)
    S = compute_splits(feats, ds, {"c_train": c_tr, "c_val": c_va}, ("train", "val"))
    names = S["train"].columns
    Xtr, Xva = S["train"].to_numpy(), S["val"].to_numpy()
    del S
    ytr, yva = c_tr["y"].to_numpy(), c_va["y"].to_numpy()
    st, sv = seen_flag(ds, c_tr), seen_flag(ds, c_va)
    print(f"[split] screening {Xtr.shape}, repeat share train {st.mean():.2f}  ({time.time() - t0:.0f}s)")
    rank = {True: gain_rank(Xtr[st], ytr[st], Xva[sv], yva[sv], names),
            False: gain_rank(Xtr[~st], ytr[~st], Xva[~sv], yva[~sv], names)}
    pred = {}
    for seg in (True, False):
        mt, mv = (st, sv) if seg else (~st, ~sv)
        for k in KS:
            idx = rank[seg][:k]
            ps = []
            for s in range(2):
                m = fit_lgb(Xtr[mt][:, idx], ytr[mt], Xva[mv][:, idx], yva[mv], {"seed": s})
                ps.append(m.predict(Xva[mv][:, idx], num_iteration=m.best_iteration))
            pred[(seg, k)] = np.mean(ps, 0)
    grid = {}
    for kr in KS:
        for kn in KS:
            sc = np.empty(len(yva))
            sc[sv], sc[~sv] = pred[(True, kr)], pred[(False, kn)]
            grid[(kr, kn)] = evaluate(c_va, sc, q_va, ks=(10,))["ndcg@10"]
            print(f"[split] K_repeat={kr:3d} K_new={kn:3d}: val ndcg@10 {grid[(kr, kn)]:.4f}")
    best = max(grid.values())
    kr, kn = min((k for k, v in grid.items() if v >= best - 0.002), key=lambda t: t[0] + t[1])
    chosen = {True: [names[i] for i in rank[True][:kr]], False: [names[i] for i in rank[False][:kn]]}
    print(f"[split] chose K_repeat={kr} K_new={kn}  ({time.time() - t0:.0f}s)")
    del Xtr, Xva

    # ---- full computation of the union, per-segment tuning with a shared 25-trial budget
    union = list(dict.fromkeys(chosen[True] + chosen[False]))
    gram = [by_name[n] for n in union if n in by_name]
    X = compute_splits(gram, ds, P)
    cols = X["train"].columns
    col = {n: i for i, n in enumerate(cols)}
    seg_idx = {s: [col[n] for n in chosen[s]] for s in (True, False)}
    M = {s: X[s].to_numpy() for s in ("train", "val", "test")}
    y = {s: P[f"c_{s}"]["y"].to_numpy() for s in ("train", "val")}
    seen = {s: seen_flag(ds, P[f"c_{s}"]) for s in ("train", "val", "test")}

    def fit_seg(seg: bool, params: dict, seed: int = 0):
        it = seg_idx[seg]
        mt, mv = (seen["train"], seen["val"]) if seg else (~seen["train"], ~seen["val"])
        return fit_lgb(M["train"][mt][:, it], y["train"][mt], M["val"][mv][:, it], y["val"][mv],
                       {**params, "seed": seed})

    def seg_pred(m, seg: bool, split: str) -> np.ndarray:
        mm = seen[split] if seg else ~seen[split]
        return m.predict(M[split][mm][:, seg_idx[seg]], num_iteration=m.best_iteration)

    default = {s: seg_pred(fit_seg(s, {}), s, "val") for s in (True, False)}
    tuned = {}
    budget = {True: (a.trials + 1) // 2, False: a.trials // 2}
    for seg in (True, False):
        def obj(trial: optuna.Trial) -> float:
            p = dict(learning_rate=trial.suggest_float("learning_rate", 0.04, 0.2, log=True),
                     num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
                     min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 20, 1000, log=True),
                     feature_fraction=trial.suggest_float("feature_fraction", 0.3, 1.0),
                     bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
                     lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 30, log=True))
            sc = np.empty(len(y["val"]))
            sc[seen["val"]] = seg_pred(fit_seg(seg, p), seg, "val") if seg else default[True]
            sc[~seen["val"]] = default[False] if seg else seg_pred(fit_seg(seg, p), seg, "val")
            v = evaluate(P["c_val"], sc, P["q_val"], ks=(10,))["ndcg@10"]
            print(f"    [{'repeat' if seg else 'new'}] trial {trial.number}: val ndcg@10 {v:.4f}", flush=True)
            return v
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        stdy = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
        stdy.enqueue_trial({k: BASE_PARAMS[k] for k in ("learning_rate", "num_leaves", "min_data_in_leaf",
                                                          "feature_fraction", "bagging_fraction", "lambda_l2")})
        stdy.optimize(obj, n_trials=budget[seg])
        tuned[seg] = stdy.best_params

    out = {split: np.empty(len(seen[split])) for split in ("val", "test")}
    for seg in (True, False):
        models = [fit_seg(seg, tuned[seg], seed) for seed in range(3)]
        for split in ("val", "test"):
            out[split][seen[split] if seg else ~seen[split]] = np.mean([seg_pred(m, seg, split) for m in models], 0)
    vnd = evaluate(P["c_val"], out["val"], P["q_val"], ks=(10,))["ndcg@10"]
    met = evaluate(P["c_test"], out["test"], P["q_test"], proba=out["test"])
    per_query_ndcg(P["c_test"], out["test"], P["q_test"]).write_parquet(out_dir / "pq_E2_split.parquet")
    np.save(out_dir / "scores_E2_split.npy", out["test"])
    res = {**met, "val_ndcg@10": vnd, "n_features": len(cols), "k_repeat": kr, "k_new": kn,
           "k_grid": {f"{a_}x{b_}": v for (a_, b_), v in grid.items()},
           "chosen_repeat": chosen[True], "chosen_new": chosen[False], "params": {str(k): v for k, v in tuned.items()},
           "seconds": time.time() - t0}
    (out_dir / "E2_split.json").write_text(json.dumps(res, indent=2))
    print("[split]", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in met.items()}, "val", round(vnd, 4))
    print(f"[split] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
