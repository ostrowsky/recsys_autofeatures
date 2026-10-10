"""Shared training / evaluation helpers."""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl

from metrics import evaluate, per_query_ndcg
from protocol import CandidateCfg, CandidateGenerator, Dataset, Split, build_queries, candidate_recall

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"

BASE_PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=63, min_data_in_leaf=100,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                   verbose=-1, num_threads=4, max_bin=63, seed=0)


def prepare(ds: Dataset, sp: Split, cc: CandidateCfg) -> dict:
    d = CACHE / ds.name
    d.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    if (d / "cands_test.parquet").exists():
        for s in ("train", "val", "test"):
            out[f"q_{s}"] = pl.read_parquet(d / f"queries_{s}.parquet")
            out[f"c_{s}"] = pl.read_parquet(d / f"cands_{s}.parquet")
        out["meta"] = json.loads((d / "meta.json").read_text())
        return out
    t = time.time()
    qs = build_queries(ds, sp)
    gen = CandidateGenerator(ds, cc)
    meta = {"n_queries_all": {}, "candidate_recall": {}}
    for s in ("train", "val", "test"):
        q = qs[s]
        c = gen.build(q)
        q.write_parquet(d / f"queries_{s}.parquet")
        c.write_parquet(d / f"cands_{s}.parquet")
        out[f"q_{s}"], out[f"c_{s}"] = q, c
        meta["n_queries_all"][s] = qs[s + "_n_all"]
        meta["candidate_recall"][s] = candidate_recall(c, q)
        print(f"[prepare] {s}: {q.height:,} queries (of {qs[s + '_n_all']:,}), {c.height:,} candidate rows, "
              f"recall {meta['candidate_recall'][s]['recall_all_candidates']:.3f}  ({time.time() - t:.0f}s)")
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    out["meta"] = meta
    return out


def src_features(c: pl.DataFrame) -> pl.DataFrame:
    """Candidate-source indicators: given to EVERY method (same candidate generator)."""
    return c.select([((pl.col("src") & b) > 0).cast(pl.Float32).alias(f"src_{n}")
                     for n, b in (("hist", 1), ("covis", 2), ("catpop", 4), ("glob", 8))])


def safe_names(names: list[str] | None) -> list[str] | str:
    """LightGBM rejects JSON-special characters; DSL names are kept outside the booster."""
    return [f"f{i}" for i in range(len(names))] if names else "auto"


def compute_splits(feats: list, ds: Dataset, P: dict, splits=("train", "val", "test")) -> dict[str, pl.DataFrame]:
    """One engine pass over the concatenated candidate rows of all splits (each split keeps
    its own cutoffs, so this is identical to computing them separately, only cheaper)."""
    from fe_engine import compute
    cs = [P[f"c_{s}"] for s in splits]
    F = compute(feats, ds.events, pl.concat(cs, how="vertical_relaxed"))
    out, o = {}, 0
    for s, c in zip(splits, cs):
        out[s] = pl.concat([src_features(c), F.slice(o, c.height)], how="horizontal")
        o += c.height
    return out


def fit_lgb(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, params: dict | None = None,
            names: list[str] | None = None, rounds: int = 1000) -> lgb.Booster:
    p = {**BASE_PARAMS, **(params or {})}
    dtr = lgb.Dataset(Xtr, ytr, feature_name=safe_names(names), free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    return lgb.train(p, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(30, verbose=False)])


def tune_lgb(Xtr, ytr, Xva, yva, c_va: pl.DataFrame, q_va: pl.DataFrame, n_trials: int = 25,
             names=None, seed: int = 0) -> tuple[dict, float]:
    """Optuna over LightGBM params, objective = val NDCG@10. Same budget for every method."""
    def obj(trial: optuna.Trial) -> float:
        p = dict(learning_rate=trial.suggest_float("learning_rate", 0.04, 0.2, log=True),
                 num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
                 min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 20, 1000, log=True),
                 feature_fraction=trial.suggest_float("feature_fraction", 0.3, 1.0),
                 bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
                 lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 30, log=True))
        m = fit_lgb(Xtr, ytr, Xva, yva, p, names)
        trial.set_user_attr("best_iter", m.best_iteration)
        v = evaluate(c_va, m.predict(Xva, num_iteration=m.best_iteration), q_va, ks=(10,))["ndcg@10"]
        print(f"    trial {trial.number}: val ndcg@10 {v:.4f}  iters {m.best_iteration}", flush=True)
        return v

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.enqueue_trial({k: BASE_PARAMS[k] for k in ("learning_rate", "num_leaves", "min_data_in_leaf",
                                                    "feature_fraction", "bagging_fraction", "lambda_l2")})
    st.optimize(obj, n_trials=n_trials)
    return st.best_params, st.best_value


def final_eval(Xtr, ytr, Xva, yva, Xte, c_te, q_te, params, names=None, seeds=(0, 1, 2)) -> tuple[dict, pl.DataFrame, np.ndarray]:
    """Average of seeds. Returns metrics, per-query NDCG@10 and test scores."""
    preds = []
    for s in seeds:
        m = fit_lgb(Xtr, ytr, Xva, yva, {**params, "seed": s}, names)
        preds.append(m.predict(Xte, num_iteration=m.best_iteration))
    p = np.mean(preds, axis=0)
    return evaluate(c_te, p, q_te, proba=p), per_query_ndcg(c_te, p, q_te), p
