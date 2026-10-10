"""Ranking + pointwise metrics. Ideal DCG and recall use ALL targets of a query, including the
ones the candidate generator missed, so a method is never rewarded for a narrow candidate set."""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score


def _per_query(cands: pl.DataFrame, score: np.ndarray, queries: pl.DataFrame, k: int) -> pl.DataFrame:
    c = cands.select("qid", "rel").with_columns(pl.Series("s", score))
    c = c.with_columns(pl.col("s").rank("ordinal", descending=True).over("qid").alias("rank"))
    top = c.filter(pl.col("rank") <= k)
    dcg = top.group_by("qid").agg(
        ((2.0 ** pl.col("rel").cast(pl.Float64) - 1) / (pl.col("rank") + 1).cast(pl.Float64).log(2)).sum().alias("dcg"),
        (pl.col("rel") > 0).sum().alias("hits"),
        (pl.col("rel") >= 2).sum().alias("hits_cb"),
    )
    tg = queries.select("qid", "targets").explode("targets").unnest("targets")
    tg = tg.with_columns(pl.col("rel").rank("ordinal", descending=True).over("qid").alias("irank"))
    idcg = tg.group_by("qid").agg(
        ((2.0 ** pl.col("rel").cast(pl.Float64) - 1) / (pl.col("irank") + 1).cast(pl.Float64).log(2))
        .filter(pl.col("irank") <= k).sum().alias("idcg"),
        pl.len().alias("n_t"),
        (pl.col("rel") >= 2).sum().alias("n_cb"),
    )
    q = idcg.join(dcg, on="qid", how="left").fill_null(0)
    return q.with_columns(
        (pl.col("dcg") / pl.col("idcg")).alias("ndcg"),
        (pl.col("hits") / pl.min_horizontal(pl.col("n_t"), pl.lit(k))).alias("recall"),
        pl.when(pl.col("n_cb") > 0).then(pl.col("hits_cb") / pl.min_horizontal(pl.col("n_cb"), pl.lit(k)))
        .otherwise(None).alias("recall_cb"),
    )


def evaluate(cands: pl.DataFrame, score: np.ndarray, queries: pl.DataFrame,
             ks=(10, 20), proba: np.ndarray | None = None) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        q = _per_query(cands, score, queries, k)
        out[f"ndcg@{k}"] = float(q["ndcg"].mean())
        out[f"recall@{k}"] = float(q["recall"].mean())
        out[f"recall_cartbuy@{k}"] = float(q["recall_cb"].mean())
    y = cands["y"].to_numpy()
    out["auc"] = float(roc_auc_score(y, score))
    if proba is not None:
        out["logloss"] = float(log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6)))
    out["base_rate"] = float(y.mean())
    return out


def per_query_ndcg(cands: pl.DataFrame, score: np.ndarray, queries: pl.DataFrame, k: int = 10) -> pl.DataFrame:
    return _per_query(cands, score, queries, k).select("qid", "ndcg").sort("qid")


def paired_bootstrap(a: pl.DataFrame, b: pl.DataFrame, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean difference (b - a) of per-query NDCG with a 95% bootstrap CI over queries."""
    j = a.join(b, on="qid", suffix="_b")
    d = (j["ndcg_b"] - j["ndcg"]).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
