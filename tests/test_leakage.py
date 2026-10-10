"""Leakage guards for the feature engine.

1. Engine == brute force (`ts < cutoff`) on random rows, for every aggregate.
2. Changing / deleting every event at or after the cutoff does not change any feature value.
3. Canary: a deliberately leaky feature (counts the target day) must be detected by the
   same check that the honest features pass, otherwise the check proves nothing.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fe_engine import D, H, Feat, Ratio, compute  # noqa: E402


def _toy(seed: int = 0, n: int = 4000) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    ev = pl.DataFrame({
        "user": rng.integers(0, 30, n), "item": rng.integers(0, 50, n),
        "ts": np.sort(rng.integers(0, 60 * D, n)), "label": rng.choice([0, 0, 0, 1, 2], n).astype(np.int8),
    }).with_columns((pl.col("item") % 7).alias("cat"))
    rows = pl.DataFrame({
        "user": rng.integers(0, 30, 300), "item": rng.integers(0, 50, 300),
        "cutoff": (rng.integers(5, 60, 300) * D).astype(np.int64),
    }).with_columns((pl.col("item") % 7).alias("cat"))
    return ev, rows


FEATS = [
    Feat(("user",), (0,), "cnt", 7 * D), Feat(("user", "item"), (0, 1, 2), "cnt", 3 * H),
    Feat(("user", "cat"), (1, 2), "dec", 2 * D), Feat(("item",), (0,), "dec", 12 * H),
    Feat(("user", "item"), (0,), "tsl"), Feat(("cat",), (2,), "age"),
    Ratio(Feat(("item",), (1, 2), "dec", 7 * D), Feat(("item",), (0,), "dec", 7 * D), "conv"),
]


def _brute(f: Feat, ev: pl.DataFrame, row: dict) -> float:
    m = ev.filter(pl.col("label").is_in(list(f.events)) & (pl.col("ts") < row["cutoff"]))
    for k in f.key:
        m = m.filter(pl.col(k) == row[k])
    T = row["cutoff"]
    ts = m["ts"].to_numpy().astype(float)
    if f.agg == "cnt":
        return float(((ts >= T - f.param)).sum())
    if f.agg == "dec":
        return float(np.exp(-math.log(2) * (T - ts) / f.param).sum())
    if f.agg == "tsl":
        return (T - ts.max()) / H if len(ts) else np.nan
    return (T - ts.min()) / H if len(ts) else np.nan


def test_engine_matches_brute_force():
    ev, rows = _toy()
    got = compute(FEATS, ev, rows)
    for i, row in enumerate(rows.head(120).iter_rows(named=True)):
        for f in FEATS:
            if isinstance(f, Ratio):
                a, b = _brute(f.num, ev, row), _brute(f.den, ev, row)
                exp = a / (b + 1)
            else:
                exp = _brute(f, ev, row)
            v = got[f.name][i]
            assert (np.isnan(exp) and np.isnan(v)) or abs(v - exp) <= 1e-3 * max(1.0, abs(exp)), (f.name, v, exp)


def test_future_events_do_not_change_features():
    ev, rows = _toy(1)
    cut = 30 * D
    rows = rows.filter(pl.col("cutoff") <= cut)
    a = compute(FEATS, ev, rows)
    future = ev.filter(pl.col("ts") >= cut)
    rng = np.random.default_rng(3)
    mutated = pl.concat([
        ev.filter(pl.col("ts") < cut),
        future.with_columns(pl.Series("user", rng.permutation(future["user"].to_numpy())),
                            pl.Series("label", np.full(future.height, 2, dtype=np.int8))),
    ]).sort("ts")
    b = compute(FEATS, mutated, rows)
    c = compute(FEATS, ev.filter(pl.col("ts") < cut), rows)
    for name in a.columns:
        np.testing.assert_allclose(a[name].to_numpy(), b[name].to_numpy(), rtol=1e-5, equal_nan=True)
        np.testing.assert_allclose(a[name].to_numpy(), c[name].to_numpy(), rtol=1e-5, equal_nan=True)


def test_canary_leak_is_detected():
    """Shift the cutoff by one day (the classic off-by-one-day leak): the same future-mutation
    check must now FAIL. If it did not, test 2 would be vacuous."""
    ev, rows = _toy(1)
    cut = 30 * D
    rows = rows.filter(pl.col("cutoff") <= cut)
    leaky_rows = rows.with_columns(pl.col("cutoff") + D)
    f = [Feat(("user",), (0, 1, 2), "cnt", 7 * D)]
    a = compute(f, ev, leaky_rows)
    b = compute(f, ev.filter(pl.col("ts") < cut), leaky_rows)
    assert not np.allclose(a[f[0].name].to_numpy(), b[f[0].name].to_numpy())
