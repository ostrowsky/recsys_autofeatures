"""Feature grammar + leak-free as-of computation.

A feature is   Agg( events filtered by type, over a time window / decay, grouped by a key )
evaluated for a candidate row (user, item, attrs..., cutoff T) using ONLY events with ts < T.

    key     : path from the candidate row to a group of events, e.g. ("user",), ("item",),
              ("user","cat") = "this user's events on items of this item's category"
    events  : tuple of labels kept (e.g. (0,) views, (1,2) cart+buy)
    agg     : cnt  -- number of events in [T - w, T)
              dec  -- sum of exp(-ln2 * (T - ts) / h)   (exponentially decayed count, half-life h)
              tsl  -- hours since the last event (NaN if none)
              age  -- hours since the first event (NaN if none)
    param   : w or h in seconds

Derived features are ratios of two base features (conversion, share).

Computation: events of a (key, event-filter) group are sorted by time; cumulative counts and
cumulative exp-weighted sums are taken per key; each candidate row is matched to the last event
strictly before its cutoff by an as-of join. One pass per group, independent of how many windows
or half-lives are requested. The same arrays could be maintained incrementally in a streaming
feature store (O(1) update per event for `cnt` buckets and `dec`).
"""
from __future__ import annotations

import math
import warnings
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import polars as pl

warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked")

H = 3600
D = 86_400


@dataclass(frozen=True)
class Feat:
    key: tuple[str, ...]
    events: tuple[int, ...]
    agg: str
    param: float = 0.0

    @property
    def name(self) -> str:
        ev = "".join(str(e) for e in self.events)
        p = "" if self.agg in ("tsl", "age") else "_" + fmt_dur(self.param)
        return f"{'x'.join(self.key)}__e{ev}__{self.agg}{p}"

    def to_sql(self, ev_label_name: str = "label") -> str:
        on = " AND ".join(f"e.{k} = c.{k}" for k in self.key)
        lab = ", ".join(str(e) for e in self.events)
        base = f"FROM events e WHERE {on} AND e.{ev_label_name} IN ({lab}) AND e.ts < c.cutoff"
        if self.agg == "cnt":
            return f"SELECT COUNT(*) {base} AND e.ts >= c.cutoff - {int(self.param)}"
        if self.agg == "dec":
            return f"SELECT SUM(EXP(-LN(2) * (c.cutoff - e.ts) / {int(self.param)})) {base}"
        if self.agg == "tsl":
            return f"SELECT (c.cutoff - MAX(e.ts)) / 3600.0 {base}"
        return f"SELECT (c.cutoff - MIN(e.ts)) / 3600.0 {base}"


@dataclass(frozen=True)
class Ratio:
    num: Feat
    den: Feat
    kind: str  # "conv" (num/(den+1)) or "share" (num/den)

    @property
    def name(self) -> str:
        return f"{self.kind}[{self.num.name}/{self.den.name}]"


AnyFeat = Feat | Ratio


def fmt_dur(s: float) -> str:
    if s % D == 0:
        return f"{int(s // D)}d"
    if s % H == 0:
        return f"{int(s // H)}h"
    return f"{s / H:.1f}h"


def _key_expr(key: tuple[str, ...]) -> pl.Expr:
    if len(key) == 1:
        return pl.col(key[0]).cast(pl.Int64).alias("_k")
    return pl.struct([pl.col(k) for k in key]).hash(seed=7).reinterpret(signed=True).alias("_k")


def base_feats(feats: list[AnyFeat]) -> list[Feat]:
    out: dict[str, Feat] = {}
    for f in feats:
        for b in ((f.num, f.den) if isinstance(f, Ratio) else (f,)):
            out[b.name] = b
    return list(out.values())


def compute(feats: list[AnyFeat], events: pl.DataFrame, rows: pl.DataFrame,
            verbose: bool = False) -> pl.DataFrame:
    """rows must contain every key column + `cutoff`. Returns a frame aligned with `rows`
    (same order) with one float32 column per requested feature."""
    bases = base_feats(feats)
    groups: dict[tuple, list[Feat]] = defaultdict(list)
    for f in bases:
        groups[(f.key, f.events)].append(f)

    rows = rows.with_row_index("_r")
    cols: dict[str, np.ndarray] = {}
    for gi, ((key, evs), fs) in enumerate(groups.items()):
        taus = sorted({f.param / math.log(2) for f in fs if f.agg == "dec"})
        windows = sorted({f.param for f in fs if f.agg == "cnt"})
        e = (events.filter(pl.col("label").is_in(list(evs)))
             .select(_key_expr(key), "ts").sort(["_k", "ts"]))
        e = e.with_columns(
            (pl.int_range(pl.len()).over("_k") + 1).cast(pl.Float64).alias("_n"),
            pl.col("ts").first().over("_k").alias("_t0"),
            pl.col("ts").alias("_tl"),
        )
        span = (e["ts"].max() - e["ts"].min()) if e.height else 0
        for t in taus:
            if span / t > 700:
                raise ValueError(f"half-life too short for exp-cumsum: span/tau={span / t:.0f}")
        e = e.with_columns([
            ((pl.col("ts") - pl.col("_t0")).cast(pl.Float64) / t).exp().cum_sum().over("_k").alias(f"_c{i}")
            for i, t in enumerate(taus)
        ]).sort("ts")
        r = rows.select("_r", _key_expr(key), "cutoff").sort("cutoff")
        m = r.join_asof(e, left_on="cutoff", right_on="ts", by="_k",
                        strategy="backward", allow_exact_matches=False)
        T = m["cutoff"].to_numpy().astype(np.float64)
        n = m["_n"].fill_null(0).to_numpy()
        last = m["_tl"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        t0 = m["_t0"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        order = m["_r"].to_numpy()
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        res: dict[str, np.ndarray] = {}
        for f in fs:
            if f.agg == "tsl":
                res[f.name] = (T - last) / H
            elif f.agg == "age":
                res[f.name] = (T - t0) / H
            elif f.agg == "dec":
                i = taus.index(f.param / math.log(2))
                c = m[f"_c{i}"].fill_null(0).to_numpy()
                with np.errstate(over="ignore", invalid="ignore"):
                    v = c * np.exp(-(T - t0) / taus[i])
                res[f.name] = np.nan_to_num(v, nan=0.0)
        n_full = np.empty(len(order))
        n_full[order] = n
        e_n = e.select("_k", "ts", "_n")
        for w in windows:
            rw = r.with_columns((pl.col("cutoff") - int(w)).alias("_tw"))   # still sorted: constant shift
            mw = rw.join_asof(e_n, left_on="_tw", right_on="ts", by="_k",
                              strategy="backward", allow_exact_matches=False)
            nw = np.empty(len(order))
            nw[mw["_r"].to_numpy()] = mw["_n"].fill_null(0).to_numpy()
            for f in fs:
                if f.agg == "cnt" and f.param == w:
                    cols[f.name] = (n_full - nw).astype(np.float32)
        for name, v in res.items():
            out = np.empty(len(order), dtype=np.float32)
            out[order] = v
            cols[name] = out
        if verbose:
            print(f"  group {gi + 1}/{len(groups)} {key} e{evs}: {len(fs)} feats, {e.height:,} events")
    for f in feats:
        if isinstance(f, Ratio):
            a, b = cols[f.num.name], cols[f.den.name]
            with np.errstate(divide="ignore", invalid="ignore"):
                cols[f.name] = (a / (b + 1.0) if f.kind == "conv" else np.where(b > 0, a / b, np.nan)).astype(np.float32)
    keep = [f.name for f in feats]
    return pl.DataFrame({k: cols[k] for k in keep})


# ----------------------------------------------------------------------------- grammar
WINDOWS = [1 * H, 3 * H, 6 * H, 12 * H, 1 * D, 2 * D, 3 * D, 5 * D, 7 * D, 10 * D, 14 * D, 21 * D, 30 * D,
           45 * D, 60 * D, 90 * D]
HALF_LIVES = [6 * H, 12 * H, 1 * D, 2 * D, 4 * D, 7 * D, 14 * D, 30 * D, 60 * D]


def grammar(user_keys: list[tuple[str, ...]], item_keys: list[tuple[str, ...]],
            event_sets: list[tuple[int, ...]], windows=WINDOWS, half_lives=HALF_LIVES,
            conv_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]] = (),
            ) -> list[AnyFeat]:
    """Enumerate the feature space. user_keys: keys that include `user` (user, user x item,
    user x cat, ...); item_keys: item-side keys (item, cat, ...)."""
    out: list[AnyFeat] = []
    keys = user_keys + item_keys
    for k in keys:
        for ev in event_sets:
            out += [Feat(k, ev, "cnt", w) for w in windows]
            out += [Feat(k, ev, "dec", h) for h in half_lives]
            out += [Feat(k, ev, "tsl"), Feat(k, ev, "age")]
        for num, den in conv_pairs:                       # conversion, e.g. cart+buy / view
            out += [Ratio(Feat(k, num, "dec", h), Feat(k, den, "dec", h), "conv") for h in half_lives]
    user_base = ("user",)
    for k in user_keys:                                   # share of the user's activity on this key
        if k == user_base:
            continue
        for ev in event_sets:
            out += [Ratio(Feat(k, ev, "dec", h), Feat(user_base, ev, "dec", h), "share") for h in half_lives]
    return out
