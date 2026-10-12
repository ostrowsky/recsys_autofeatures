"""Evaluation protocol shared by every method (fixed before any model is fitted).

Query      = (user, day d): the user has >=1 event on day d AND >=1 event before d (warm user).
Cutoff     = start of day d. Every feature and every candidate uses only events with ts < cutoff.
Targets    = items the user interacted with on day d; relevance = max(label on that day) + 1
             (view 1, cart 2, buy 3 for RetailRocket).
Split      = by DAY, never at random: train | val (feature/model selection) | test (looked at once).
Candidates = union of (a) user's own history, (b) item-to-item co-visitation of recent items,
             (c) recent popularity inside the user's top categories, (d) global recent popularity.
             The same candidate rows are scored by every method, so differences in ranking
             metrics come from the features/model and not from the candidate generator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

DAY = 86_400


@dataclass(frozen=True)
class Split:
    train_days: int = 28
    val_days: int = 14
    test_days: int = 14
    n_train_queries: int = 12_000
    n_val_queries: int = 4_000
    n_test_queries: int = 15_000
    seed: int = 17
    target_min_label: int = 0   # events with label in [min, 2] define targets
    target_organic_only: bool = False  # Yambda: targets only from user-initiated (organic) events
    test_end_offset_days: int = 0  # rolling-origin evaluation: shift the whole train|val|test block back


@dataclass
class CandidateCfg:
    n_hist: int = 30
    n_covisit: int = 30
    n_cat: int = 15
    n_pop: int = 10
    max_cands: int = 80
    recent_items_for_covisit: int = 5
    covisit_max_gap_s: int = 3_600
    pop_days: int = 7
    covisit_snapshot_days: int = 7


@dataclass
class Dataset:
    name: str
    events: pl.DataFrame        # user,item,ts,label,(attrs...) sorted by ts
    item_attrs: list[str]       # categorical item attributes, e.g. ["cat", "parent"]
    origin: int                 # ts of day-0 start
    n_days: int = field(init=False)

    def __post_init__(self) -> None:
        self.events = self.events.with_columns(((pl.col("ts") - self.origin) // DAY).cast(pl.Int32).alias("day"))
        self.n_days = int(self.events["day"].max()) + 1

    def cutoff(self, day: int) -> int:
        return self.origin + day * DAY


def day_ranges(ds: Dataset, sp: Split) -> dict[str, tuple[int, int]]:
    end = ds.n_days - sp.test_end_offset_days  # exclusive
    te = (end - sp.test_days, end)
    va = (te[0] - sp.val_days, te[0])
    tr = (va[0] - sp.train_days, va[0])
    return {"train": tr, "val": va, "test": te}


def build_queries(ds: Dataset, sp: Split) -> dict[str, pl.DataFrame]:
    """Returns per split: qid,user,day,cutoff + targets as list[struct(item,rel)]."""
    ev = ds.events
    first_day = ev.group_by("user").agg(pl.col("day").min().alias("first_day"))
    tev = ev.filter(pl.col("is_organic") == 1) if sp.target_organic_only else ev
    daily = (
        tev.filter(pl.col("label").is_between(sp.target_min_label, 2))
        .group_by(["user", "day", "item"]).agg(pl.col("label").max().alias("lab"))
        .with_columns((pl.col("lab") + 1 - sp.target_min_label).cast(pl.Int8).alias("rel"))
    )
    q = (
        daily.group_by(["user", "day"]).agg(pl.struct("item", "rel").alias("targets"))
        .join(first_day, on="user")
        .filter(pl.col("first_day") < pl.col("day"))   # warm: history before the cutoff
        .drop("first_day")
    )
    out: dict[str, pl.DataFrame] = {}
    caps = {"train": sp.n_train_queries, "val": sp.n_val_queries, "test": sp.n_test_queries}
    for i, (name, (a, b)) in enumerate(day_ranges(ds, sp).items()):
        part = q.filter((pl.col("day") >= a) & (pl.col("day") < b)).sort(["day", "user"])
        n_all = part.height
        if n_all > caps[name]:
            part = part.sample(caps[name], seed=sp.seed + i).sort(["day", "user"])
        part = part.with_columns(
            (pl.lit(ds.origin) + pl.col("day").cast(pl.Int64) * DAY).alias("cutoff"),
        )
        out[name] = part.with_row_index("qid", offset=i * 10_000_000).with_columns(pl.col("qid").cast(pl.Int64))
        out[name + "_n_all"] = n_all  # type: ignore[assignment]
    return out


# ----------------------------------------------------------------------------- candidates
class CandidateGenerator:
    def __init__(self, ds: Dataset, cfg: CandidateCfg):
        self.ds, self.cfg = ds, cfg
        e = ds.events.sort(["user", "ts"])
        self.u = e["user"].to_numpy()
        self.it = e["item"].to_numpy()
        self.ts = e["ts"].to_numpy()
        self.attr = {a: e[a].to_numpy() for a in ds.item_attrs[:1]}  # top category level only
        self.item_attr = dict(
            zip(ds.events["item"].to_list(), ds.events[ds.item_attrs[0]].to_list())
        ) if ds.item_attrs else {}
        uniq, start = np.unique(self.u, return_index=True)
        self.u_start = dict(zip(uniq.tolist(), start.tolist()))
        self.u_end = dict(zip(uniq.tolist(), np.r_[start[1:], len(self.u)].tolist()))
        self._pairs = self._covisit_pairs(e)
        self._snap_cache: dict[int, dict[int, list[int]]] = {}
        self._pop_cache: dict[int, tuple[list[int], dict[int, list[int]]]] = {}

    # co-visitation: consecutive distinct items of a user within covisit_max_gap_s, both directions
    def _covisit_pairs(self, e: pl.DataFrame) -> pl.DataFrame:
        g = e.select("user", "item", "ts").with_columns(
            pl.col("item").shift(-1).over("user").alias("nxt"),
            pl.col("ts").shift(-1).over("user").alias("nts"),
        ).filter(
            pl.col("nxt").is_not_null() & (pl.col("nxt") != pl.col("item"))
            & ((pl.col("nts") - pl.col("ts")) <= self.cfg.covisit_max_gap_s)
        )
        a = g.select(pl.col("item").alias("a"), pl.col("nxt").alias("b"), pl.col("nts").alias("ts"))
        b = g.select(pl.col("nxt").alias("a"), pl.col("item").alias("b"), pl.col("nts").alias("ts"))
        return pl.concat([a, b])

    def _snapshot(self, day: int) -> dict[int, list[int]]:
        snap_day = day - (day % self.cfg.covisit_snapshot_days)
        if snap_day not in self._snap_cache:
            cut = self.ds.cutoff(snap_day)
            top = (
                self._pairs.filter(pl.col("ts") < cut)
                .group_by(["a", "b"]).len()
                .sort(["a", "len"], descending=[False, True])
                .group_by("a", maintain_order=True).head(20)
                .group_by("a", maintain_order=True).agg("b")
            )
            self._snap_cache[snap_day] = dict(zip(top["a"].to_list(), top["b"].to_list()))
        return self._snap_cache[snap_day]

    def _popular(self, day: int) -> tuple[list[int], dict[int, list[int]]]:
        if day not in self._pop_cache:
            lo, hi = self.ds.cutoff(day - self.cfg.pop_days), self.ds.cutoff(day)
            w = self.ds.events.filter((pl.col("ts") >= lo) & (pl.col("ts") < hi))
            glob = w.group_by("item").len().sort("len", descending=True).head(self.cfg.n_pop)["item"].to_list()
            per_cat: dict[int, list[int]] = {}
            if self.ds.item_attrs:
                a = self.ds.item_attrs[0]
                pc = (w.group_by([a, "item"]).len().sort([a, "len"], descending=[False, True])
                      .group_by(a, maintain_order=True).head(self.cfg.n_cat)
                      .group_by(a, maintain_order=True).agg("item"))
                per_cat = dict(zip(pc[a].to_list(), pc["item"].to_list()))
            self._pop_cache[day] = (glob, per_cat)
        return self._pop_cache[day]

    def for_query(self, user: int, day: int) -> list[tuple[int, int]]:
        """Returns [(item, source_bitmask)] ordered by source priority, length <= max_cands."""
        cfg, cut = self.cfg, self.ds.cutoff(day)
        s, e = self.u_start[user], self.u_end[user]
        k = s + int(np.searchsorted(self.ts[s:e], cut, side="left"))   # events strictly before cutoff
        hist_items = self.it[s:k][::-1]
        cands: dict[int, int] = {}

        def add(item: int, bit: int) -> None:
            cands[item] = cands.get(item, 0) | bit

        seen: list[int] = []
        for x in hist_items.tolist():
            if x not in cands:
                seen.append(x)
                add(x, 1)
                if len(seen) >= cfg.n_hist:
                    break
        snap = self._snapshot(day)
        score: dict[int, float] = {}
        for rank, x in enumerate(seen[: cfg.recent_items_for_covisit]):
            for j, y in enumerate(snap.get(x, [])):
                score[y] = score.get(y, 0.0) + 1.0 / (1 + rank) / (1 + j)
        for y, _ in sorted(score.items(), key=lambda t: -t[1])[: cfg.n_covisit]:
            add(y, 2)
        glob, per_cat = self._popular(day)
        if per_cat:
            cat_cnt: dict[int, int] = {}
            for x in seen:
                c = self.item_attr.get(x, -1)
                if c >= 0:
                    cat_cnt[c] = cat_cnt.get(c, 0) + 1
            for c, _ in sorted(cat_cnt.items(), key=lambda t: -t[1])[:2]:
                for y in per_cat.get(c, []):
                    add(y, 4)
        for y in glob:
            add(y, 8)
        return list(cands.items())[: cfg.max_cands]

    def build(self, queries: pl.DataFrame) -> pl.DataFrame:
        rows_q, rows_i, rows_s = [], [], []
        for qid, user, day in zip(queries["qid"].to_list(), queries["user"].to_list(), queries["day"].to_list()):
            for item, src in self.for_query(user, day):
                rows_q.append(qid)
                rows_i.append(item)
                rows_s.append(src)
        c = pl.DataFrame({"qid": rows_q, "item": rows_i, "src": rows_s},
                         schema={"qid": pl.Int64, "item": pl.Int64, "src": pl.Int8})
        tgt = queries.select("qid", "targets").explode("targets").unnest("targets")
        c = (c.join(queries.select("qid", "user", "day", "cutoff"), on="qid")
             .join(tgt, on=["qid", "item"], how="left")
             .with_columns(pl.col("rel").fill_null(0).cast(pl.Int8),
                           (pl.col("rel").fill_null(0) > 0).cast(pl.Int8).alias("y")))
        if self.ds.item_attrs:
            attrs = self.ds.events.group_by("item").agg([pl.col(a).first() for a in self.ds.item_attrs])
            c = c.join(attrs, on="item", how="left").with_columns(
                [pl.col(a).fill_null(-1) for a in self.ds.item_attrs])
        return c.sort(["qid", "item"])


def candidate_recall(cands: pl.DataFrame, queries: pl.DataFrame) -> dict[str, float]:
    n_tgt = queries.select(pl.col("targets").list.len().sum()).item()
    hit = cands.filter(pl.col("y") == 1)
    by_src = {name: hit.filter((pl.col("src") & bit) > 0).height / n_tgt
              for name, bit in (("history", 1), ("covisit", 2), ("category_pop", 4), ("global_pop", 8))}
    return {"recall_all_candidates": hit.height / n_tgt, **{f"recall_{k}": v for k, v in by_src.items()},
            "cands_per_query": cands.height / queries.height, "targets_per_query": n_tgt / queries.height}


def load_retailrocket(root: Path) -> Dataset:
    ev = pl.read_parquet(root / "data" / "proc" / "retailrocket" / "events.parquet")
    # day boundary of the export is 03:00 UTC (first event 03:00:04, last 02:59:47)
    t0 = int(ev["ts"].min())
    origin = t0 - ((t0 - 3 * 3600) % DAY)
    return Dataset("retailrocket", ev, ["cat", "parent", "root"], origin)
