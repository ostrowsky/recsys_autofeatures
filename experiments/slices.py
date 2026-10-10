"""Report-only slices of the test set (never used to choose a method).

  new_items : targets the user had NO event with before the cutoff, scored among candidates the
              user has not seen either. Here "the user's own history" cannot carry the ranking,
              so the slice isolates how much the features know about relevance beyond repetition.
  repeat    : targets the user had seen before; candidates restricted to seen items.

usage: python experiments/slices.py retailrocket
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_dataset  # noqa: E402
from fe_engine import Feat, compute  # noqa: E402
from metrics import evaluate, paired_bootstrap, per_query_ndcg  # noqa: E402
from pipeline import prepare  # noqa: E402


def seen_flag(ds, frame: pl.DataFrame) -> np.ndarray:
    labels = sorted(ds.events["label"].unique().to_list())
    f = Feat(("user", "item"), tuple(labels), "age")
    return ~np.isnan(compute([f], ds.events, frame)[f.name].to_numpy())


def main(name: str) -> None:
    ds, sp, cc = load_dataset(name)
    d = ROOT / "experiments" / "results" / name
    P = prepare(ds, sp, cc)
    c, q = P["c_test"], P["q_test"]
    c_seen = seen_flag(ds, c)
    tg = q.select("qid", "user", "cutoff", "targets").explode("targets").unnest("targets")
    t_seen = seen_flag(ds, tg)
    tg = tg.with_columns(pl.Series("seen", t_seen))
    methods = {p.stem.replace("scores_", ""): np.load(p) for p in sorted(d.glob("scores_*.npy"))}
    out = ["| slice | method | queries | NDCG@10 | Recall@10 | Δ NDCG@10 vs E1 [95% CI] |", "|---|---|---|---|---|---|"]
    for sl, want_seen in (("new_items", False), ("repeat", True)):
        t = tg.filter(pl.col("seen") == want_seen)
        qs = (t.group_by("qid").agg(pl.struct("item", "rel").alias("targets"))
              .join(q.select("qid"), on="qid").sort("qid"))
        keep = (c_seen == want_seen) & c["qid"].is_in(qs["qid"].implode()).to_numpy()
        cs = c.filter(pl.Series(keep)).drop("rel", "y").join(
            t.select("qid", "item", "rel"), on=["qid", "item"], how="left").with_columns(
            pl.col("rel").fill_null(0).cast(pl.Int8), (pl.col("rel").fill_null(0) > 0).cast(pl.Int8).alias("y"))
        # keep the original row order of c so that score arrays line up
        order = c.with_row_index("_i").filter(pl.Series(keep)).select("qid", "item", "_i")
        cs = order.join(cs, on=["qid", "item"], how="left").sort("_i")
        ref = None
        rows = {}
        for m, s in methods.items():
            sc = s[cs["_i"].to_numpy()]
            r = evaluate(cs, sc, qs, ks=(10,))
            pq = per_query_ndcg(cs, sc, qs)
            rows[m] = (r, pq)
            if m == "E1_hand_lgb":
                ref = pq
        for m, (r, pq) in rows.items():
            delta = "—" if ref is None or m == "E1_hand_lgb" else "{:+.4f} [{:+.4f}, {:+.4f}]".format(*paired_bootstrap(ref, pq))
            out.append(f"| {sl} | {m} | {qs.height:,} | {r['ndcg@10']:.4f} | {r['recall@10']:.4f} | {delta} |")
        n_t = t.height
        out.append(f"| {sl} | (targets {n_t:,}; candidate rows {cs.height:,}; "
                   f"candidate recall {cs['y'].sum() / max(n_t, 1):.3f}) | | | | |")
    txt = "\n".join(out)
    (d / "slices_table.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main(sys.argv[1])
