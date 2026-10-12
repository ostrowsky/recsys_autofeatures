"""Collect every result JSON of a dataset into one Markdown table with paired-bootstrap CIs
against E1 (hand features) and against the strongest non-learned baseline.

usage: python experiments/collect_results.py retailrocket
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from metrics import paired_bootstrap  # noqa: E402


def main(name: str) -> None:
    d = ROOT / "experiments" / "results" / name
    rows: dict[str, dict] = {}
    base = json.loads((d / "baselines.json").read_text())
    rows.update(base["results"])
    for f in sorted(d.glob("E1plus_*.json")) + sorted(d.glob("E2_*.json")):
        rows[f.stem] = json.loads(f.read_text())
    if (d / "neural.json").exists():
        rows.update(json.loads((d / "neural.json").read_text()))
    pq_name = {k: d / f"pq_{k}.parquet" for k in rows}
    pq_name.update({k: d / f"pq_{k}.parquet" for k in rows if k.startswith("E2_")})
    ref = pl.read_parquet(d / "pq_E1_hand_lgb.parquet")
    rec = pl.read_parquet(d / "pq_E0_history_recency.parquet")
    lines = ["| method | features | val NDCG@10 | test NDCG@10 | Δ vs E1 [95% CI] | Δ vs history-recency [95% CI] "
             "| Recall@10 | Recall@10 cart+buy | AUC |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k, r in rows.items():
        p = pq_name[k]
        if not p.exists():
            continue
        pq = pl.read_parquet(p)
        m1, l1, h1 = paired_bootstrap(ref, pq)
        m0, l0, h0 = paired_bootstrap(rec, pq)
        v = r.get("val_ndcg@10")
        lines.append(
            f"| {k} | {r.get('n_features', '—')} | {v:.4f} | {r['ndcg@10']:.4f} | {m1:+.4f} [{l1:+.4f}, {h1:+.4f}] | "
            f"{m0:+.4f} [{l0:+.4f}, {h0:+.4f}] | {r['recall@10']:.4f} | {r['recall_cartbuy@10']:.4f} | "
            f"{r['auc']:.3f} |" if v is not None else
            f"| {k} | — | — | {r['ndcg@10']:.4f} | {m1:+.4f} [{l1:+.4f}, {h1:+.4f}] | {m0:+.4f} [{l0:+.4f}, {h0:+.4f}] "
            f"| {r['recall@10']:.4f} | {r['recall_cartbuy@10']:.4f} | {r['auc']:.3f} |")
    meta = base["meta"]
    cr = meta["candidate_recall"]["test"]
    lines += ["", f"Test queries: {meta['n_queries_all']['test']:,} available (evaluated on the seeded sample). "
              f"Candidate recall (ceiling for every method): {cr['recall_all_candidates']:.3f}; "
              f"base rate of positive candidate rows: {list(rows.values())[0]['base_rate']:.4f}."]
    out = "\n".join(lines)
    (d / "results_table.md").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
