"""Rolling-origin summary on Yambda (see reports/prereg_rolling.md).

usage: python experiments/rolling_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_dataset  # noqa: E402
from metrics import paired_bootstrap  # noqa: E402
from pipeline import prepare  # noqa: E402

WINDOWS = ["yambda_w3", "yambda_w2", "yambda_w1", "yambda"]     # oldest -> newest; "yambda" = W0 (seen)
PAIRS = [("E1_hand_lgb", "E2_full"), ("E1plus_hand_lgb", "E2_full"), ("E1_hand_lgb", "E1plus_hand_lgb")]


def diffs(name: str, a: str, b: str) -> pl.DataFrame:
    d = ROOT / "experiments" / "results" / name
    ds, sp, cc = load_dataset(name)
    q = prepare(ds, sp, cc)["q_test"].select("qid", "user")
    j = pl.read_parquet(d / f"pq_{a}.parquet").join(pl.read_parquet(d / f"pq_{b}.parquet"), on="qid", suffix="_b")
    return j.join(q, on="qid").select("user", (pl.col("ndcg_b") - pl.col("ndcg")).alias("d")) \
        .with_columns(pl.lit(name).alias("window"))


def cluster_ci(df: pl.DataFrame, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap over users (a user can appear in several days/weeks)."""
    g = df.group_by(["window", "user"]).agg(pl.col("d").sum().alias("s"), pl.len().alias("c"))
    s, c = g["s"].to_numpy(), g["c"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n, len(s)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def main() -> None:
    out = {"per_window": {}, "pooled_new": {}}
    lines = ["| window (test days) | queries | E1 | E1+ | E2-full | E2-full − E1 [95% CI] | E2-full − E1+ [95% CI] | E1+ − E1 [95% CI] |",
             "|---|---|---|---|---|---|---|---|"]
    days = {"yambda_w3": "43–49", "yambda_w2": "50–56", "yambda_w1": "57–63", "yambda": "64–70 (W0, seen)"}
    for w in WINDOWS:
        d = ROOT / "experiments" / "results" / w
        nd = {"E1_hand_lgb": json.loads((d / "baselines.json").read_text())["results"]["E1_hand_lgb"]["ndcg@10"],
              "E1plus_hand_lgb": json.loads((d / "E1plus_hand_lgb.json").read_text())["ndcg@10"],
              "E2_full": json.loads((d / "E2_full.json").read_text())["ndcg@10"]}
        cis = {f"{b}-{a}": paired_bootstrap(pl.read_parquet(d / f"pq_{a}.parquet"),
                                            pl.read_parquet(d / f"pq_{b}.parquet")) for a, b in PAIRS}
        nq = pl.read_parquet(d / "pq_E2_full.parquet").height
        out["per_window"][w] = {"ndcg": nd, "ci": cis, "queries": nq}
        f = lambda t: f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"  # noqa: E731
        lines.append(f"| {days[w]} | {nq:,} | {nd['E1_hand_lgb']:.4f} | {nd['E1plus_hand_lgb']:.4f} | "
                     f"{nd['E2_full']:.4f} | {f(cis['E2_full-E1_hand_lgb'])} | {f(cis['E2_full-E1plus_hand_lgb'])} | "
                     f"{f(cis['E1plus_hand_lgb-E1_hand_lgb'])} |")
    new = [w for w in WINDOWS if w != "yambda"]
    lines += ["", "| pooled over W1–W3 (new weeks), user-clustered bootstrap | estimate [95% CI] | windows with positive estimate |",
              "|---|---|---|"]
    for a, b in PAIRS:
        pooled = pl.concat([diffs(w, a, b) for w in new])
        m, lo, hi = cluster_ci(pooled)
        pos = sum(out["per_window"][w]["ci"][f"{b}-{a}"][0] > 0 for w in new)
        out["pooled_new"][f"{b}-{a}"] = {"mean": m, "lo": lo, "hi": hi, "positive_windows": pos}
        lines.append(f"| {b} − {a} | {m:+.4f} [{lo:+.4f}, {hi:+.4f}] | {pos} of 3 |")
    p = out["pooled_new"]["E2_full-E1_hand_lgb"]
    verdict = "SUPPORTED" if (p["lo"] > 0 and p["positive_windows"] >= 2) else "NOT SUPPORTED"
    out["primary_verdict"] = verdict
    lines += ["", f"Primary criterion (prereg_rolling.md): **{verdict}**"]
    txt = "\n".join(lines)
    (ROOT / "experiments" / "results" / "yambda_rolling.md").write_text(txt, encoding="utf-8")
    (ROOT / "experiments" / "results" / "yambda_rolling.json").write_text(json.dumps(out, indent=2))
    print(txt)


if __name__ == "__main__":
    main()
