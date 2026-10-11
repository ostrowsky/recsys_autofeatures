"""Report figures (static PNG, light surface). Palette: validated default categorical slots 1-2
(blue = RetailRocket, orange = Yambda); one hue per panel, text in ink tokens, recessive grid.

  fig_delta.png       test NDCG@10 difference to E1 with 95% paired-bootstrap CI, per method
  fig_kcurve.png      screening val NDCG@10 vs number of selected features K (E2-full)
  fig_timescales.png  time scales (window / half-life) of the features E2-full selected

usage: python experiments/make_figures.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fe_engine import D, H, Ratio  # noqa: E402
from grammars import auto_grammar  # noqa: E402
from metrics import paired_bootstrap  # noqa: E402

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HUE = {"retailrocket": "#2a78d6", "yambda": "#eb6834"}
TITLE = {"retailrocket": "RetailRocket (e-commerce)", "yambda": "Yambda (music)"}
OUT = ROOT / "reports" / "figures"
ORDER = ["E2_coarse", "E2_full", "E2_full_div", "E2_full_seg", "E2_split", "E4_neural",
         "E5_hybrid"]
LABEL = {"E0_history_recency": "E0 history by recency", "E2_coarse": "E2 grammar, windows 1/7/30 only",
         "E2_full": "E2 grammar, full time grid", "E2_full_div": "E2 + family cap",
         "E2_full_seg": "E2 + segment-aware selection", "E2_split": "E2 split: model per segment",
         "E4_neural": "E4 neural, learned kernels", "E5_hybrid": "E5 hybrid E2 + E4"}


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_delta():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF, sharey=True)
    for ax, ds in zip(axes, ("retailrocket", "yambda")):
        d = ROOT / "experiments" / "results" / ds
        ref = pl.read_parquet(d / "pq_E1_hand_lgb.parquet")
        rows = [(m, *paired_bootstrap(ref, pl.read_parquet(d / f"pq_{m}.parquet")))
                for m in ORDER if (d / f"pq_{m}.parquet").exists()]
        style(ax)
        ax.axvline(0, color=INK2, linewidth=1)
        for y, (m, mean, lo, hi) in enumerate(rows):
            sig = lo > 0 or hi < 0
            ax.plot([lo, hi], [y, y], color=HUE[ds], linewidth=2, solid_capstyle="round")
            ax.plot(mean, y, "o", markersize=8, color=HUE[ds] if sig else SURF, markeredgecolor=HUE[ds],
                    markeredgewidth=2)
            ax.annotate(f"{mean:+.4f}", (hi, y), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=INK2)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([LABEL[r[0]] for r in rows], color=INK, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(TITLE[ds], color=INK, fontsize=11, loc="left")
        ax.set_xlabel("test NDCG@10 minus E1 (hand-crafted + tuned LightGBM)", color=INK2, fontsize=9)
        lo_all = min(r[2] for r in rows)
        ax.set_xlim(max(lo_all * 1.15, -0.06), max(r[3] for r in rows) * 1.6 + 0.004)
    fig.text(0.01, 0.01, "Bars: 95% paired bootstrap CI over test queries. Filled marker = CI excludes 0. "
             "E0 (history by recency) is off this scale: -0.015 RetailRocket, -0.103 Yambda.",
             color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "fig_delta.png", dpi=160, facecolor=SURF)


def fig_kcurve():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), facecolor=SURF)
    for ax, ds in zip(axes, ("retailrocket", "yambda")):
        e = json.loads((ROOT / "experiments" / "results" / ds / "E2_full.json").read_text())
        ks = sorted(int(k) for k in e["k_curve"])
        vs = [e["k_curve"][str(k)] for k in ks]
        style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.plot(ks, vs, color=HUE[ds], linewidth=2, marker="o", markersize=8, markeredgecolor=SURF,
                markeredgewidth=2)
        ax.margins(y=0.25)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        k = e["k_star"]
        ax.annotate(f"chosen K={k}", (k, e["k_curve"][str(k)]), xytext=(0, -16), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK)
        ax.set_title(f"{TITLE[ds]} — grammar size {e['grammar_size']}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("K selected features (by gain)", color=INK2, fontsize=9)
        ax.set_ylabel("val NDCG@10 (screening)", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_kcurve.png", dpi=160, facecolor=SURF)


def fig_timescales():
    bins = [("≤12h", 12 * H), ("1–2d", 2 * D), ("4–7d", 7 * D), ("10–21d", 21 * D), ("30d", 30 * D),
            ("45–90d", 90 * D)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0), facecolor=SURF, sharey=True)
    for ax, ds in zip(axes, ("retailrocket", "yambda")):
        by = {f.name: f for f in auto_grammar(ds, "full")}
        e = json.loads((ROOT / "experiments" / "results" / ds / "E2_full.json").read_text())
        cnt = collections.Counter()
        for n in e["chosen"]:
            f = by.get(n)
            if f is None:
                continue
            b = f.num if isinstance(f, Ratio) else f
            if b.agg in ("cnt", "dec"):
                cnt[next(lbl for lbl, hi in bins if b.param <= hi)] += 1
        labels = [b[0] for b in bins]
        vals = [cnt[lbl] for lbl in labels]
        style(ax)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.bar(labels, vals, color=HUE[ds], width=0.7, edgecolor=SURF, linewidth=2)
        for i, v in enumerate(vals):
            if v:
                ax.annotate(str(v), (i, v), xytext=(0, 3), textcoords="offset points", ha="center",
                            fontsize=8.5, color=INK2)
        ax.set_title(TITLE[ds], color=INK, fontsize=10, loc="left")
        ax.set_xlabel("window length / half-life of selected time-based features", color=INK2, fontsize=9)
    axes[0].set_ylabel("features", color=INK2, fontsize=9)
    axes[0].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(OUT / "fig_timescales.png", dpi=160, facecolor=SURF)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_delta()
    fig_kcurve()
    fig_timescales()
    print("figures written to", OUT)
