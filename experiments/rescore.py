"""Re-create test scores for runs that finished before scores were saved: same features,
the SAVED tuned params, same 3 seeds -> identical protocol, no re-tuning.

usage: python experiments/rescore.py retailrocket E1_hand_lgb E2_full E2_coarse
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_dataset  # noqa: E402
from grammars import auto_grammar  # noqa: E402
from hand_features import hand_features  # noqa: E402
from pipeline import compute_splits, final_eval, prepare  # noqa: E402


def main(name: str, methods: list[str]) -> None:
    ds, sp, cc = load_dataset(name)
    d = ROOT / "experiments" / "results" / name
    P = prepare(ds, sp, cc)
    for m in methods:
        if (d / f"scores_{m}.npy").exists():
            print(m, "scores exist, skipped")
            continue
        if m == "E0_history_recency":
            from fe_engine import D, Feat, compute
            labels = tuple(sorted(ds.events["label"].unique().to_list()))
            f = compute([Feat(("item",), labels, "cnt", 7 * D), Feat(("user", "item"), labels, "tsl")],
                        ds.events, P["c_test"])
            pop, tsl = f[f.columns[0]].to_numpy(), f[f.columns[1]].to_numpy()
            np.save(d / f"scores_{m}.npy", np.where(np.isnan(tsl), -1e9 + pop, -tsl))
            print(m, "scored")
            continue
        if m == "E1_hand_lgb":
            feats, params = hand_features(name), json.loads((d / "baselines.json").read_text())["results"][m]["params"]
            ref = json.loads((d / "baselines.json").read_text())["results"][m]["ndcg@10"]
        else:
            r = json.loads((d / f"{m}.json").read_text())
            grid = "coarse" if m.endswith("coarse") else "full"
            by_name = {f.name: f for f in auto_grammar(name, grid)}
            feats, params, ref = [by_name[n] for n in r["chosen"] if n in by_name], r["params"], r["ndcg@10"]
        X = compute_splits(feats, ds, P)
        fn = X["train"].columns
        Xtr, Xva, Xte = (X[s].to_numpy() for s in ("train", "val", "test"))
        met, _, sc = final_eval(Xtr, P["c_train"]["y"].to_numpy(), Xva, P["c_val"]["y"].to_numpy(), Xte,
                                P["c_test"], P["q_test"], params, fn)
        np.save(d / f"scores_{m}.npy", sc)
        print(f"{m}: re-scored ndcg@10 {met['ndcg@10']:.4f} (original run {ref:.4f})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
