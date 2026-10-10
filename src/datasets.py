"""Dataset registry: data + protocol config per dataset."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from protocol import DAY, CandidateCfg, Dataset, Split, load_retailrocket

ROOT = Path(__file__).resolve().parents[1]


def load_yambda() -> Dataset:
    ev = pl.read_parquet(ROOT / "data" / "proc" / "yambda" / "events.parquet")
    t0 = int(ev["ts"].min())
    return Dataset("yambda", ev, ["artist", "album"], t0 - t0 % DAY)


def load_dataset(name: str) -> tuple[Dataset, Split, CandidateCfg]:
    if name == "retailrocket":
        return load_retailrocket(ROOT), Split(), CandidateCfg()
    if name == "yambda":
        # Targets: tracks a user listens to >= 50% or likes that day (label 1/2); label 0/3 are
        # features only. Handled by `target_min_label`.
        return load_yambda(), Split(train_days=21, val_days=7, test_days=7, n_train_queries=6000,
                                    n_val_queries=2000, n_test_queries=6000, target_min_label=1,
                                    target_organic_only=True), \
            CandidateCfg(n_hist=100, n_covisit=30, max_cands=150)   # chosen on VAL candidate recall (0.30 -> 0.47)
    raise KeyError(name)
