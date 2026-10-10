"""Yambda-50M (Yandex Music) -> unified events table, CPU-sized subsample.

Labels: 0 = listen played < 50% (skip-like), 1 = listen played >= 50%, 2 = like.
dislike / unlike / undislike are kept as label 3 (usable by features, never a target).
Item attributes: artist, album (first mapping per track).
Subsample: the last `DAYS` days of the log, users sampled at random (seeded), so the sample is
defined by the data's own time axis and not by any outcome.
Timestamps are 5-second bins relative to an unknown origin -> ts = timestamp * 5 seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "yambda"
OUT = ROOT / "data" / "proc" / "yambda"

DAYS = 70
N_USERS = 3000
SEED = 23


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ev = pl.scan_parquet(RAW / "multi_event.parquet")
    t_max = ev.select(pl.col("timestamp").max()).collect().item()
    t_lo = t_max - DAYS * 86_400 // 5
    ev = ev.filter(pl.col("timestamp") >= t_lo)
    users = ev.select("uid").unique().collect().sample(N_USERS, seed=SEED)
    ev = (
        ev.join(users.lazy(), on="uid")
        .select(
            pl.col("uid").cast(pl.Int64).alias("user"),
            pl.col("item_id").cast(pl.Int64).alias("item"),
            (pl.col("timestamp").cast(pl.Int64) * 5).alias("ts"),
            pl.when(pl.col("event_type") == "like").then(2)
            .when(pl.col("event_type") == "listen")
            .then(pl.when(pl.col("played_ratio_pct") >= 50).then(1).otherwise(0))
            .otherwise(3).cast(pl.Int8).alias("label"),
            pl.col("is_organic").cast(pl.Int8),
        )
        .collect()
    )
    art = (pl.read_parquet(RAW / "artist_item_mapping.parquet")
           .group_by("item_id").agg(pl.col("artist_id").first())
           .select(pl.col("item_id").cast(pl.Int64).alias("item"), pl.col("artist_id").cast(pl.Int64).alias("artist")))
    alb = (pl.read_parquet(RAW / "album_item_mapping.parquet")
           .group_by("item_id").agg(pl.col("album_id").first())
           .select(pl.col("item_id").cast(pl.Int64).alias("item"), pl.col("album_id").cast(pl.Int64).alias("album")))
    ev = (ev.join(art, on="item", how="left").join(alb, on="item", how="left")
          .with_columns(pl.col("artist").fill_null(-1), pl.col("album").fill_null(-1))
          .sort("ts"))
    ev.write_parquet(OUT / "events.parquet")
    print(f"events {ev.height:,}  users {ev['user'].n_unique():,}  items {ev['item'].n_unique():,}  "
          f"days {(ev['ts'].max() - ev['ts'].min()) / 86400:.1f}")
    print(ev.group_by("label").len().sort("label"))
    print("organic share", ev["is_organic"].mean(), " known artist", (ev["artist"] >= 0).mean())


if __name__ == "__main__":
    sys.exit(main())
