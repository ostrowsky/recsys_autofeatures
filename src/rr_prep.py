"""RetailRocket -> unified events table.

Output: data/proc/retailrocket/events.parquet with columns
    user:int64, item:int64, ts:int64 (seconds), label:int8 (0 view, 1 addtocart, 2 transaction),
    cat:int32, parent:int32, root:int32   (item attributes, -1 = unknown)

Item category: RetailRocket stores `categoryid` as a time-stamped item property (weekly
snapshots). We treat it as a static attribute and take the FIRST observed value per item.
This is an item attribute, not an interaction, so it carries no information about the label.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "retailrocket"
OUT = ROOT / "data" / "proc" / "retailrocket"

LABELS = {"view": 0, "addtocart": 1, "transaction": 2}


def item_categories() -> pl.DataFrame:
    parts = []
    for name in ("item_properties_part1.csv", "item_properties_part2.csv"):
        parts.append(
            pl.scan_csv(RAW / name, schema_overrides={"value": pl.Utf8})
            .filter(pl.col("property") == "categoryid")
            .select(pl.col("itemid").alias("item"), "timestamp", pl.col("value").cast(pl.Int32).alias("cat"))
        )
    cats = (
        pl.concat(parts).collect()
        .sort(["item", "timestamp"])
        .group_by("item", maintain_order=True).first()
        .select("item", "cat")
    )
    tree = pl.read_csv(RAW / "category_tree.csv").rename({"categoryid": "cat", "parentid": "parent"})
    parent_of = dict(zip(tree["cat"].to_list(), [(-1 if p is None else int(p)) for p in tree["parent"].to_list()]))

    def root(c: int) -> int:
        seen = 0
        while parent_of.get(c, -1) != -1 and seen < 20:
            c = parent_of[c]
            seen += 1
        return c

    cats = cats.with_columns(
        pl.col("cat").map_elements(lambda c: parent_of.get(c, -1), return_dtype=pl.Int32).alias("parent"),
        pl.col("cat").map_elements(root, return_dtype=pl.Int32).alias("root"),
    )
    return cats


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ev = (
        pl.read_csv(RAW / "events.csv")
        .select(
            pl.col("visitorid").cast(pl.Int64).alias("user"),
            pl.col("itemid").cast(pl.Int64).alias("item"),
            (pl.col("timestamp") // 1000).cast(pl.Int64).alias("ts"),
            pl.col("event").replace_strict(LABELS).cast(pl.Int8).alias("label"),
        )
    )
    cats = item_categories()
    ev = (
        ev.join(cats.with_columns(pl.col("item").cast(pl.Int64)), on="item", how="left")
        .with_columns([pl.col(c).fill_null(-1).cast(pl.Int32) for c in ("cat", "parent", "root")])
        .sort("ts")
    )
    ev.write_parquet(OUT / "events.parquet")
    known = (ev["cat"] >= 0).mean()
    print(f"events {ev.height:,}  users {ev['user'].n_unique():,}  items {ev['item'].n_unique():,}  "
          f"events with known category {known:.1%}")
    print(ev.group_by("label").len().sort("label"))


if __name__ == "__main__":
    sys.exit(main())
