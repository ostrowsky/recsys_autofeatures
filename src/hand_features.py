"""E1: the hand-crafted feature set a RecSys engineer would write by the usual heuristics
(user / item / user-item / user-category aggregates, conversions, recency, windows 1/7/30 days).
Expressed in the same DSL as the grammar so that E1 vs E2 isolates the CHOICE of features,
not the computation code."""
from __future__ import annotations

from fe_engine import D, Feat, Ratio

V, C, B, ANY, CB = (0,), (1,), (2,), (0, 1, 2), (1, 2)


def hand_features_retailrocket() -> list:
    u, i, c, p = ("user",), ("item",), ("cat",), ("parent",)
    ui, uc, up = ("user", "item"), ("user", "cat"), ("user", "parent")
    f = [
        # user
        Feat(u, ANY, "cnt", 1 * D), Feat(u, ANY, "cnt", 7 * D), Feat(u, ANY, "cnt", 30 * D),
        Feat(u, CB, "cnt", 30 * D), Feat(u, ANY, "tsl"), Feat(u, ANY, "age"),
        Ratio(Feat(u, CB, "cnt", 30 * D), Feat(u, V, "cnt", 30 * D), "conv"),
        # item
        Feat(i, V, "cnt", 1 * D), Feat(i, V, "cnt", 7 * D), Feat(i, V, "cnt", 30 * D),
        Feat(i, C, "cnt", 7 * D), Feat(i, C, "cnt", 30 * D), Feat(i, B, "cnt", 30 * D),
        Feat(i, ANY, "tsl"), Feat(i, ANY, "age"),
        Ratio(Feat(i, CB, "cnt", 30 * D), Feat(i, V, "cnt", 30 * D), "conv"),
        # category
        Feat(c, V, "cnt", 7 * D), Feat(c, ANY, "cnt", 30 * D), Feat(p, ANY, "cnt", 30 * D),
        # user x item (direct)
        Feat(ui, V, "cnt", 1 * D), Feat(ui, V, "cnt", 7 * D), Feat(ui, V, "cnt", 30 * D),
        Feat(ui, C, "cnt", 30 * D), Feat(ui, B, "cnt", 30 * D), Feat(ui, ANY, "tsl"), Feat(ui, C, "tsl"),
        Feat(ui, ANY, "age"),
        # user x category (indirect)
        Feat(uc, ANY, "cnt", 7 * D), Feat(uc, ANY, "cnt", 30 * D), Feat(uc, CB, "cnt", 30 * D), Feat(uc, ANY, "tsl"),
        Ratio(Feat(uc, ANY, "cnt", 30 * D), Feat(u, ANY, "cnt", 30 * D), "share"),
        Feat(up, ANY, "cnt", 30 * D), Feat(up, ANY, "tsl"),
    ]
    return f


def hand_features_yambda() -> list:
    """Same heuristics, music vocabulary: skip (0), full listen (1), like (2), negative (3);
    item attributes artist / album instead of category / parent."""
    S, L, LK, NEG, POS = (0,), (1,), (2,), (3,), (1, 2)
    u, i, ar, al = ("user",), ("item",), ("artist",), ("album",)
    ui, uar, ual = ("user", "item"), ("user", "artist"), ("user", "album")
    return [
        Feat(u, (0, 1, 2), "cnt", 1 * D), Feat(u, (0, 1, 2), "cnt", 7 * D), Feat(u, POS, "cnt", 30 * D),
        Feat(u, LK, "cnt", 30 * D), Feat(u, (0, 1, 2), "tsl"),
        Ratio(Feat(u, S, "cnt", 30 * D), Feat(u, L, "cnt", 30 * D), "conv"),
        Feat(i, L, "cnt", 1 * D), Feat(i, L, "cnt", 7 * D), Feat(i, L, "cnt", 30 * D), Feat(i, LK, "cnt", 30 * D),
        Feat(i, S, "cnt", 7 * D), Feat(i, (0, 1, 2), "age"),
        Ratio(Feat(i, S, "cnt", 30 * D), Feat(i, L, "cnt", 30 * D), "conv"),
        Feat(ar, L, "cnt", 7 * D), Feat(ar, L, "cnt", 30 * D), Feat(al, L, "cnt", 30 * D),
        Feat(ui, L, "cnt", 1 * D), Feat(ui, L, "cnt", 7 * D), Feat(ui, L, "cnt", 30 * D), Feat(ui, S, "cnt", 30 * D),
        Feat(ui, LK, "cnt", 30 * D), Feat(ui, NEG, "cnt", 30 * D), Feat(ui, (0, 1, 2), "tsl"), Feat(ui, (0, 1, 2), "age"),
        Feat(uar, L, "cnt", 7 * D), Feat(uar, L, "cnt", 30 * D), Feat(uar, LK, "cnt", 30 * D), Feat(uar, S, "cnt", 30 * D),
        Feat(uar, (0, 1, 2), "tsl"),
        Ratio(Feat(uar, L, "cnt", 30 * D), Feat(u, L, "cnt", 30 * D), "share"),
        Feat(ual, L, "cnt", 30 * D), Feat(ual, (0, 1, 2), "tsl"),
    ]


def hand_features(name: str) -> list:
    return {"retailrocket": hand_features_retailrocket, "yambda": hand_features_yambda}[name]()
