"""Per-dataset grammar instantiation. The ONLY dataset-specific input is the schema:
which columns are item attributes and which label codes exist. No windows, no feature list."""
from __future__ import annotations

from fe_engine import D, HALF_LIVES, WINDOWS, grammar


def schema(name: str) -> dict:
    if name == "retailrocket":
        attrs = ["cat", "parent", "root"]
        return dict(attrs=attrs, labels=[0, 1, 2], positive=[1, 2], base=[0])
    if name == "yambda":
        return dict(attrs=["artist", "album"], labels=[0, 1, 2, 3], positive=[1, 2], base=[0])
    raise KeyError(name)


def auto_grammar(name: str, grid: str = "full") -> list:
    """Generic rule: keys = user, item, each item attribute, user x item, user x attribute;
    event sets = each label alone, all non-negative labels, the 'positive' labels;
    conversions = each label over each other label."""
    s = schema(name)
    user_keys = [("user",), ("user", "item")] + [("user", a) for a in s["attrs"]]
    item_keys = [("item",)] + [(a,) for a in s["attrs"]]
    labels = s["labels"]
    ev_sets = [(l,) for l in labels] + [tuple(l for l in labels if l <= 2), tuple(s["positive"])]
    ev_sets = list(dict.fromkeys(ev_sets))
    conv = [((a,), (b,)) for a in labels for b in labels if a != b and a <= 2 and b <= 2]
    if grid == "coarse":      # ablation: the human default 1/7/30 days, no decay
        return grammar(user_keys, item_keys, ev_sets, windows=[1 * D, 7 * D, 30 * D], half_lives=[],
                       conv_pairs=[])
    return grammar(user_keys, item_keys, ev_sets, windows=WINDOWS, half_lives=HALF_LIVES, conv_pairs=conv)
