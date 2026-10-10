"""E4: a model that learns its own aggregates instead of being given them.

For a candidate (user u, item i, cutoff T) the model reads u's last L events before T and
computes, for every relation r in {same item, same attr_1, same attr_2, ...} and every event
type e, a LEARNED temporal aggregate

      s_{r,e} = sum_j 1[r(j, i)] 1[type_j = e] K_{r,e}(T - t_j),
      K_{r,e}(dt) = sum_m softplus(a_{r,e,m}) exp(-dt / tau_m),   tau_m learnable

i.e. a differentiable version of "how many times did u do e on things related to i recently",
where both the window shape and its length are fitted by gradient descent instead of being
chosen by hand (7 days vs 10 days). Next to it, DIN-style target attention over the history
embeddings captures soft (collaborative) relations that exact-match features cannot.
Item-side popularity enters through a fixed bank of exponentially decayed counts (half-lives
6h..60d); a linear layer over that bank is again a learned kernel.

CPU-sized: L = 50, embedding dim 32.
"""
from __future__ import annotations

import math
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from fe_engine import HALF_LIVES, Feat, compute

L = 50
EMB = 32
torch.set_num_threads(4)


class HistoryBuilder:
    def __init__(self, events: pl.DataFrame, attrs: list[str]):
        e = events.filter(pl.col("label") <= 3).sort(["user", "ts"])
        self.attrs = attrs
        self.u = e["user"].to_numpy()
        self.ts = e["ts"].to_numpy()
        self.label = e["label"].to_numpy().astype(np.int64)
        items = e["item"].to_numpy()
        self.item_vocab = {x: k + 1 for k, x in enumerate(np.unique(items).tolist())}
        self.item = np.array([self.item_vocab[x] for x in items.tolist()], dtype=np.int64)
        self.attr_vocab, self.attr = {}, {}
        for a in attrs:
            v = e[a].to_numpy()
            self.attr_vocab[a] = {x: k + 1 for k, x in enumerate(np.unique(v).tolist())}
            self.attr[a] = np.array([self.attr_vocab[a][x] for x in v.tolist()], dtype=np.int64)
        uniq, start = np.unique(self.u, return_index=True)
        self.span = dict(zip(uniq.tolist(), zip(start.tolist(), np.r_[start[1:], len(self.u)].tolist())))

    def queries(self, q: pl.DataFrame) -> dict[str, np.ndarray]:
        n = q.height
        out = {"item": np.zeros((n, L), np.int64), "label": np.zeros((n, L), np.int64),
               "dt": np.zeros((n, L), np.float32), "mask": np.zeros((n, L), np.float32)}
        for a in self.attrs:
            out[a] = np.zeros((n, L), np.int64)
        for r, (user, cut) in enumerate(zip(q["user"].to_list(), q["cutoff"].to_list())):
            s, e = self.span[user]
            k = s + int(np.searchsorted(self.ts[s:e], cut, side="left"))
            lo = max(s, k - L)
            m = k - lo
            if m == 0:
                continue
            out["item"][r, :m] = self.item[lo:k][::-1]
            out["label"][r, :m] = self.label[lo:k][::-1]
            out["dt"][r, :m] = (cut - self.ts[lo:k][::-1]) / 3600.0
            out["mask"][r, :m] = 1.0
            for a in self.attrs:
                out[a][r, :m] = self.attr[a][lo:k][::-1]
        return out

    def rows(self, c: pl.DataFrame) -> dict[str, np.ndarray]:
        out = {"item": np.array([self.item_vocab.get(x, 0) for x in c["item"].to_list()], np.int64)}
        for a in self.attrs:
            out[a] = np.array([self.attr_vocab[a].get(x, 0) for x in c[a].to_list()], np.int64)
        return out


def item_bank(events: pl.DataFrame, c: pl.DataFrame, attrs: list[str], labels: list[int]) -> np.ndarray:
    feats = []
    for key in [("item",)] + [(a,) for a in attrs[:1]]:
        for ev in [tuple(labels), (labels[-1],) if len(labels) > 1 else tuple(labels)]:
            feats += [Feat(key, ev, "dec", h) for h in HALF_LIVES]
    feats = list({f.name: f for f in feats}.values())
    return np.log1p(compute(feats, events, c).to_numpy()).astype(np.float32)


class KernelAttn(nn.Module):
    def __init__(self, n_items: int, attr_sizes: list[int], n_labels: int, n_bank: int, n_src: int, M: int = 4):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, EMB, padding_idx=0, sparse=True)
        self.attr_emb = nn.ModuleList([nn.Embedding(s + 1, EMB // 2, padding_idx=0, sparse=True) for s in attr_sizes])
        self.lab_emb = nn.Embedding(n_labels, 8)
        self.n_rel = 1 + len(attr_sizes)
        self.n_labels = n_labels
        # learnable time constants (hours), init log-spaced 3h .. 30d, and per (relation, label) mix weights
        self.log_tau = nn.Parameter(torch.log(torch.tensor(np.geomspace(3, 720, M), dtype=torch.float32)))
        self.mix = nn.Parameter(torch.zeros(self.n_rel, n_labels, M))
        d_tok = EMB + (EMB // 2) * len(attr_sizes) + 8 + 4
        d_in = EMB + (EMB // 2) * len(attr_sizes)
        d_q = 32
        self.q_proj = nn.Linear(d_in, d_q)
        self.tok_proj = nn.Linear(d_tok, d_q)
        self.att = nn.Sequential(nn.Linear(4 * d_q, 32), nn.ReLU(), nn.Linear(32, 1))
        n_in = d_q * 3 + self.n_rel * n_labels + n_bank + n_src
        self.mlp = nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, 1))

    def forward(self, h: dict, r: dict, bank: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        mask, dt = h["mask"], h["dt"]
        q_parts = [self.item_emb(r["item"])] + [emb(r[k]) for emb, k in zip(self.attr_emb, r["attr_keys"])]
        q = self.q_proj(torch.cat(q_parts, -1))                                     # B x dq
        tdt = torch.log1p(dt).unsqueeze(-1)
        tfeat = torch.cat([tdt, tdt ** 2 / 10, torch.sin(dt.unsqueeze(-1) / 24 * 2 * math.pi),
                           torch.cos(dt.unsqueeze(-1) / 24 * 2 * math.pi)], -1)
        tok = torch.cat([self.item_emb(h["item"])] + [emb(h[k]) for emb, k in zip(self.attr_emb, r["attr_keys"])]
                        + [self.lab_emb(h["label"]), tfeat], -1)
        k = self.tok_proj(tok)                                                      # B x L x dq
        qe = q.unsqueeze(1).expand_as(k)
        a = self.att(torch.cat([qe, k, qe * k, qe - k], -1)).squeeze(-1)
        a = a.masked_fill(mask == 0, -1e4)
        w = torch.softmax(a, -1) * mask
        pooled = (w.unsqueeze(-1) * k).sum(1)
        # learned temporal kernels on exact relations
        tau = self.log_tau.exp()                                                    # M
        basis = torch.exp(-dt.unsqueeze(-1) / tau) * mask.unsqueeze(-1)             # B x L x M
        rels = [(h["item"] == r["item"].unsqueeze(1))] + \
               [(h[kk] == r[kk].unsqueeze(1)) & (r[kk].unsqueeze(1) > 0) for kk in r["attr_keys"]]
        lab1h = F.one_hot(h["label"], self.n_labels).float()                        # B x L x E
        s = []
        for ri, rel in enumerate(rels):
            per_lab = torch.einsum("blm,ble->bem", basis * rel.unsqueeze(-1).float(), lab1h)  # B x E x M
            s.append((per_lab * F.softplus(self.mix[ri])).sum(-1))                  # B x E
        s = torch.log1p(torch.cat(s, -1))
        x = torch.cat([q, pooled, q * pooled, s, bank, src], -1)
        return self.mlp(x).squeeze(-1)


def _batches(n: int, bs: int, rng: np.random.Generator | None):
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


class NeuralRanker:
    def __init__(self, hb: HistoryBuilder, n_labels: int, n_bank: int, n_src: int, seed: int = 0):
        torch.manual_seed(seed)
        self.hb = hb
        self.model = KernelAttn(len(hb.item_vocab), [len(hb.attr_vocab[a]) for a in hb.attrs], n_labels, n_bank, n_src)

    def _tensors(self, part: dict, idx: np.ndarray):
        qi = part["qidx"][idx]
        h = {k: torch.from_numpy(v[qi]) for k, v in part["hist"].items()}
        r = {k: torch.from_numpy(v[idx]) for k, v in part["rows"].items()}
        r["attr_keys"] = self.hb.attrs
        return h, r, torch.from_numpy(part["bank"][idx]), torch.from_numpy(part["src"][idx])

    def fit(self, tr: dict, va: dict, epochs: int = 4, bs: int = 1024, lr: float = 2e-3, eval_fn=None):
        sparse = [p for n, p in self.model.named_parameters() if "_emb" in n and "lab_emb" not in n]
        dense = [p for n, p in self.model.named_parameters() if not ("_emb" in n and "lab_emb" not in n)]
        opts = [torch.optim.SparseAdam(sparse, lr=lr), torch.optim.Adam(dense, lr=lr, weight_decay=1e-6)]
        rng = np.random.default_rng(0)
        best, best_state = -1.0, None
        for ep in range(epochs):
            self.model.train()
            t, tot = time.time(), 0.0
            for idx in _batches(len(tr["y"]), bs, rng):
                h, r, b, s = self._tensors(tr, idx)
                y = torch.from_numpy(tr["y"][idx].astype(np.float32))
                loss = F.binary_cross_entropy_with_logits(self.model(h, r, b, s), y)
                for o in opts:
                    o.zero_grad()
                loss.backward()
                for o in opts:
                    o.step()
                tot += loss.item() * len(idx)
            score = eval_fn(self.predict(va)) if eval_fn else -tot
            print(f"  [E4] epoch {ep + 1}: train logloss {tot / len(tr['y']):.4f}  val {score:.4f}  "
                  f"taus(h) {np.round(self.model.log_tau.exp().detach().numpy(), 1).tolist()}  ({time.time() - t:.0f}s)")
            if score > best:
                best, best_state = score, {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(best_state)
        return best

    @torch.no_grad()
    def predict(self, part: dict, bs: int = 4096) -> np.ndarray:
        self.model.eval()
        out = []
        for idx in _batches(len(part["y"]), bs, None):
            out.append(torch.sigmoid(self.model(*self._tensors(part, idx))).numpy())
        return np.concatenate(out)


def make_part(hb: HistoryBuilder, events: pl.DataFrame, q: pl.DataFrame, c: pl.DataFrame, attrs: list[str],
              labels: list[int], src: np.ndarray) -> dict:
    qpos = {x: k for k, x in enumerate(q["qid"].to_list())}
    return {"hist": hb.queries(q), "qidx": np.array([qpos[x] for x in c["qid"].to_list()], np.int64),
            "rows": hb.rows(c), "bank": item_bank(events, c, attrs, labels), "src": src.astype(np.float32),
            "y": c["y"].to_numpy()}
