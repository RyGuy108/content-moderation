"""fastText-style multi-label prompt classifier in PyTorch.

Three EmbeddingBags (word-level tokens, word bigrams, char 3-5-grams) are mean-pooled
separately and concatenated, so a single decisive token like <person> isn't drowned out by the
40 char n-grams around it. Then a small MLP and one sigmoid per policy label. Runs on CPU in a
fraction of a millisecond.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from .features import N_BUCKETS, Featurized, Gazetteer, featurize
from .lexicon import LABELS

KINDS = ("special", "ids", "bigrams", "chars")


class PromptNet(nn.Module):
    def __init__(self, dim: int = 48, hidden: int = 96, n_labels: int = len(LABELS)):
        super().__init__()
        # one shared table (the hash already separates kinds by prefix), three pooled views
        self.emb = nn.EmbeddingBag(N_BUCKETS, dim, mode="mean", sparse=True)
        self.mlp = nn.Sequential(nn.Dropout(0.2), nn.Linear(len(KINDS) * dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_labels))
        nn.init.normal_(self.emb.weight, 0, 0.05)

    def forward(self, packed: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        return self.mlp(torch.cat([self.emb(i, o) for i, o in packed], dim=1))


def pack(feats: list[Featurized], keep: torch.Tensor | None = None, drop: float = 0.0,
         gen: torch.Generator | None = None):
    """Pack features into (ids, offsets) per kind. `keep` masks out ids never seen in training
    (so unknown n-grams don't dilute the mean; an all-unknown bag pools to zeros).

    `drop` is training-time dropout: each plain-text bag (words, bigrams, chars) is dropped
    whole with probability `drop`, and surviving bags lose `drop` of their ids. Special tokens
    are never dropped. This simulates prompts whose wording the model has never seen, and
    forces it to learn what <person>, <lex_sexual> or <flag_leet> mean on their own."""
    out = []
    bag_drop = None
    if drop:
        bag_drop = torch.rand(len(feats), len(KINDS), generator=gen) < drop
        bag_drop[:, 0] = False
    for k, kind in enumerate(KINDS):
        ids, offsets, n = [], [], 0
        for j, f in enumerate(feats):
            t = torch.tensor(getattr(f, kind), dtype=torch.long)
            if keep is not None:
                t = t[keep[t]]
            if bag_drop is not None and k > 0:
                if bag_drop[j, k]:
                    t = t[:0]
                elif len(t) > 1:
                    # per-example rate up to 0.9: real prompts often have most n-grams unseen
                    rate = float(torch.rand(1, generator=gen)) * 0.9
                    m = torch.rand(len(t), generator=gen) >= rate
                    if m.any():
                        t = t[m]
            offsets.append(n)
            ids.append(t)
            n += len(t)
        out.append((torch.cat(ids) if ids else torch.zeros(0, dtype=torch.long),
                    torch.tensor(offsets, dtype=torch.long)))
    return out


def save_compact(net: PromptNet, path: Path, touched: torch.Tensor) -> None:
    """Store only embedding rows that training touched, in fp16. A 2^20 x d table is ~200 MB;
    the rows a few thousand prompts touch are ~1 MB. Untouched ids are masked at inference."""
    state = {k: v for k, v in net.state_dict().items() if k != "emb.weight"}
    idx = touched.nonzero().squeeze(1)
    state["emb.idx"] = idx.to(torch.int32)
    state["emb.rows"] = net.emb.weight.detach()[idx].half()
    torch.save(state, path)


def load_compact(net: PromptNet, path: Path) -> torch.Tensor:
    state = torch.load(path, map_location="cpu")
    idx, rows = state.pop("emb.idx").long(), state.pop("emb.rows").float()
    w = torch.zeros_like(net.emb.weight)
    w[idx] = rows
    state["emb.weight"] = w
    net.load_state_dict(state)
    keep = torch.zeros(net.emb.num_embeddings, dtype=torch.bool)
    keep[idx] = True
    return keep


class PromptClassifier:
    """Loads a trained PromptNet and scores raw prompts."""

    def __init__(self, path: str | Path, gazetteer: Gazetteer | None = None):
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        self.labels: list[str] = meta["labels"]
        self.net = PromptNet(meta["dim"], meta["hidden"], len(self.labels))
        self.keep = load_compact(self.net, path / "model.pt")
        self.net.eval()
        self.gaz = gazetteer
        self.version = meta.get("version", "dev")
        torch.set_num_threads(1)

    @torch.inference_mode()
    def score_batch(self, prompts: list[str]) -> list[tuple[dict[str, float], Featurized]]:
        feats = [featurize(p, self.gaz) for p in prompts]
        probs = torch.sigmoid(self.net(pack(feats, self.keep))).tolist()
        return [({l: round(p, 4) for l, p in zip(self.labels, row)}, f)
                for row, f in zip(probs, feats)]

    def score(self, prompt: str) -> tuple[dict[str, float], Featurized]:
        return self.score_batch([prompt])[0]
