"""Train the prompt classifier on data/prompts/train.jsonl.

Holds out 10% of train for early stopping and threshold calibration. The test, adversarial and
natural sets are never touched here. Writes models/prompt/{model.pt,meta.json}.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modpipe.text.features import featurize  # noqa: E402
from modpipe.text.lexicon import LABELS  # noqa: E402
from modpipe.text.model import PromptNet, pack, save_compact  # noqa: E402


def load(name):
    return [json.loads(l) for l in open(ROOT / "data" / "prompts" / f"{name}.jsonl")]


def target(rows):
    y = torch.zeros(len(rows), len(LABELS))
    for i, r in enumerate(rows):
        for l in r["labels"]:
            y[i, LABELS.index(l)] = 1
    return y


def augment(text: str, rng: random.Random) -> str:
    """Light training-time noise: casing, dropped commas, extra style words. Obfuscation is NOT
    augmented here; the normalizer is what handles it, and the adversarial set tests that."""
    if rng.random() < 0.3:
        text = text.title() if rng.random() < .5 else text.upper()
    if rng.random() < 0.3:
        text = text.replace(",", "")
    if rng.random() < 0.2:
        text = "please make " + text
    return text


def main(seed=0, epochs=12, out=ROOT / "models" / "prompt", extra: list[dict] | None = None, extra_repeat: int = 4):
    """extra: reviewer-labeled rows (modpipe/feedback.py). They go into training only, repeated
    `extra_repeat` times so a few hundred real examples aren't drowned by ~20k templated ones."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    rows = load("train")
    rng.shuffle(rows)
    n_val = len(rows) // 10
    val, tr = rows[:n_val], rows[n_val:]
    tr = tr + [{"text": r["text"], "labels": r["labels"]} for r in (extra or [])] * extra_repeat
    net = PromptNet()
    sparse = torch.optim.SparseAdam(list(net.emb.parameters()), lr=0.02)
    dense = torch.optim.AdamW(net.mlp.parameters(), lr=3e-3, weight_decay=1e-4)
    pos = target(tr).sum(0)
    pos_weight = ((len(tr) - pos) / pos.clamp(min=1)).clamp(max=8.0)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    vf = [featurize(r["text"]) for r in val]
    g = torch.Generator().manual_seed(seed)
    vy = target(val)
    best, best_state = 1e9, None
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        rng.shuffle(tr)
        tot = 0.0
        for i in range(0, len(tr), 64):
            batch = tr[i:i + 64]
            packed = pack([featurize(augment(r["text"], rng)) for r in batch], drop=0.3, gen=g)
            loss = lossf(net(packed), target(batch))
            sparse.zero_grad(); dense.zero_grad()
            loss.backward()
            sparse.step(); dense.step()
            tot += loss.item() * len(batch)
        net.eval()
        with torch.no_grad():
            vl = lossf(net(pack(vf)), vy).item()
        print(f"epoch {ep} train {tot/len(tr):.4f} val {vl:.4f}")
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    out.mkdir(parents=True, exist_ok=True)
    touched = torch.zeros(net.emb.num_embeddings, dtype=torch.bool)
    for r in tr:
        for t in (r["text"], r["text"].title(), r["text"].replace(",", ""), "please make " + r["text"]):
            f = featurize(t)
            touched[torch.tensor(f.special + f.ids + f.bigrams + f.chars)] = True
    save_compact(net, out / "model.pt", touched)
    with torch.no_grad():
        vp = torch.sigmoid(net(pack(vf)))
    del vp
    json.dump({"labels": LABELS, "dim": 48, "hidden": 96, "version": f"prompt-{int(time.time())}",
               "train_rows": len(tr), "val_rows": len(val),
               "reviewed_rows": len(extra or []), "extra_repeat": extra_repeat if extra else 0, "seconds": round(time.time() - t0, 1)},
              open(out / "meta.json", "w"), indent=1)
    print("saved", out, f"{time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
