"""Retrain the prompt model with reviewer-labeled data, and promote it only if it passes the gate.

    python -m modpipe.feedback export                 # resolved review items -> data/prompts/reviewed.jsonl
    python bench/retrain.py [--promote]               # train models/prompt-candidate, compare, maybe promote

Gate (candidate vs. the deployed model):
  * recall on the templated test set may drop by at most GATE_RECALL_DROP
  * false-positive rate on hard negatives may rise by at most GATE_FPR_RISE
natural_v2 is reported but never gated on: it is the final held-out set.
Promotion copies the candidate over models/prompt. Its meta.json version changes, which is part of
the decision-cache key, so cached decisions from the old model stop being served.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.eval_text import evaluate  # noqa: E402
from bench.train_text import main as train  # noqa: E402
from modpipe.policy import Policy  # noqa: E402
from modpipe.text.model import PromptClassifier  # noqa: E402

GATE_RECALL_DROP = 0.01
GATE_FPR_RISE = 0.01


def load(name):
    return [json.loads(l) for l in open(ROOT / "data" / "prompts" / f"{name}.jsonl")]


def evaluate_model(model_dir: Path) -> dict:
    clf = PromptClassifier(model_dir)
    pol = Policy.load(ROOT / "policy.yaml")
    test = load("test")
    sets = {"test": test, "hard_negatives": [r for r in test if r["kind"] == "benign_hard"],
            "natural_final": load("natural_v2")}
    out = {"model": clf.version}
    for name, rows in sets.items():
        o = evaluate(rows, clf, pol)["ours"]["unsafe_vs_safe"]
        out[name] = {"precision": o["precision"], "recall": o["recall"], "fpr": o["false_positive_rate"]}
    return out


def gate(current: dict, cand: dict) -> dict:
    fails = []
    if cand["test"]["recall"] < current["test"]["recall"] - GATE_RECALL_DROP:
        fails.append(f"test recall {current['test']['recall']} -> {cand['test']['recall']}")
    if cand["hard_negatives"]["fpr"] > current["hard_negatives"]["fpr"] + GATE_FPR_RISE:
        fails.append(f"hard-negative FPR {current['hard_negatives']['fpr']} -> {cand['hard_negatives']['fpr']}")
    return {"pass": not fails, "failures": fails}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviewed", type=Path, default=ROOT / "data" / "prompts" / "reviewed.jsonl")
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--promote", action="store_true")
    a = ap.parse_args()
    extra = [json.loads(l) for l in open(a.reviewed)]
    cand_dir, live_dir = ROOT / "models" / "prompt-candidate", ROOT / "models" / "prompt"
    train(out=cand_dir, extra=extra, extra_repeat=a.repeat)
    cur, cand = evaluate_model(live_dir), evaluate_model(cand_dir)
    g = gate(cur, cand)
    print(json.dumps({"reviewed_rows": len(extra), "current": cur, "candidate": cand, "gate": g}, indent=1))
    if a.promote:
        if not g["pass"]:
            sys.exit("gate failed, not promoting")
        backup = ROOT / "models" / f"prompt-prev-{cur['model']}"
        shutil.copytree(live_dir, backup, dirs_exist_ok=True)
        for f in ("model.pt", "meta.json"):
            shutil.copy2(cand_dir / f, live_dir / f)
        print(f"promoted {cand['model']} (previous model kept in {backup.name}); restart workers to load it")


if __name__ == "__main__":
    main()
