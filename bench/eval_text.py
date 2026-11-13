"""Evaluate the prompt pre-check against the keyword baseline.

Sets: test (held-out templates + held-out names), adversarial (test positives obfuscated the way
people dodge filters), natural (hand-written). Writes results/text.json.
"""
from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.baseline import baseline_labels  # noqa: E402
from modpipe.policy import Policy, decide  # noqa: E402
from modpipe.text import lexicon as lx  # noqa: E402
from modpipe.text.model import PromptClassifier  # noqa: E402

CYR = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у", "i": "і"}
LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
TERMS = sorted({t for v in lx.BLOCKLIST.values() for t in v} | set(lx.PUBLIC_FIGURES) | set(lx.BRANDS), key=len, reverse=True)


def obfuscate(text: str, rng: random.Random, how: str) -> str:
    def tx(w: str) -> str:
        if how == "leet":
            return "".join(LEET.get(c, c) if rng.random() < .6 else c for c in w)
        if how == "spaced":
            return rng.choice([" ", ".", "-", "*"]).join(w)
        if how == "homoglyph":
            return "".join(CYR.get(c, c) if rng.random() < .5 else c for c in w)
        if how == "zero_width":
            mid = len(w) // 2
            return w[:mid] + "​" + w[mid:]
        raise ValueError(how)
    low = text.lower()
    # obfuscate the terms a filter would key on, plus some other words
    for t in TERMS:
        if len(t) >= 3 and t in low:
            for part in t.split():
                low = low.replace(part, tx(part), 1)
            break
    words = low.split(" ")
    words = [tx(w) if len(w) >= 4 and rng.random() < 0.25 and w.isalpha() else w for w in words]
    return " ".join(words)


def load(name):
    return [json.loads(l) for l in open(ROOT / "data" / "prompts" / f"{name}.jsonl")]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


def evaluate(rows, clf, pol):
    scored = clf.score_batch([r["text"] for r in rows])
    out = {"n": len(rows), "ours": {}, "baseline": {}}
    flagged = {"ours": [], "baseline": []}
    per_label = {s: {l: [0, 0, 0] for l in lx.LABELS} for s in ("ours", "baseline")}
    errors = []
    for r, (scores, feat) in zip(rows, scored):
        truth = set(r["labels"])
        d = decide(pol, scores, feat.flags if feat.flags & {"homoglyph", "zero_width", "spaced", "leet"} else None)
        ours = {l for l in lx.LABELS if scores[l] >= pol.rules[l].review}
        base = baseline_labels(r["text"])
        flagged["ours"].append((bool(truth), d.action != "allow"))
        flagged["baseline"].append((bool(truth), bool(base)))
        for sysname, pred in (("ours", ours), ("baseline", base)):
            for l in lx.LABELS:
                c = per_label[sysname][l]
                if l in truth and l in pred:
                    c[0] += 1
                elif l in pred:
                    c[1] += 1
                elif l in truth:
                    c[2] += 1
        if bool(truth) != (d.action != "allow"):
            errors.append({"text": r["text"], "truth": sorted(truth), "action": d.action,
                           "top": sorted(scores.items(), key=lambda kv: -kv[1])[:2]})
    for s in ("ours", "baseline"):
        pairs = flagged[s]
        tp = sum(t and p for t, p in pairs)
        fp = sum((not t) and p for t, p in pairs)
        fn = sum(t and not p for t, p in pairs)
        neg = sum(not t for t, _ in pairs)
        out[s]["unsafe_vs_safe"] = {**prf(tp, fp, fn), "false_positive_rate": round(fp / neg, 4) if neg else 0}
        out[s]["per_label"] = {l: prf(*per_label[s][l]) for l in lx.LABELS}
    out["ours_errors"] = errors[:40]
    return out


def latency(clf, pol, rows, n=3000):
    texts = [r["text"] for r in rows]
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        sc, f = clf.score(texts[i % len(texts)])
        decide(pol, sc, f.flags)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {"p50_ms": round(ts[len(ts) // 2], 3), "p99_ms": round(ts[int(len(ts) * .99)], 3),
            "mean_ms": round(statistics.mean(ts), 3)}


def main():
    clf = PromptClassifier(ROOT / "models" / "prompt")
    pol = Policy.load(ROOT / "policy.yaml")
    rng = random.Random(11)
    test, natural, final = load("test"), load("natural"), load("natural_v2")
    adv = []
    for how in ("leet", "spaced", "homoglyph", "zero_width"):
        for r in test:
            if r["labels"]:
                adv.append({**r, "text": obfuscate(r["text"], rng, how), "kind": how})
    # Benign prompts obfuscated too: keeps the adversarial set from rewarding "flag anything weird"
    for r in rng.sample([r for r in test if not r["labels"]], 200):
        adv.append({**r, "text": obfuscate(r["text"], rng, rng.choice(["leet", "spaced"])), "kind": "benign_obf"})
    hard = [r for r in test if r["kind"] == "benign_hard"] + [r for r in natural + final if not r["labels"]]
    res = {"model": clf.version, "policy_version": pol.version}
    for name, rows in (("test", test), ("adversarial", adv), ("natural_dev", natural), ("natural_final", final), ("hard_negatives", hard)):
        res[name] = evaluate(rows, clf, pol)
        o, b = res[name]["ours"]["unsafe_vs_safe"], res[name]["baseline"]["unsafe_vs_safe"]
        print(f"{name:15s} n={len(rows):5d}  ours P {o['precision']:.3f} R {o['recall']:.3f} FPR {o['false_positive_rate']:.3f}"
              f"   baseline P {b['precision']:.3f} R {b['recall']:.3f} FPR {b['false_positive_rate']:.3f}")
    res["latency"] = latency(clf, pol, test)
    print("latency", res["latency"])
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(res, open(ROOT / "results" / "text.json", "w"), indent=1)


if __name__ == "__main__":
    main()
