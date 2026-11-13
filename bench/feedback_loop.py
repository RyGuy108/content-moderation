"""End-to-end test of the reviewer -> retraining loop, through the real queue.

1. A fresh database. The 203 hand-written prompts of data/prompts/natural.tsv go through the
   pre-check as if they were live traffic. 'review' decisions enter the queue as usual, and a
   random AUDIT share of 'allow' decisions enters it as audits.
2. Simulated reviewers (the file's labels play the reviewer) claim and resolve every queued
   item via db.claim / db.resolve: the same path the console uses.
3. modpipe.feedback exports the resolved items as training rows.
4. The prompt model is retrained twice from the same seed: without and with the reviewed rows,
   so the difference is the feedback alone.
5. Promotion gate (bench/retrain.py): the candidate must not lose recall on the templated test
   set or gain false positives on its hard negatives. natural_v2 (written after the model
   was frozen, never trained or tuned on) is reported, not gated on.

    python bench/feedback_loop.py [audit_rate ...]      # default: 0.25 1.0
Writes results/feedback.json.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SIM_DB = "postgresql://postgres:postgres@localhost:5432/modpipe_feedback_sim"
os.environ["DATABASE_URL"] = SIM_DB

import psycopg  # noqa: E402

from bench import retrain  # noqa: E402
from bench.train_text import main as train  # noqa: E402
from modpipe import db, feedback  # noqa: E402
from modpipe.checker import Checker, sha256  # noqa: E402


def fresh_db():
    db.close()
    with psycopg.connect(SIM_DB.rsplit("/", 1)[0] + "/postgres", autocommit=True) as c:
        c.execute("DROP DATABASE IF EXISTS modpipe_feedback_sim WITH (FORCE)")
        c.execute("CREATE DATABASE modpipe_feedback_sim")
    db.migrate()


def run_traffic(chk: Checker, rows: list[dict], audit_rate: float, seed: int) -> dict:
    random.seed(seed)
    truth = {}
    acts = {"allow": 0, "review": 0, "block": 0}
    with db.pool().connection() as conn:
        for r in rows:
            d, _ = chk.precheck("default", r["text"], [])
            acts[d.action] += 1
            cid = db.insert_check(conn, tenant="default", stage="pre", subject="prompt", content_hash=sha256(r["text"]),
                                  input={"prompt": r["text"]}, decision=d, policy_version="sim",
                                  model_versions=chk.model_versions, audit_rate=audit_rate)
            truth[cid] = r["labels"]
        # reviewers work the queue until it is empty
        reviewed = {"flagged": 0, "audit": 0}
        while (item := db.claim(conn, f"rev{len(reviewed)}", 300)) is not None:
            labels = truth[str(item["check_id"])]
            db.resolve(conn, item["id"], f"rev{len(reviewed)}", "block" if labels else "allow", labels or None, None)
            reviewed[item["source"]] += 1
    return {"decisions": acts, "reviewed": reviewed}


def main(rates=(0.25, 1.0)):
    rows = retrain.load("natural")
    base_dir = ROOT / "models" / "prompt-sim-base"
    print("training baseline (no feedback)")
    train(out=base_dir)
    chk = Checker(model_dir=ROOT / "models" / "prompt", load_vision=False)   # the deployed model takes the traffic
    results = {"traffic_rows": len(rows), "baseline_retrained": retrain.evaluate_model(base_dir), "runs": []}
    for rate in rates:
        fresh_db()
        traffic = run_traffic(chk, rows, rate, seed=1)
        out = ROOT / "data" / "prompts" / f"reviewed_sim_{rate}.jsonl"
        with db.pool().connection() as conn:
            stats = feedback.export(conn, out)
        cand = ROOT / "models" / f"prompt-sim-{rate}"
        print(f"audit {rate}: {traffic} -> {stats}; retraining")
        train(out=cand, extra=[json.loads(l) for l in open(out)])
        ev = retrain.evaluate_model(cand)
        results["runs"].append({"audit_rate": rate, **traffic, "export": stats, "candidate": ev,
                                "gate": retrain.gate(results["baseline_retrained"], ev)})
        shutil.rmtree(cand)
        out.unlink()
    shutil.rmtree(base_dir)
    db.close()
    json.dump(results, open(ROOT / "results" / "feedback.json", "w"), indent=1)
    b = results["baseline_retrained"]
    print("baseline   ", {k: b[k] for k in ("test", "hard_negatives", "natural_final")})
    for r in results["runs"]:
        print(f"audit {r['audit_rate']}", {k: r["candidate"][k] for k in ("test", "hard_negatives", "natural_final")}, r["gate"])


if __name__ == "__main__":
    main(tuple(float(a) for a in sys.argv[1:]) or (0.25, 1.0))
