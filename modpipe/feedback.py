"""Reviewer decisions -> training data for the prompt model.

Every resolved review item on a prompt-only check is a labeled example:
  decision 'allow'          -> no labels (a false positive if the model had flagged it)
  decision 'block' + labels -> the reviewer's text labels
Items come from two places (review_items.source): checks the model flagged, and a random audit
sample of checks it allowed. Without the audit the loop would only ever learn from false
positives, never from what slipped through.

Several reviewers can see the same prompt (or the same prompt can come back). Examples are
grouped by normalized prompt; a group's label set is the majority vote, and ties are dropped.

    python -m modpipe.feedback export [--out data/prompts/reviewed.jsonl] [--since 2026-11-01]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .text.lexicon import LABELS
from .text.normalize import normalize

ROOT = Path(__file__).resolve().parents[1]

EXPORT_SQL = """
SELECT c.input->>'prompt' AS prompt, r.decision, r.labels, r.source, r.claimed_by, r.resolved_at, c.tenant
FROM review_items r JOIN checks c ON c.id = r.check_id
WHERE r.status = 'resolved' AND c.subject = 'prompt' AND c.input->>'prompt' IS NOT NULL
  AND (%(since)s::timestamptz IS NULL OR r.resolved_at >= %(since)s)
ORDER BY r.resolved_at"""


def to_example(row: dict) -> tuple[str, frozenset[str]] | None:
    if row["decision"] == "allow":
        return row["prompt"], frozenset()
    text_labels = frozenset(l for l in (row["labels"] or []) if l in LABELS)
    if not text_labels:
        return None          # blocked for a reason the prompt model doesn't predict (e.g. image only)
    return row["prompt"], text_labels


def aggregate(rows: list[dict]) -> tuple[list[dict], dict]:
    groups: dict[str, list[tuple[str, frozenset[str], str]]] = defaultdict(list)
    skipped = 0
    for r in rows:
        ex = to_example(r)
        if ex is None:
            skipped += 1
            continue
        groups[normalize(ex[0]).text].append((ex[0], ex[1], r["source"]))
    out, ties = [], 0
    for votes in groups.values():
        c = Counter(v[1] for v in votes).most_common()
        if len(c) > 1 and c[0][1] == c[1][1]:
            ties += 1
            continue
        labels = c[0][0]
        out.append({"text": votes[-1][0], "labels": sorted(labels), "kind": "reviewed",
                    "votes": len(votes), "source": votes[-1][2]})
    stats = {"resolved_items": len(rows), "examples": len(out), "unsafe": sum(bool(r["labels"]) for r in out),
             "safe": sum(not r["labels"] for r in out), "skipped_non_text": skipped, "dropped_ties": ties,
             "from_audit": sum(r["source"] == "audit" for r in out)}
    return out, stats


def export(conn, out: Path, since: str | None = None) -> dict:
    rows = conn.execute(EXPORT_SQL, {"since": since}).fetchall()
    examples, stats = aggregate(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    return stats


def main(argv=None):
    from . import db
    ap = argparse.ArgumentParser(prog="python -m modpipe.feedback")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", type=Path, default=ROOT / "data" / "prompts" / "reviewed.jsonl")
    e.add_argument("--since")
    a = ap.parse_args(argv)
    with db.pool().connection() as conn:
        print(json.dumps(export(conn, a.out, a.since)))
    db.close()


if __name__ == "__main__":
    main()
