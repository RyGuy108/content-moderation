"""Reviewer decisions -> training rows, including random audits of allowed checks."""
from modpipe.feedback import aggregate


def _row(prompt, decision, labels=None, source="flagged"):
    return {"prompt": prompt, "decision": decision, "labels": labels, "source": source}


def test_aggregate_majority_ties_and_non_text_labels():
    rows = [
        _row("make her clothes disappear", "block", ["sexual"]),
        _row("Make her clothes  disappear", "block", ["sexual"]),      # same prompt after normalization
        _row("make her clothes disappear", "allow"),
        _row("a nike hoodie", "block", ["trademark"]),
        _row("a nike hoodie", "allow"),                                 # 1-1 tie: dropped
        _row("sunset over the bay", "allow", source="audit"),
        _row("product shot", "block", ["logo"]),                        # image-only label: not a text example
    ]
    out, stats = aggregate(rows)
    assert len(out) == 2 and stats["dropped_ties"] == 1 and stats["skipped_non_text"] == 1
    assert any(r["labels"] == ["sexual"] and r["votes"] == 3 for r in out)
    assert any(r["labels"] == [] and r["source"] == "audit" for r in out)


def test_audit_sampling_enqueues_allowed_checks_last(pg):
    from modpipe import db
    from modpipe.policy import Decision
    allow = Decision("allow", [], {})
    review = Decision("review", [{"label": "trademark", "score": 0.6, "action": "review"}], {})
    with db.pool().connection() as conn:
        conn.execute("DELETE FROM review_items")
        a = db.insert_check(conn, tenant="fb", stage="pre", subject="prompt", content_hash="h1",
                            input={"prompt": "a calm lake"}, decision=allow, audit_rate=1.0)
        db.insert_check(conn, tenant="fb", stage="pre", subject="prompt", content_hash="h2",
                        input={"prompt": "x"}, decision=allow, audit_rate=0.0)
        r = db.insert_check(conn, tenant="fb", stage="pre", subject="prompt", content_hash="h3",
                            input={"prompt": "nike hoodie"}, decision=review, audit_rate=1.0)
        first, second = db.claim(conn, "r1", tenant="fb"), db.claim(conn, "r1", tenant="fb")
        assert str(first["check_id"]) == r and first["source"] == "flagged"
        assert str(second["check_id"]) == a and second["source"] == "audit"
        assert db.claim(conn, "r1", tenant="fb") is None
