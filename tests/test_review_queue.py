"""The review queue must never hand one item to two reviewers, must hand out highest priority
first, and must recover items whose reviewer vanished (lease expiry)."""
import threading
import time
from collections import Counter

from modpipe.policy import Decision


def _seed(db, n, reasons=None):
    with db.pool().connection() as c:
        ids = []
        for i in range(n):
            d = Decision("review", reasons or [{"label": "trademark", "score": 0.6, "action": "review"}], {})
            ids.append(db.insert_check(c, tenant="t", stage="pre", subject="prompt", content_hash=f"h{i}",
                                       input={"prompt": f"p{i}"}, decision=d, policy_version="1"))
        return ids


def test_no_double_claims_under_concurrency(pg):
    db = pg
    with db.pool().connection() as c:
        c.execute("TRUNCATE review_items, checks CASCADE")
    _seed(db, 400)
    claimed: list[int] = []
    lock = threading.Lock()

    def reviewer(name):
        while True:
            with db.pool().connection() as c:
                it = db.claim(c, name, lease_s=60)
                if not it:
                    return
                with lock:
                    claimed.append(it["id"])
                assert db.resolve(c, it["id"], name, "allow", [], None)

    ts = [threading.Thread(target=reviewer, args=(f"r{i}",)) for i in range(24)]
    t0 = time.perf_counter()
    [t.start() for t in ts]
    [t.join() for t in ts]
    dt = time.perf_counter() - t0
    dup = [k for k, v in Counter(claimed).items() if v > 1]
    assert not dup and len(claimed) == 400
    print(f"400 claim+resolve by 24 reviewers in {dt:.2f}s ({400/dt:.0f}/s)")
    with db.pool().connection() as c:
        assert db.queue_stats(c)["resolved"] == 400


def test_priority_and_lease_expiry(pg):
    db = pg
    with db.pool().connection() as c:
        c.execute("TRUNCATE review_items, checks CASCADE")
    _seed(db, 3)
    _seed(db, 1, [{"label": "csam_risk", "score": 0.2, "action": "review"}])
    with db.pool().connection() as c:
        first = db.claim(c, "alice", lease_s=1)
        assert first["priority"] == 100            # minors first, regardless of age
        time.sleep(1.2)                             # alice walks away
        again = db.claim(c, "bob", lease_s=60)
        assert again["id"] == first["id"]          # expired lease returns to the pool
        assert not db.resolve(c, first["id"], "alice", "block", [], None)   # alice lost it
        assert db.resolve(c, first["id"], "bob", "block", ["minor_sexual"], None)
        row = c.execute("SELECT final_action FROM checks WHERE id=%s", (first["check_id"],)).fetchone()
        assert row["final_action"] == "block"
