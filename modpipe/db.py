"""Postgres access: checks, the review queue, and the identity/logo registries."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/modpipe")
MIGRATIONS = Path(__file__).resolve().parents[1] / "migrations"

# Higher = reviewed first. Anything involving minors jumps the whole queue.
PRIORITY = {"csam_risk": 100, "minor_sexual": 100, "real_person_sexual": 80, "protected_face": 60,
            "self_harm": 55, "gore": 40, "hate": 40, "sexual": 30, "nsfw_exposed": 30,
            "real_person": 20, "obfuscation": 15, "trademark": 10, "logo": 10}

_pool: ConnectionPool | None = None


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=int(os.environ.get("DB_POOL", "10")),
                               kwargs={"row_factory": dict_row, "autocommit": True}, open=True)
    return _pool


def close() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def migrate() -> None:
    with psycopg.connect(DATABASE_URL, autocommit=True) as c:
        c.execute("SELECT pg_advisory_lock(424242)")
        for f in sorted(MIGRATIONS.glob("*.sql")):
            c.execute(f.read_text())
        c.execute("SELECT pg_advisory_unlock(424242)")


def priority_for(reasons: list[dict]) -> int:
    return max((PRIORITY.get(r["label"], 5) for r in reasons), default=5)


def insert_check(conn, *, tenant, stage, subject, content_hash, input, status="done", decision=None,
                 policy_version=None, model_versions=None, latency_ms=None, frames=None,
                 webhook_url=None) -> str:
    row = conn.execute(
        """INSERT INTO checks (tenant, stage, subject, content_hash, input, status, action, scores, reasons,
                               policy_version, model_versions, latency_ms, frames_checked, webhook_url,
                               final_action, finished_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, CASE WHEN %s = 'done' THEN now() END)
           RETURNING id""",
        (tenant, stage, subject, content_hash, Jsonb(input), status,
         decision.action if decision else None,
         Jsonb(decision.scores) if decision else None, Jsonb(decision.reasons) if decision else None,
         policy_version, Jsonb(model_versions or {}), latency_ms, frames, webhook_url,
         (decision.action if decision and decision.action != "review" else None), status)).fetchone()
    cid = str(row["id"])
    if decision and decision.action == "review":
        enqueue_review(conn, cid, tenant, decision.reasons)
    return cid


def finish_check(conn, cid: str, decision, latency_ms: float, frames: int | None, model_versions: dict,
                 policy_version: str) -> dict:
    with conn.transaction():
        row = conn.execute(
            """UPDATE checks SET status='done', action=%s, scores=%s, reasons=%s, latency_ms=%s,
                   frames_checked=%s, model_versions=%s, policy_version=%s, finished_at=now(),
                   final_action = CASE WHEN %s <> 'review' THEN %s END
               WHERE id=%s RETURNING tenant, webhook_url""",
            (decision.action, Jsonb(decision.scores), Jsonb(decision.reasons), latency_ms, frames,
             Jsonb(model_versions), policy_version, decision.action, decision.action, cid)).fetchone()
        if decision.action == "review":
            enqueue_review(conn, cid, row["tenant"], decision.reasons)
    return row


def fail_check(conn, cid: str, err: str) -> None:
    conn.execute("UPDATE checks SET status='error', error=%s, finished_at=now() WHERE id=%s", (err[:2000], cid))


def enqueue_review(conn, check_id: str, tenant: str, reasons: list[dict]) -> None:
    conn.execute("""INSERT INTO review_items (check_id, tenant, priority) VALUES (%s,%s,%s)
                    ON CONFLICT (check_id) DO NOTHING""", (check_id, tenant, priority_for(reasons)))


# ---- review queue ---------------------------------------------------------------------------

CLAIM_SQL = """
UPDATE review_items r SET status='claimed', claimed_by=%(who)s,
       lease_until=now() + make_interval(secs => %(lease)s), attempts = r.attempts + 1
WHERE r.id = (
    SELECT id FROM review_items
    WHERE status <> 'resolved' AND (status = 'pending' OR lease_until < now())
      AND (%(tenant)s::text IS NULL OR tenant = %(tenant)s)
    ORDER BY priority DESC, created_at
    LIMIT 1
    FOR UPDATE SKIP LOCKED)
RETURNING r.*"""


def claim(conn, reviewer: str, lease_s: int = 300, tenant: str | None = None) -> dict | None:
    """Hand the highest-priority open item to one reviewer. SKIP LOCKED lets any number of
    reviewers claim concurrently without blocking each other or double-claiming; an expired
    lease (reviewer closed the tab) puts the item back in the pool."""
    item = conn.execute(CLAIM_SQL, {"who": reviewer, "lease": lease_s, "tenant": tenant}).fetchone()
    if not item:
        return None
    chk = conn.execute("SELECT * FROM checks WHERE id=%s", (item["check_id"],)).fetchone()
    return {**item, "check": chk}


def resolve(conn, item_id: int, reviewer: str, decision: str, labels: list[str] | None, note: str | None) -> bool:
    """Only the current lease holder can resolve. Returns False if the lease was lost."""
    with conn.transaction():
        row = conn.execute(
            """UPDATE review_items SET status='resolved', decision=%s, labels=%s, note=%s, resolved_at=now()
               WHERE id=%s AND claimed_by=%s AND status='claimed' AND lease_until >= now()
               RETURNING check_id""", (decision, labels, note, item_id, reviewer)).fetchone()
        if not row:
            return False
        conn.execute("UPDATE checks SET final_action=%s WHERE id=%s", (decision, row["check_id"]))
        conn.execute("INSERT INTO audit_log (actor, action, target, detail) VALUES (%s,'review.resolve',%s,%s)",
                     (reviewer, str(item_id), Jsonb({"decision": decision, "labels": labels})))
    return True


def queue_stats(conn) -> dict:
    return conn.execute(
        """SELECT count(*) FILTER (WHERE status='pending') AS pending,
                  count(*) FILTER (WHERE status='claimed' AND lease_until >= now()) AS claimed,
                  count(*) FILTER (WHERE status='claimed' AND lease_until < now()) AS expired_leases,
                  count(*) FILTER (WHERE status='resolved') AS resolved,
                  extract(epoch FROM percentile_cont(0.5) WITHIN GROUP (ORDER BY resolved_at - created_at)
                          FILTER (WHERE status='resolved')) AS p50_time_to_decision_s
           FROM review_items""").fetchone()


# ---- registries -----------------------------------------------------------------------------

def bump_registry(conn) -> int:
    v = conn.execute("UPDATE registry_version SET v = v + 1 RETURNING v").fetchone()["v"]
    conn.execute("SELECT pg_notify('registry_changed', %s)", (str(v),))
    return v


def registry_version(conn) -> int:
    return conn.execute("SELECT v FROM registry_version").fetchone()["v"]


def enroll_identity(conn, identity_id: str, name: str, vecs: list[np.ndarray], tenant: str | None, source: str,
                    model: str = "yunet+sface+dlib-v2") -> int:
    with conn.transaction():
        conn.execute("""INSERT INTO protected_identities (id, tenant, display_name) VALUES (%s,%s,%s)
                        ON CONFLICT (id) DO UPDATE SET display_name = EXCLUDED.display_name""",
                     (identity_id, tenant, name))
        # enrolling again replaces that identity's photos (re-enrollment, not accumulation)
        conn.execute("DELETE FROM identity_embeddings WHERE identity_id=%s", (identity_id,))
        for v in vecs:
            conn.execute("INSERT INTO identity_embeddings (identity_id, vec, source, model) VALUES (%s,%s,%s,%s)",
                         (identity_id, [float(x) for x in v], source, model))
        conn.execute("INSERT INTO audit_log (actor, action, target) VALUES ('api','identity.enroll',%s)", (identity_id,))
        return bump_registry(conn)


def load_identities(conn) -> list[tuple[str, np.ndarray]]:
    rows = conn.execute("SELECT identity_id, vec FROM identity_embeddings ORDER BY id").fetchall()
    return [(r["identity_id"], np.asarray(r["vec"], np.float32)) for r in rows]


def add_logo(conn, logo_id: str, name: str, image: bytes, tenant: str | None) -> int:
    with conn.transaction():
        conn.execute("""INSERT INTO logo_refs (id, tenant, display_name, image) VALUES (%s,%s,%s,%s)
                        ON CONFLICT (id) DO UPDATE SET image = EXCLUDED.image, display_name = EXCLUDED.display_name""",
                     (logo_id, tenant, name, image))
        conn.execute("INSERT INTO audit_log (actor, action, target) VALUES ('api','logo.add',%s)", (logo_id,))
        return bump_registry(conn)


def load_logos(conn) -> list[tuple[str, bytes]]:
    return [(r["id"], bytes(r["image"])) for r in conn.execute("SELECT id, image FROM logo_refs ORDER BY id")]


def dumps(o) -> str:
    return json.dumps(o, default=str)
