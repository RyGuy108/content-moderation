"""RQ worker for post-checks.

Uses rq's SimpleWorker (no fork per job) so the models load once per process, not once per job.
Run several of these for parallelism: `python -m modpipe.worker` (WORKERS=N to spawn N).
A crashed worker's job lands in RQ's StartedJobRegistry and is re-queued by the next worker
that cleans it up; the check row stays 'running' until then, so nothing is silently lost.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import multiprocessing as mp
import os
import time

import httpx
from rq import Queue, SimpleWorker

from . import db, runtime

log = logging.getLogger("modpipe.worker")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "dev-secret").encode()


def run_postcheck(check_id: str) -> dict:
    chk = runtime.checker()
    with db.pool().connection() as conn:
        row = conn.execute("UPDATE checks SET status='running' WHERE id=%s RETURNING *", (check_id,)).fetchone()
    if row is None:
        return {"missing": check_id}
    path = runtime.MEDIA_DIR / row["input"]["media"][0]
    tenant = row["tenant"]
    t0 = time.perf_counter()
    try:
        cache_key = runtime.cache_key(tenant, row["content_hash"])
        cached = runtime.cache_get(cache_key)
        if cached:
            from .policy import Decision
            d, info = Decision(cached["action"], cached["reasons"], cached["scores"]), {"frames": cached.get("frames")}
        else:
            with runtime.vision_lock:
                if row["subject"] == "video":
                    d, info = chk.postcheck_video(tenant, str(path))
                else:
                    d, info = chk.postcheck_image(tenant, path.read_bytes())
            runtime.cache_put(cache_key, {**d.to_dict(), "frames": info.get("frames")})
        latency = (time.perf_counter() - t0) * 1000
        with db.pool().connection() as conn:
            r = db.finish_check(conn, check_id, d, latency, info.get("frames"), chk.model_versions,
                                chk.policy(tenant).version)
    except Exception as e:
        log.exception("postcheck %s failed", check_id)
        with db.pool().connection() as conn:
            db.fail_check(conn, check_id, repr(e))
        raise
    if r and r["webhook_url"]:
        _webhook(r["webhook_url"], {"check_id": check_id, **d.to_dict()})
    return {"check_id": check_id, "action": d.action}


def _webhook(url: str, payload: dict) -> None:
    body = json.dumps(payload).encode()
    sig = hmac.new(WEBHOOK_SECRET, body, hashlib.sha256).hexdigest()
    for attempt in range(4):
        try:
            httpx.post(url, content=body, timeout=10,
                       headers={"content-type": "application/json", "x-modpipe-signature": sig}).raise_for_status()
            return
        except httpx.HTTPError as e:
            log.warning("webhook attempt %d failed: %s", attempt + 1, e)
            time.sleep(0.5 * 2 ** attempt)


def _work(n: int) -> None:
    logging.basicConfig(level=logging.INFO)
    runtime.checker()
    q = Queue("postcheck", connection=runtime.rds())
    SimpleWorker([q], connection=runtime.rds(), name=f"modpipe-{os.getpid()}").work(with_scheduler=False)


if __name__ == "__main__":
    n = int(os.environ.get("WORKERS", "1"))
    if n == 1:
        _work(0)
    else:
        procs = [mp.get_context("spawn").Process(target=_work, args=(i,), daemon=False) for i in range(n)]
        [p.start() for p in procs]
        [p.join() for p in procs]
