"""HTTP API.

  POST /v1/precheck          sync: prompt and/or input images, before a generation is paid for
  POST /v1/postcheck         async: a Magic Hour output (image or video) -> RQ worker
  GET  /v1/checks/{id}       result of either
  POST /v1/identities        enroll a protected face (likeness protection)
  POST /v1/logos             register a mark
  POST /v1/review/claim      reviewer takes the next item (lease)
  POST /v1/review/{id}       reviewer resolves it
  GET  /v1/review/stats
  GET  /v1/media/{name}      stored media for the review console (signed URL)

Auth: every endpoint except /healthz and /metrics takes a tenant API key (modpipe/auth.py).
The key's tenant overrides any tenant in the request body.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
from typing import Literal

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field
from rq import Queue

from . import auth, db, runtime
from .auth import Principal, principal, tenant_of
from .checker import decode_image, sha256
from .policy import Decision

app = FastAPI(title="modpipe", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
                   allow_methods=["*"], allow_headers=["*"])
MEDIA_KEY = os.environ.get("MEDIA_SIGNING_KEY", "dev-media-key").encode()
MEDIA_URL_TTL = int(os.environ.get("MEDIA_URL_TTL", "900"))
MAX_MEDIA = int(os.environ.get("MAX_MEDIA_BYTES", str(200 * 1024 * 1024)))
METRICS = {"precheck_total": 0, "precheck_cache_hits": 0, "postcheck_enqueued": 0,
           "action_allow": 0, "action_review": 0, "action_block": 0}


@app.on_event("startup")
def _startup():
    db.migrate()
    runtime.checker()          # load models before taking traffic


@app.on_event("shutdown")
def _shutdown():
    db.close()


class PrecheckIn(BaseModel):
    tenant: str = "default"
    prompt: str | None = Field(None, max_length=8000)
    images_b64: list[str] = Field(default_factory=list, max_length=8)
    image_urls: list[str] = Field(default_factory=list, max_length=8)


class PostcheckIn(BaseModel):
    tenant: str = "default"
    kind: Literal["image", "video"]
    media_url: str | None = None
    media_b64: str | None = None
    magic_hour_project_id: str | None = None
    webhook_url: str | None = None


class ResolveIn(BaseModel):
    reviewer: str | None = None            # ignored when authenticated: the key's name is used
    decision: Literal["allow", "block"]
    labels: list[str] | None = None
    note: str | None = None


def _fetch(url: str) -> bytes:
    with httpx.stream("GET", url, timeout=30, follow_redirects=True) as r:
        r.raise_for_status()
        buf = bytearray()
        for chunk in r.iter_bytes():
            buf += chunk
            if len(buf) > MAX_MEDIA:
                raise HTTPException(413, "media too large")
        return bytes(buf)


def sign_media(name: str, ttl: int = MEDIA_URL_TTL) -> str:
    """Short-lived URL for <img>/<video> tags in the review console, which can't send headers."""
    exp = int(time.time()) + ttl
    sig = hmac.new(MEDIA_KEY, f"{name}:{exp}".encode(), hashlib.sha256).hexdigest()[:32]
    return f"/v1/media/{name}?exp={exp}&sig={sig}"


def _own(p: Principal | None, tenant: str) -> None:
    """Non-admin keys only see their own tenant's checks ('*' = every tenant)."""
    if p is not None and p.role != "admin" and p.tenant not in ("*", tenant):
        raise HTTPException(404)


def _count(action: str) -> None:
    METRICS[f"action_{action}"] += 1


@app.post("/v1/precheck")
def precheck(body: PrecheckIn, p: Principal | None = Depends(principal("client"))):
    t0 = time.perf_counter()
    body.tenant = tenant_of(p, body.tenant)
    METRICS["precheck_total"] += 1
    images = [base64.b64decode(b) for b in body.images_b64] + [_fetch(u) for u in body.image_urls]
    for im in images:
        try:
            decode_image(im)
        except ValueError:
            raise HTTPException(422, "an input image could not be decoded")
    h = sha256((body.prompt or "") + "|" + "|".join(sha256(i) for i in images))
    key = runtime.cache_key(body.tenant, h)
    cached = runtime.cache_get(key)
    chk = runtime.checker()
    if cached:
        METRICS["precheck_cache_hits"] += 1
        d = Decision(cached["action"], cached["reasons"], cached["scores"])
        info = {"cached": True, "timings_ms": {}, "flags": cached.get("flags", [])}
    else:
        if images:
            with runtime.vision_lock:
                d, info = chk.precheck(body.tenant, body.prompt, images)
        else:
            d, info = chk.precheck(body.tenant, body.prompt, [])
        runtime.cache_put(key, {**d.to_dict(), "flags": info.get("flags", [])})
    media = [runtime.store_media(sha256(i), i, ".img") for i in images] if d.action != "allow" else []
    latency = (time.perf_counter() - t0) * 1000
    with db.pool().connection() as conn:
        cid = db.insert_check(conn, tenant=body.tenant, stage="pre",
                              subject="mixed" if images and body.prompt else ("image" if images else "prompt"),
                              content_hash=h, input={"prompt": body.prompt, "media": media},
                              decision=d, policy_version=chk.policy(body.tenant).version,
                              model_versions=chk.model_versions, latency_ms=latency)
    _count(d.action)
    return {"check_id": cid, **d.to_dict(), **info, "latency_ms": round(latency, 2)}


@app.post("/v1/postcheck", status_code=202)
def postcheck(body: PostcheckIn, p: Principal | None = Depends(principal("client", cost=5))):
    body.tenant = tenant_of(p, body.tenant)
    if not body.media_url and not body.media_b64:
        raise HTTPException(422, "media_url or media_b64 required")
    data = base64.b64decode(body.media_b64) if body.media_b64 else _fetch(body.media_url)
    h = sha256(data)
    name = runtime.store_media(h, data, ".mp4" if body.kind == "video" else ".img")
    with db.pool().connection() as conn:
        cid = db.insert_check(conn, tenant=body.tenant, stage="post", subject=body.kind, content_hash=h,
                              input={"media": [name], "magic_hour_project_id": body.magic_hour_project_id,
                                     "source_url": body.media_url},
                              status="queued", webhook_url=body.webhook_url)
    Queue("postcheck", connection=runtime.rds()).enqueue(
        "modpipe.worker.run_postcheck", cid, job_timeout=600, retry=None, result_ttl=3600, job_id=f"post-{cid}")
    METRICS["postcheck_enqueued"] += 1
    return {"check_id": cid, "status": "queued"}


@app.get("/v1/checks/{cid}")
def get_check(cid: str, p: Principal | None = Depends(principal("client", cost=0.2))):
    with db.pool().connection() as conn:
        row = conn.execute("SELECT * FROM checks WHERE id=%s", (cid,)).fetchone()
    if not row:
        raise HTTPException(404)
    _own(p, row["tenant"])
    return row


@app.post("/v1/identities")
async def enroll(identity_id: str = Form(...), name: str = Form(...), tenant: str | None = Form(None),
                 photos: list[UploadFile] = File(...), _: Principal | None = Depends(principal("admin"))):
    chk = runtime.checker()
    vecs = []
    for ph in photos:
        img = decode_image(await ph.read())
        with runtime.vision_lock:
            faces = chk.faces.faces(img)
        if len(faces) != 1:
            raise HTTPException(422, f"{ph.filename}: expected exactly one face, found {len(faces)}")
        vecs.append(faces[0].vec)
    with db.pool().connection() as conn:
        v = db.enroll_identity(conn, identity_id, name, vecs, tenant, source="api")
    return {"identity_id": identity_id, "embeddings": len(vecs), "registry_version": v}


@app.post("/v1/logos")
async def add_logo(logo_id: str = Form(...), name: str = Form(...), tenant: str | None = Form(None),
                   image: UploadFile = File(...), _: Principal | None = Depends(principal("admin"))):
    data = await image.read()
    from .image.logos import LogoIndex
    try:
        n = LogoIndex().add(logo_id, decode_image(data))
    except ValueError as e:
        raise HTTPException(422, str(e))
    with db.pool().connection() as conn:
        v = db.add_logo(conn, logo_id, name, data, tenant)
    return {"logo_id": logo_id, "keypoints": n, "registry_version": v}


@app.post("/v1/review/claim")
def claim(reviewer: str | None = None, tenant: str | None = None, lease_s: int = Query(300, ge=30, le=3600),
          p: Principal | None = Depends(principal("reviewer"))):
    if p is not None:
        reviewer = p.name                      # reviewers act as themselves, not a query param
        if p.tenant != "*":
            tenant = p.tenant                  # a tenant's own moderators see only their queue
    if not reviewer:
        raise HTTPException(422, "reviewer required")
    with db.pool().connection() as conn:
        item = db.claim(conn, reviewer, lease_s, tenant)
    if not item:
        return {"empty": True}
    item["media_urls"] = [sign_media(m) for m in (item["check"]["input"] or {}).get("media") or []]
    return item


@app.post("/v1/review/{item_id}")
def resolve(item_id: int, body: ResolveIn, p: Principal | None = Depends(principal("reviewer"))):
    if p is not None:
        body.reviewer = p.name
    if not body.reviewer:
        raise HTTPException(422, "reviewer required")
    with db.pool().connection() as conn:
        ok = db.resolve(conn, item_id, body.reviewer, body.decision, body.labels, body.note)
    if not ok:
        raise HTTPException(409, "lease expired or item claimed by someone else")
    return {"ok": True}


@app.get("/v1/review/stats")
def stats(_: Principal | None = Depends(principal("reviewer", cost=0.2))):
    with db.pool().connection() as conn:
        return db.queue_stats(conn)


@app.get("/v1/media/{name}")
def media(name: str, exp: int = 0, sig: str = ""):
    if auth.AUTH_MODE != "open":
        good = hmac.new(MEDIA_KEY, f"{name}:{exp}".encode(), hashlib.sha256).hexdigest()[:32]
        if exp < time.time() or not hmac.compare_digest(good, sig):
            raise HTTPException(403, "media URL expired or not signed")
    p = (runtime.MEDIA_DIR / name).resolve()
    if p.parent != runtime.MEDIA_DIR.resolve() or not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="video/mp4" if name.endswith(".mp4") else "image/jpeg")


@app.get("/healthz")
def healthz():
    return {"ok": True, "registry_version": runtime.checker().registry_version}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return "".join(f"modpipe_{k} {v}\n" for k, v in METRICS.items())
