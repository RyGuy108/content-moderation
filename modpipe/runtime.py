"""Process-level wiring shared by the API and the workers: one Checker, a lock around the
vision models (OpenCV DNN nets are not thread-safe), a registry watcher, the Redis cache, and
media storage."""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

import psycopg
import redis

from . import db
from .checker import Checker

log = logging.getLogger("modpipe")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
MEDIA_DIR = Path(os.environ.get("MEDIA_DIR", Path(__file__).resolve().parents[1] / "var" / "media"))
CACHE_TTL = int(os.environ.get("CACHE_TTL", "86400"))

_checker: Checker | None = None
_lock = threading.Lock()
vision_lock = threading.Lock()
_redis: redis.Redis | None = None


def checker() -> Checker:
    global _checker
    with _lock:
        if _checker is None:
            _checker = Checker()
            reload_registry()
            threading.Thread(target=_watch_registry, daemon=True, name="registry-watch").start()
    return _checker


def reload_registry() -> None:
    with db.pool().connection() as conn:
        v = db.registry_version(conn)
        ids, logos = db.load_identities(conn), db.load_logos(conn)
    with vision_lock:
        _checker.load_registry(ids, logos, v)
    log.info("registry v%s loaded: %d face embeddings, %d logos", v, len(ids), len(logos))


def _watch_registry() -> None:
    """LISTEN for enrollments so every API/worker process picks them up within milliseconds."""
    while True:
        try:
            with psycopg.connect(db.DATABASE_URL, autocommit=True) as conn:
                conn.execute("LISTEN registry_changed")
                # catch anything that changed between load and LISTEN
                if conn.execute("SELECT v FROM registry_version").fetchone()[0] != _checker.registry_version:
                    reload_registry()
                for _ in conn.notifies():
                    reload_registry()
        except Exception as e:  # reconnect forever; stale registry beats a dead process
            log.warning("registry watcher: %s", e)
            threading.Event().wait(2)


def rds() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.Redis.from_url(REDIS_URL)
    return _redis


def cache_key(tenant: str, content_hash: str) -> str:
    c = checker()
    return f"mod:v1:{tenant}:{c.policy(tenant).version}:{c.text.version}:{c.registry_version}:{content_hash}"


def cache_get(key: str) -> dict | None:
    try:
        raw = rds().get(key)
        return json.loads(raw) if raw else None
    except redis.RedisError:
        return None          # the cache is an optimization; never fail a check because of it


def cache_put(key: str, value: dict) -> None:
    try:
        rds().set(key, json.dumps(value), ex=CACHE_TTL)
    except redis.RedisError:
        pass


def store_media(content_hash: str, data: bytes, ext: str) -> str:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    p = MEDIA_DIR / f"{content_hash}{ext}"
    if not p.exists():
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.rename(p)
    return p.name
