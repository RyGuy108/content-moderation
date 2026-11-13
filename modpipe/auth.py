"""Tenant API keys and per-key rate limiting.

Every request carries `Authorization: Bearer mp_...` (or `X-API-Key`). The key decides the
tenant, so a client can't pick a looser tenant policy by putting another name in the body, and
it decides the role:

  client    precheck / postcheck / read its own checks
  reviewer  the review queue (claim/resolve as the key's name) + media
  admin     everything, including enrolling protected faces and logos

Rate limiting is a token bucket per key held in Redis and updated by one Lua script, so it is
atomic across any number of API processes. If Redis is down the limiter fails open (moderation
must keep working; the decision cache is already best-effort the same way).

Keys are stored as SHA-256 only. Lookups are cached in-process for KEY_CACHE_S seconds, so
a revoked key stops working within that window.

    python -m modpipe.auth create --tenant demo --name gateway [--role client] [--rate 20 --burst 40]
    python -m modpipe.auth list
    python -m modpipe.auth revoke <prefix>
"""
from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import threading
import time
from dataclasses import dataclass

import redis
from fastapi import Header, HTTPException, Response

from . import db

AUTH_MODE = os.environ.get("AUTH_MODE", "keys")      # "keys" | "open" (local dev: tenant from body)
KEY_CACHE_S = float(os.environ.get("KEY_CACHE_S", "30"))
ROLES = {"client": 0, "reviewer": 1, "admin": 2}


@dataclass(frozen=True)
class Principal:
    tenant: str
    name: str
    role: str
    key_id: int | None = None
    rate_per_s: float = 0.0
    burst: int = 0

    def can(self, role: str) -> bool:
        # reviewers are not clients and vice versa; admin can do both
        return self.role == "admin" or self.role == role


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def create_key(conn, tenant: str, name: str, role: str = "client", rate_per_s: float = 20, burst: int = 40) -> str:
    if role not in ROLES:
        raise ValueError(f"role must be one of {sorted(ROLES)}")
    key = "mp_" + secrets.token_urlsafe(30)
    conn.execute("""INSERT INTO api_keys (tenant, name, role, key_hash, prefix, rate_per_s, burst)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)""", (tenant, name, role, hash_key(key), key[:10], rate_per_s, burst))
    conn.execute("INSERT INTO audit_log (actor, action, target, detail) VALUES ('cli','apikey.create',%s,%s)",
                 (key[:10], db.Jsonb({"tenant": tenant, "name": name, "role": role})))
    return key


def revoke(conn, prefix: str) -> int:
    n = conn.execute("UPDATE api_keys SET revoked_at=now() WHERE prefix=%s AND revoked_at IS NULL", (prefix,)).rowcount
    conn.execute("INSERT INTO audit_log (actor, action, target) VALUES ('cli','apikey.revoke',%s)", (prefix,))
    return n


_cache: dict[str, tuple[float, Principal | None]] = {}
_cache_lock = threading.Lock()


def lookup(key: str) -> Principal | None:
    h = hash_key(key)
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(h)
        if hit and now - hit[0] < KEY_CACHE_S:
            return hit[1]
    with db.pool().connection() as conn:
        row = conn.execute("""UPDATE api_keys SET last_used_at=now() WHERE key_hash=%s AND revoked_at IS NULL
                              RETURNING id, tenant, name, role, rate_per_s, burst""", (h,)).fetchone()
    p = Principal(row["tenant"], row["name"], row["role"], row["id"], row["rate_per_s"], row["burst"]) if row else None
    with _cache_lock:
        if len(_cache) > 10_000:
            _cache.clear()
        _cache[h] = (now, p)
    return p


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


# Token bucket: KEYS[1] = bucket hash; ARGV = rate/s, burst, now (s), cost.
# Returns {allowed (0/1), tokens_left (floor), retry_after_ms}.
_BUCKET = """
local rate, burst, now, cost = tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3]), tonumber(ARGV[4])
local s = redis.call('HMGET', KEYS[1], 't', 'ts')
local tokens, ts = tonumber(s[1]), tonumber(s[2])
if tokens == nil then tokens, ts = burst, now end
tokens = math.min(burst, tokens + math.max(0, now - ts) * rate)
local ok, wait = 0, 0
if tokens >= cost then tokens = tokens - cost; ok = 1 else wait = math.ceil((cost - tokens) / rate * 1000) end
redis.call('HSET', KEYS[1], 't', tokens, 'ts', now)
redis.call('PEXPIRE', KEYS[1], math.ceil(burst / rate * 1000) + 1000)
return {ok, math.floor(tokens), wait}
"""
_script = None


def take(p: Principal, cost: float = 1.0) -> tuple[bool, int, int]:
    """Spend `cost` tokens from the key's bucket. Returns (allowed, remaining, retry_after_ms)."""
    global _script
    if p.key_id is None or p.rate_per_s <= 0:
        return True, -1, 0
    from . import runtime
    try:
        if _script is None:
            _script = runtime.rds().register_script(_BUCKET)
        ok, left, wait = _script(keys=[f"rl:key:{p.key_id}"], args=[p.rate_per_s, p.burst, time.time(), cost])
        return bool(ok), int(left), int(wait)
    except redis.RedisError:
        return True, -1, 0          # fail open


def principal(role: str, cost: float = 1.0):
    """FastAPI dependency: authenticate, check role, and rate-limit. In AUTH_MODE=open every
    caller is an unlimited admin and endpoints fall back to the tenant named in the body."""
    def dep(response: Response, authorization: str | None = Header(None), x_api_key: str | None = Header(None)) -> Principal | None:
        if AUTH_MODE == "open":
            return None
        key = x_api_key or (authorization[7:] if authorization and authorization.lower().startswith("bearer ") else None)
        if not key:
            raise HTTPException(401, "missing API key", headers={"WWW-Authenticate": "Bearer"})
        p = lookup(key)
        if p is None:
            raise HTTPException(401, "invalid or revoked API key", headers={"WWW-Authenticate": "Bearer"})
        if not p.can(role):
            raise HTTPException(403, f"this key's role ({p.role}) can't call {role} endpoints")
        ok, left, wait = take(p, cost)
        if left >= 0:
            response.headers["X-RateLimit-Limit"] = str(p.burst)
            response.headers["X-RateLimit-Remaining"] = str(left)
        if not ok:
            raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(max(1, -(-wait // 1000))),
                                                                     "X-RateLimit-Remaining": "0"})
        return p
    return dep


def tenant_of(p: Principal | None, body_tenant: str | None) -> str:
    return p.tenant if p is not None else (body_tenant or "default")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m modpipe.auth")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--tenant", required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--role", default="client", choices=sorted(ROLES))
    c.add_argument("--rate", type=float, default=20)
    c.add_argument("--burst", type=int, default=40)
    sub.add_parser("list")
    r = sub.add_parser("revoke")
    r.add_argument("prefix")
    a = ap.parse_args(argv)
    db.migrate()
    with db.pool().connection() as conn:
        if a.cmd == "create":
            print(create_key(conn, a.tenant, a.name, a.role, a.rate, a.burst))
        elif a.cmd == "list":
            for row in conn.execute("SELECT prefix, tenant, name, role, rate_per_s, burst, last_used_at, revoked_at "
                                    "FROM api_keys ORDER BY id"):
                print(row)
        else:
            print("revoked", revoke(conn, a.prefix))
    db.close()


if __name__ == "__main__":
    main()
