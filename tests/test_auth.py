"""API keys: authentication, roles, tenant binding, revocation, rate limits, signed media URLs."""
import pytest


@pytest.fixture(scope="module")
def app_client(pg):
    from fastapi.testclient import TestClient

    from modpipe.api import app
    with TestClient(app) as c:
        yield c


def _key(tenant, name, **kw):
    from modpipe import auth, db
    with db.pool().connection() as conn:
        return auth.create_key(conn, tenant, name, **kw)


def H(k):
    return {"Authorization": f"Bearer {k}"}


def test_requires_a_valid_key(app_client):
    assert app_client.post("/v1/precheck", json={"prompt": "a cat"}).status_code == 401
    assert app_client.post("/v1/precheck", headers=H("mp_nope"), json={"prompt": "a cat"}).status_code == 401
    assert app_client.get("/healthz").status_code == 200


def test_roles_are_enforced(app_client):
    client = _key("default", "c-roles", rate_per_s=100, burst=100)
    rev = _key("*", "r-roles", role="reviewer", rate_per_s=100, burst=100)
    assert app_client.post("/v1/review/claim", headers=H(client)).status_code == 403
    assert app_client.post("/v1/precheck", headers=H(rev), json={"prompt": "a cat"}).status_code == 403
    assert app_client.post("/v1/logos", headers=H(rev), data={"logo_id": "x", "name": "x"},
                           files={"image": ("x.png", b"x", "image/png")}).status_code == 403


def test_key_tenant_overrides_body_tenant(app_client):
    """A default-tenant key can't borrow the adult platform's looser policy by naming it."""
    k = _key("default", "c-spoof", rate_per_s=100, burst=100)
    r = app_client.post("/v1/precheck", headers=H(k),
                        json={"tenant": "adult-platform", "prompt": "a nude woman posing on a beach"}).json()
    assert r["action"] in ("review", "block")
    from modpipe import db
    with db.pool().connection() as conn:
        assert conn.execute("SELECT tenant FROM checks WHERE id=%s", (r["check_id"],)).fetchone()["tenant"] == "default"
    adult = _key("adult-platform", "c-adult", rate_per_s=100, burst=100)
    assert app_client.post("/v1/precheck", headers=H(adult),
                           json={"prompt": "a nude woman posing on a beach"}).json()["action"] == "allow"


def test_rate_limit_token_bucket(app_client):
    k = _key("default", "c-limited", rate_per_s=0.5, burst=3)
    codes = [app_client.post("/v1/precheck", headers=H(k), json={"prompt": f"a boat {i}"}) for i in range(5)]
    assert [c.status_code for c in codes] == [200, 200, 200, 429, 429]
    assert codes[0].headers["X-RateLimit-Remaining"] == "2"
    assert int(codes[3].headers["Retry-After"]) >= 1


def test_revoked_key_stops_working(app_client):
    from modpipe import auth, db
    k = _key("default", "c-revoke", rate_per_s=100, burst=100)
    assert app_client.post("/v1/precheck", headers=H(k), json={"prompt": "a tree"}).status_code == 200
    with db.pool().connection() as conn:
        assert auth.revoke(conn, k[:10]) == 1
    auth.clear_cache()      # otherwise revocation lands within KEY_CACHE_S
    assert app_client.post("/v1/precheck", headers=H(k), json={"prompt": "a tree"}).status_code == 401


def test_other_tenants_checks_are_hidden(app_client):
    a = _key("tenant-a", "c-a", rate_per_s=100, burst=100)
    b = _key("tenant-b", "c-b", rate_per_s=100, burst=100)
    cid = app_client.post("/v1/precheck", headers=H(a), json={"prompt": "a lighthouse"}).json()["check_id"]
    assert app_client.get(f"/v1/checks/{cid}", headers=H(a)).status_code == 200
    assert app_client.get(f"/v1/checks/{cid}", headers=H(b)).status_code == 404


def test_media_urls_are_signed(app_client):
    from modpipe.api import sign_media
    from modpipe import runtime
    runtime.MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    (runtime.MEDIA_DIR / "probe.img").write_bytes(b"\xff\xd8\xff")
    assert app_client.get("/v1/media/probe.img").status_code == 403
    assert app_client.get(sign_media("probe.img")).status_code == 200
    assert app_client.get(sign_media("probe.img", ttl=-5)).status_code == 403
    assert app_client.get(sign_media("probe.img").replace("sig=", "sig=0")).status_code == 403
