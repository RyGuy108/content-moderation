import os

import pytest


@pytest.fixture(scope="module")
def client(pg):
    from fastapi.testclient import TestClient

    from modpipe.api import app
    with TestClient(app) as c:
        yield c


def test_precheck_blocks_obfuscated_deepfake_prompt(client):
    r = client.post("/v1/precheck", json={"prompt": "t@ylor sw1ft n.u.d.e on a beach"}).json()
    assert r["action"] == "block"
    assert {x["label"] for x in r["reasons"]} >= {"sexual", "real_person_sexual"}
    assert set(r["flags"]) >= {"leet", "spaced"}


def test_precheck_allows_and_caches(client):
    a = client.post("/v1/precheck", json={"prompt": "a golden retriever chasing bubbles"}).json()
    assert a["action"] == "allow"
    b = client.post("/v1/precheck", json={"prompt": "a golden retriever chasing bubbles"}).json()
    assert b.get("cached") is True or os.environ.get("NO_REDIS")


def test_review_roundtrip(client):
    r = client.post("/v1/precheck", json={"prompt": "nike logo on a hoodie"}).json()
    assert r["action"] == "review"
    item = client.post("/v1/review/claim", params={"reviewer": "t1"}).json()
    assert item["check"]["id"]
    ok = client.post(f"/v1/review/{item['id']}", json={"reviewer": "t1", "decision": "allow"}).json()
    assert ok["ok"]
