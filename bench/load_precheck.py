"""HTTP load test for the synchronous pre-check (prompt only, cache bypassed with unique
prompts). Measures what the moderation hop adds to a generation request end to end:
HTTP + normalize + classify + policy + a Postgres insert of the decision."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


async def main(n=3000, conc=32, url="http://localhost:8000/v1/precheck"):
    rows = [json.loads(l)["text"] for l in open(ROOT / "data" / "prompts" / "test.jsonl")]
    prompts = [f"{rows[i % len(rows)]} #{i}" for i in range(n)]
    lat, actions = [], {}
    sem = asyncio.Semaphore(conc)
    key = os.environ.get("MODPIPE_API_KEY")   # mint one with a high limit: python -m modpipe.auth create --tenant load ...
    async with httpx.AsyncClient(timeout=30, limits=httpx.Limits(max_connections=conc),
                                 headers={"Authorization": f"Bearer {key}"} if key else {}) as c:
        async def one(p):
            async with sem:
                t0 = time.perf_counter()
                r = await c.post(url, json={"tenant": "load", "prompt": p})
                lat.append((time.perf_counter() - t0) * 1000)
                a = r.json()["action"]
                actions[a] = actions.get(a, 0) + 1
        t0 = time.perf_counter()
        await asyncio.gather(*(one(p) for p in prompts))
        wall = time.perf_counter() - t0
    res = {"requests": n, "concurrency": conc, "req_per_s": round(n / wall, 1),
           "p50_ms": round(float(np.percentile(lat, 50)), 2), "p95_ms": round(float(np.percentile(lat, 95)), 2),
           "p99_ms": round(float(np.percentile(lat, 99)), 2), "actions": actions}
    print(res)
    json.dump(res, open(ROOT / "results" / "load.json", "w"), indent=1)


if __name__ == "__main__":
    asyncio.run(main(*(int(a) for a in sys.argv[1:3])))
