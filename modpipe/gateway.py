"""Moderated Magic Hour client: pre-check before paying for a render, post-check before handing
the output to the end user.

    gw = ModeratedMagicHour(api="http://localhost:8000", tenant="acme")
    out = gw.image(prompt="...")                      # ai_image_generator
    out = gw.face_swap_photo(source=..., target=...)  # face_swap_photo

A blocked pre-check raises Blocked and costs no credits. A post-check that blocks quarantines
the output (the URL is never returned); "review" returns the check id so the caller can hold the
asset until a reviewer decides.
"""
from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from magic_hour import Client


class Blocked(Exception):
    def __init__(self, stage: str, check: dict):
        self.stage, self.check = stage, check
        top = ", ".join(r["label"] for r in check.get("reasons", [])[:3])
        super().__init__(f"{stage}-check blocked: {top}")


@dataclass
class Result:
    status: str                  # "allow" | "review"
    path: str | None
    project_id: str
    precheck: dict
    postcheck: dict
    credits: int


class ModeratedMagicHour:
    def __init__(self, api: str = "http://localhost:8000", tenant: str = "default", token: str | None = None):
        self.api = httpx.Client(base_url=api, timeout=60)
        self.tenant = tenant
        self.mh = Client(token=token or os.environ["MAGIC_HOUR_API_KEY"])

    def _pre(self, prompt: str | None, images: list[str] = ()) -> dict:
        body = {"tenant": self.tenant, "prompt": prompt,
                "images_b64": [base64.b64encode(Path(p).read_bytes()).decode() for p in images]}
        r = self.api.post("/v1/precheck", json=body).raise_for_status().json()
        if r["action"] == "block":
            raise Blocked("pre", r)
        return r

    def _post(self, path: str, kind: str, project_id: str) -> dict:
        data = base64.b64encode(Path(path).read_bytes()).decode()
        r = self.api.post("/v1/postcheck", json={"tenant": self.tenant, "kind": kind, "media_b64": data,
                                                 "magic_hour_project_id": project_id}).raise_for_status().json()
        for _ in range(600):
            c = self.api.get(f"/v1/checks/{r['check_id']}").json()
            if c["status"] in ("done", "error"):
                return c
            time.sleep(0.2)
        raise TimeoutError(r["check_id"])

    def _finish(self, pre, project_id, path, kind, credits) -> Result:
        post = self._post(path, kind, project_id)
        if post["action"] == "block":
            Path(path).unlink(missing_ok=True)       # quarantine: never hand it out
            raise Blocked("post", post)
        # a prompt that needed review still renders (so the reviewer can see the output), but the
        # asset is held until a human decides
        status = "review" if "review" in (pre["action"], post["action"]) else "allow"
        return Result(status, path, project_id, pre, post, credits)

    def _wait_image(self, pid: str, out_dir: Path) -> str:
        for _ in range(200):
            p = self.mh.v1.image_projects.get(id=pid)
            if p.status == "complete":
                path = out_dir / f"{pid}.png"
                path.write_bytes(httpx.get(p.downloads[0].url, timeout=60).content)
                return str(path)
            if p.status in ("error", "canceled"):
                raise RuntimeError(f"magic hour project {pid}: {p.status}")
            time.sleep(2)
        raise TimeoutError(pid)

    def image(self, prompt: str, out_dir: str = "var/outputs", **kw) -> Result:
        pre = self._pre(prompt)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        r = self.mh.v1.ai_image_generator.create(image_count=1, style={"prompt": prompt},
                                                 model=kw.get("model", "z-image-turbo"),
                                                 resolution=kw.get("resolution", "1k"), aspect_ratio="1:1")
        return self._finish(pre, r.id, self._wait_image(r.id, Path(out_dir)), "image", r.credits_charged)

    def face_swap_photo(self, source: str, target: str, out_dir: str = "var/outputs") -> Result:
        # The source face is checked against protected identities *before* the swap is paid for.
        pre = self._pre(None, [source, target])
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        r = self.mh.v1.face_swap_photo.create(assets={
            "face_swap_mode": "all-faces", "source_file_path": self.mh.v1.files.upload_file(source),
            "target_file_path": self.mh.v1.files.upload_file(target)})
        return self._finish(pre, r.id, self._wait_image(r.id, Path(out_dir)), "image", r.credits_charged)
