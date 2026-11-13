"""The moderation engine: prompts, images and videos in; scores, evidence and a decision out.

One Checker per process. It loads every model once and keeps the face gallery / logo index in
memory, reloading them when the registry version in Postgres moves (see db.bump_registry).
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .image.faces import FaceEmbedder, IdentityGallery, distance_to_score
from .image.logos import LogoIndex
from .image.nsfw import NSFWDetector
from .policy import Decision, Policy, decide
from .text.model import PromptClassifier
from .video import sample_frames

ROOT = Path(__file__).resolve().parents[1]
OBFUSCATION_FLAGS = {"homoglyph", "zero_width", "spaced", "leet"}


def sha256(b: bytes | str) -> str:
    return hashlib.sha256(b.encode() if isinstance(b, str) else b).hexdigest()


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not a decodable image")
    return img


@dataclass
class MediaResult:
    scores: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, dict] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    frames: int = 1

    def merge(self, other: "MediaResult", t: float | None = None) -> None:
        for k, v in other.scores.items():
            if v > self.scores.get(k, -1):
                self.scores[k] = v
                ev = dict(other.evidence.get(k, {}))
                if t is not None:
                    ev["t"] = round(t, 2)
                self.evidence[k] = ev
        for k, v in other.timings_ms.items():
            self.timings_ms[k] = self.timings_ms.get(k, 0) + v


class Checker:
    def __init__(self, policy_path: str | Path = ROOT / "policy.yaml", model_dir: str | Path = ROOT / "models" / "prompt",
                 load_vision: bool = True):
        self.policy_path = Path(policy_path)
        self._policies: dict[str | None, Policy] = {}
        self.text = PromptClassifier(model_dir)
        self.nsfw = NSFWDetector() if load_vision else None
        self.faces = FaceEmbedder() if load_vision else None
        self.gallery = IdentityGallery()
        self.logos = LogoIndex()
        self.registry_version = -1

    # ---- registries ----
    def load_registry(self, identities: list[tuple[str, np.ndarray]], logos: list[tuple[str, bytes]], version: int) -> None:
        self.gallery.load(identities, version)
        idx = LogoIndex()
        for name, data in logos:
            idx.add(name, decode_image(data))
        self.logos = idx
        self.registry_version = version

    def policy(self, tenant: str | None) -> Policy:
        if tenant not in self._policies:
            self._policies[tenant] = Policy.load(self.policy_path, tenant)
        return self._policies[tenant]

    @property
    def model_versions(self) -> dict:
        return {"prompt": self.text.version, "nsfw": "nudenet-320n", "faces": "dlib-resnet-v1",
                "registry": self.registry_version}

    # ---- detectors ----
    def check_prompt(self, prompt: str) -> tuple[dict[str, float], set[str], dict]:
        scores, feat = self.text.score(prompt)
        return scores, feat.flags & OBFUSCATION_FLAGS, {"normalized": feat.text, "gazetteer": feat.hits}

    def check_images(self, images: list[np.ndarray], faces: bool = True, logos: bool = True) -> list[MediaResult]:
        out = [MediaResult() for _ in images]
        t0 = time.perf_counter()
        for r, n in zip(out, self.nsfw.detect_batch(images)):
            r.scores["nsfw_exposed"] = round(n["score"], 4)
            if n["parts"]:
                r.evidence["nsfw_exposed"] = {"parts": n["parts"]}
        nsfw_ms = (time.perf_counter() - t0) * 1000 / max(1, len(images))
        for img, r in zip(images, out):
            r.timings_ms["nsfw"] = nsfw_ms
            if faces and len(self.gallery.ids):
                t1 = time.perf_counter()
                best = (None, 9.9, None)
                for f in self.faces.faces(img):
                    ident, d = self.gallery.match(f.vec)
                    if d < best[1]:
                        best = (ident, d, f.box)
                r.scores["protected_face"] = round(distance_to_score(best[1]), 4)
                if best[0] is not None and r.scores["protected_face"] > 0:
                    r.evidence["protected_face"] = {"identity": best[0], "distance": round(best[1], 4), "box": best[2]}
                r.timings_ms["faces"] = (time.perf_counter() - t1) * 1000
            if logos and self.logos.refs:
                t1 = time.perf_counter()
                hits = self.logos.detect(img)
                r.scores["logo"] = round(hits[0].score, 4) if hits else 0.0
                if hits:
                    r.evidence["logo"] = {"logo": hits[0].name, "inliers": hits[0].inliers, "quad": hits[0].quad}
                r.timings_ms["logos"] = (time.perf_counter() - t1) * 1000
        return out

    def check_video(self, path: str, early_exit: Policy | None = None, **sample_kw) -> MediaResult:
        t0 = time.perf_counter()
        frames, info = sample_frames(path, **sample_kw)
        total = MediaResult(frames=len(frames))
        total.timings_ms["decode"] = (time.perf_counter() - t0) * 1000
        # NSFW is cheap and batched: run it on every sampled frame first. Faces/logos are the
        # expensive part; if a frame already blocks, the clip is blocked and we stop there.
        for i in range(0, len(frames), 8):
            chunk = frames[i:i + 8]
            for fr, res in zip(chunk, self.check_images([f.image for f in chunk])):
                total.merge(res, fr.t)
            if early_exit and decide(early_exit, total.scores).action == "block":
                total.evidence.setdefault("_meta", {})["early_exit_after_frames"] = i + len(chunk)
                break
        total.evidence.setdefault("_meta", {}).update(info)
        return total

    # ---- end to end ----
    def precheck(self, tenant: str | None, prompt: str | None, images: list[bytes] = ()) -> tuple[Decision, dict]:
        """Synchronous: called before spending credits on a generation."""
        pol = self.policy(tenant)
        t0 = time.perf_counter()
        scores: dict[str, float] = {}
        flags: set[str] = set()
        evidence: dict[str, dict] = {}
        timings: dict[str, float] = {}
        if prompt:
            s, flags, ev = self.check_prompt(prompt)
            scores.update(s)
            timings["prompt"] = (time.perf_counter() - t0) * 1000
            if ev["gazetteer"]:
                for kind, label in (("<person>", "real_person"), ("<brand>", "trademark"), ("<character>", "trademark")):
                    names = [h[1] for h in ev["gazetteer"] if h[0] == kind]
                    if names:
                        evidence.setdefault(label, {})["mentions"] = names
        if images:
            agg = MediaResult()
            for r in self.check_images([decode_image(b) for b in images]):
                agg.merge(r)
            for k, v in agg.scores.items():
                scores[k] = max(scores.get(k, 0.0), v)
            evidence.update(agg.evidence)
            timings.update(agg.timings_ms)
        d = decide(pol, scores, flags or None, evidence)
        timings["total"] = (time.perf_counter() - t0) * 1000
        return d, {"timings_ms": {k: round(v, 3) for k, v in timings.items()}, "flags": sorted(flags)}

    def postcheck_image(self, tenant: str | None, data: bytes) -> tuple[Decision, dict]:
        pol = self.policy(tenant)
        t0 = time.perf_counter()
        r = self.check_images([decode_image(data)])[0]
        d = decide(pol, r.scores, None, r.evidence)
        return d, {"timings_ms": {**r.timings_ms, "total": (time.perf_counter() - t0) * 1000}, "frames": 1}

    def postcheck_video(self, tenant: str | None, path: str) -> tuple[Decision, dict]:
        pol = self.policy(tenant)
        t0 = time.perf_counter()
        r = self.check_video(path, early_exit=pol)
        d = decide(pol, r.scores, None, r.evidence)
        return d, {"timings_ms": {**r.timings_ms, "total": (time.perf_counter() - t0) * 1000},
                   "frames": r.frames, "video": r.evidence.get("_meta")}
