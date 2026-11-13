"""Likeness protection: does an enrolled ("protected") person's face appear in this image?

dlib HOG face detector -> 5-point alignment -> ResNet-34 128-d embedding (99.38% on LFW).
Enrolled identities live in Postgres (protected_identities / identity_embeddings); the matcher
holds them as one float32 matrix, so matching a face against N identities is one matrix-vector
product. Distance -> score is a fixed linear map calibrated on bench/eval_images.py's dev split.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import dlib
import numpy as np

MODELS = Path(__file__).resolve().parents[2] / "models"

# dlib's suggested same-person cutoff is 0.6; we map distance d to a [0, 1] score with
# score = 1 at d <= D_SURE and 0 at d >= D_NONE, linear in between.
D_SURE, D_NONE = 0.40, 0.60


def distance_to_score(d: float) -> float:
    return float(np.clip((D_NONE - d) / (D_NONE - D_SURE), 0.0, 1.0))


@dataclass
class Face:
    box: tuple[int, int, int, int]       # x, y, w, h in original pixels
    vec: np.ndarray                      # 128-d embedding


class FaceEmbedder:
    """Detection: OpenCV DNN res10 SSD (much better recall than dlib's HOG on non-frontal and
    stylized faces, ~15 ms on CPU). Embedding: dlib 5-point alignment + ResNet."""

    def __init__(self, max_side: int = 1024, conf: float = 0.5, threads: int = 1):
        cv2.setNumThreads(threads)
        self.ssd = cv2.dnn.readNetFromCaffe(str(MODELS / "deploy.prototxt"),
                                            str(MODELS / "res10_300x300_ssd_iter_140000_fp16.caffemodel"))
        self.conf = conf
        self.sp = dlib.shape_predictor(str(MODELS / "shape_predictor_5_face_landmarks.dat"))
        self.rec = dlib.face_recognition_model_v1(str(MODELS / "dlib_face_recognition_resnet_model_v1.dat"))
        self.max_side = max_side

    def detect(self, bgr: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        h, w = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.ssd.setInput(blob)
        det = self.ssd.forward()[0, 0]
        out = []
        for row in det:
            c = float(row[2])
            if c < self.conf:
                continue
            x0, y0, x1, y1 = (row[3:7] * [w, h, w, h]).astype(int)
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            if x1 - x0 > 8 and y1 - y0 > 8:
                out.append((x0, y0, x1 - x0, y1 - y0, c))
        return out

    def faces(self, bgr: np.ndarray, min_size: int = 40) -> list[Face]:
        h, w = bgr.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        img = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = []
        for x, y, bw, bh, _ in self.detect(img):
            if bw < min_size * s:
                continue
            r = dlib.rectangle(int(x), int(y), int(x + bw), int(y + bh))
            shape = self.sp(rgb, r)
            vec = np.asarray(self.rec.compute_face_descriptor(rgb, shape, 0), dtype=np.float32)
            out.append(Face((int(x / s), int(y / s), int(bw / s), int(bh / s)), vec))
        return out


class IdentityGallery:
    """In-memory view of enrolled identities. Several embeddings per identity are allowed;
    an identity's distance is its closest enrolled embedding."""

    def __init__(self):
        self.ids: list[str] = []
        self.mat = np.zeros((0, 128), np.float32)
        self.version = 0

    def load(self, rows: list[tuple[str, np.ndarray]], version: int = 0) -> None:
        self.ids = [r[0] for r in rows]
        self.mat = np.stack([r[1] for r in rows]).astype(np.float32) if rows else np.zeros((0, 128), np.float32)
        self.version = version

    def add(self, identity: str, vec: np.ndarray) -> None:
        self.ids.append(identity)
        self.mat = np.vstack([self.mat, vec[None].astype(np.float32)])

    def match(self, vec: np.ndarray) -> tuple[str | None, float]:
        if not len(self.ids):
            return None, 9.9
        d = np.linalg.norm(self.mat - vec[None], axis=1)
        i = int(d.argmin())
        return self.ids[i], float(d[i])
