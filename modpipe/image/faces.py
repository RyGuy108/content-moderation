"""Likeness protection: does an enrolled ("protected") person's face appear in this image?

Detection: OpenCV YuNet (box + 5 landmarks). Each face is embedded twice:
  * SFace (MobileFaceNet-style, ArcFace-trained), aligned with YuNet's landmarks, averaged
    with its horizontal flip, L2-normalised;
  * dlib ResNet-34 (5-point alignment), the round-1 model.
The two embeddings make mostly independent mistakes on generated faces, so they are fused:

    sim = cos(sface_a, sface_b) - FUSE_K * ||dlib_a - dlib_b||

Higher sim = more likely the same person. On the eval set this lifts recall at zero false
matches from 0.20 (dlib alone) to ~0.5 (bench/eval_images.py). A Face's vec is the 256-d
concatenation [sface(128), dlib(128)]; the gallery stores the same layout.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import dlib
import numpy as np

MODELS = Path(__file__).resolve().parents[2] / "models"
DIM = 256
FUSE_K = 0.5
# sim -> [0, 1] score, linear between S_NONE (0) and S_SURE (1). Calibrated on the dev split
# by bench/eval_images.py so that policy's review threshold (0.5) sits at ~0 dev false matches.
S_NONE, S_SURE = 0.17, 0.60


def similarity_to_score(s: float) -> float:
    return float(np.clip((s - S_NONE) / (S_SURE - S_NONE), 0.0, 1.0))


def fused_similarity(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """mat (n, 256), vec (256,) -> (n,) fused similarity."""
    cos = mat[:, :128] @ vec[:128]
    dist = np.linalg.norm(mat[:, 128:] - vec[None, 128:], axis=1)
    return cos - FUSE_K * dist


@dataclass
class Face:
    box: tuple[int, int, int, int]       # x, y, w, h in original pixels
    vec: np.ndarray                      # 256-d: [sface, dlib]
    conf: float = 1.0


class FaceEmbedder:
    def __init__(self, max_side: int = 1024, conf: float = 0.6, threads: int = 1):
        cv2.setNumThreads(threads)
        self.det = cv2.FaceDetectorYN.create(str(MODELS / "face_detection_yunet_2023mar.onnx"), "",
                                             (320, 320), conf, 0.3, 5000)
        self.sface = cv2.FaceRecognizerSF.create(str(MODELS / "face_recognition_sface_2021dec.onnx"), "")
        self.sp = dlib.shape_predictor(str(MODELS / "shape_predictor_5_face_landmarks.dat"))
        self.rec = dlib.face_recognition_model_v1(str(MODELS / "dlib_face_recognition_resnet_model_v1.dat"))
        self.max_side = max_side

    def detect(self, bgr: np.ndarray) -> np.ndarray:
        """(n, 15) YuNet rows: x, y, w, h, 5 landmark (x, y) pairs, confidence."""
        self.det.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, f = self.det.detect(bgr)
        return f if f is not None else np.zeros((0, 15), np.float32)

    def faces(self, bgr: np.ndarray, min_size: int = 40) -> list[Face]:
        h, w = bgr.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        img = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr
        rgb = None
        out = []
        for row in self.detect(img):
            if row[2] < min_size * s:
                continue
            crop = self.sface.alignCrop(img, row)
            v = self.sface.feature(crop)[0] + self.sface.feature(np.ascontiguousarray(crop[:, ::-1]))[0]
            v = v / (np.linalg.norm(v) + 1e-9)
            if rgb is None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x, y, bw, bh = (int(t) for t in row[:4])
            shape = self.sp(rgb, dlib.rectangle(max(0, x), max(0, y), x + bw, y + bh))
            d = np.asarray(self.rec.compute_face_descriptor(rgb, shape, 0), dtype=np.float32)
            out.append(Face((int(x / s), int(y / s), int(bw / s), int(bh / s)),
                            np.concatenate([v.astype(np.float32), d]), float(row[14])))
        return out


class IdentityGallery:
    """In-memory view of enrolled identities. Several embeddings per identity are allowed;
    an identity's similarity is its best-matching enrolled embedding."""

    def __init__(self):
        self.ids: list[str] = []
        self.mat = np.zeros((0, DIM), np.float32)
        self.version = 0
        self.skipped = 0

    def load(self, rows: list[tuple[str, np.ndarray]], version: int = 0) -> None:
        # rows enrolled by an older embedder (128-d dlib only) can't be compared; they need re-enrollment
        good = [r for r in rows if len(r[1]) == DIM]
        self.skipped = len(rows) - len(good)
        self.ids = [r[0] for r in good]
        self.mat = np.stack([r[1] for r in good]).astype(np.float32) if good else np.zeros((0, DIM), np.float32)
        self.version = version

    def add(self, identity: str, vec: np.ndarray) -> None:
        self.ids.append(identity)
        self.mat = np.vstack([self.mat, vec[None].astype(np.float32)])

    def match(self, vec: np.ndarray) -> tuple[str | None, float]:
        """Best identity and its fused similarity (higher = closer)."""
        if not len(self.ids):
            return None, -9.9
        s = fused_similarity(self.mat, vec)
        i = int(s.argmax())
        return self.ids[i], float(s[i])

    def similarities(self, vec: np.ndarray) -> dict[str, float]:
        """Per-identity best similarity."""
        s = fused_similarity(self.mat, vec)
        out: dict[str, float] = {}
        for ident, v in zip(self.ids, s):
            out[ident] = max(out.get(ident, -9.9), float(v))
        return out
