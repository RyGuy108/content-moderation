"""NSFW detection: NudeNet's YOLOv8n body-part detector (ONNX, bundled in the `nudenet` wheel),
run through onnxruntime with batching so a video's sampled frames go through in one call.

We keep NudeNet's pre/post-processing but run it ourselves: the package API takes one image at
a time and drops detections under 0.2, and we want batches and the raw scores for calibration.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import onnxruntime as ort

LABELS = ["FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED",
          "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED", "ANUS_EXPOSED", "FEET_EXPOSED",
          "BELLY_COVERED", "FEET_COVERED", "ARMPITS_COVERED", "ARMPITS_EXPOSED", "FACE_MALE",
          "BELLY_EXPOSED", "MALE_GENITALIA_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED",
          "BUTTOCKS_COVERED"]
EXPOSED = {"FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED", "FEMALE_BREAST_EXPOSED",
           "BUTTOCKS_EXPOSED", "ANUS_EXPOSED"}
_EXPOSED_IDX = np.array([LABELS.index(l) for l in sorted(EXPOSED)])


def _model_path() -> str:
    import nudenet
    return os.path.join(os.path.dirname(nudenet.__file__), "320n.onnx")


class NSFWDetector:
    def __init__(self, size: int = 320, threads: int = 1):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(_model_path(), so, providers=["CPUExecutionProvider"])
        self.size = size

    def _prep(self, bgr: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = bgr.shape[:2]
        m = max(h, w)
        pad = cv2.copyMakeBorder(bgr, 0, m - h, 0, m - w, cv2.BORDER_CONSTANT)
        blob = cv2.dnn.blobFromImage(pad, 1 / 255.0, (self.size, self.size), (0, 0, 0), swapRB=True, crop=False)
        return blob[0], m / self.size

    def detect_batch(self, images: list[np.ndarray]) -> list[dict]:
        """Returns per image: {"score": max exposed-part confidence, "parts": [...top boxes]}"""
        if not images:
            return []
        preps = [self._prep(im) for im in images]
        out = self.sess.run(None, {"images": np.stack([p[0] for p in preps])})[0]  # (B, 22, A)
        results = []
        for b, (_, scale) in enumerate(preps):
            o = out[b]                      # (22, A): cx, cy, w, h, 18 class scores
            cls = o[4:]
            exposed = cls[_EXPOSED_IDX]     # (5, A)
            best = exposed.max(axis=0)
            parts = []
            for a in np.argsort(-best)[:3]:
                if best[a] < 0.2:
                    break
                lab = sorted(EXPOSED)[int(exposed[:, a].argmax())]
                cx, cy, w, h = (o[:4, a] * scale).tolist()
                parts.append({"class": lab, "score": round(float(best[a]), 4),
                              "box": [int(cx - w / 2), int(cy - h / 2), int(w), int(h)]})
            results.append({"score": float(best.max()) if best.size else 0.0, "parts": parts})
        return results

    def detect(self, image: np.ndarray) -> dict:
        return self.detect_batch([image])[0]
