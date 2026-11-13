"""Pick which frames of a generated video are worth checking.

Checking every frame of a 5 s, 24 fps clip is 120 detector passes. Generated video is mostly
smooth motion, so we decode cheaply, keep a frame when the scene changes (HSV histogram jump)
or when `max_gap_s` has passed, then drop near-duplicates by 64-bit dHash. Typical result: 6-12
frames per clip. bench/eval_video.py measures what this costs in recall vs checking everything.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Frame:
    t: float
    index: int
    image: np.ndarray


def _dhash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def _hist(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(cv2.resize(bgr, (64, 36)), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def sample_frames(path: str, *, step_s: float = 0.25, max_gap_s: float = 1.0, scene_thresh: float = 0.35,
                  dedupe_bits: int = 5, max_frames: int = 48, uniform: float | None = None) -> tuple[list[Frame], dict]:
    """uniform=N: plain uniform sampling at N fps (the baseline strategy)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * (1.0 / uniform if uniform else step_s))))
    kept: list[Frame] = []
    hashes: list[int] = []
    last_hist, last_t = None, -1e9
    idx = decoded = 0
    while True:
        if idx % step:
            ok = cap.grab()          # skip decode work for frames we won't look at
            if not ok:
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        decoded += 1
        t = idx / fps
        idx += 1
        if uniform:
            kept.append(Frame(t, idx - 1, frame))
            if len(kept) >= max_frames:
                break
            continue
        hist = _hist(frame)
        changed = last_hist is None or cv2.compareHist(last_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > scene_thresh
        if not changed and t - last_t < max_gap_s:
            continue
        dh = _dhash(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if not changed and any(bin(dh ^ h).count("1") <= dedupe_bits for h in hashes):
            last_t = t        # visually the same as something we already kept
            continue
        kept.append(Frame(t, idx - 1, frame))
        hashes.append(dh)
        last_hist, last_t = hist, t
        if len(kept) >= max_frames:
            break
    cap.release()
    return kept, {"fps": fps, "total_frames": total, "decoded": decoded, "kept": len(kept)}
