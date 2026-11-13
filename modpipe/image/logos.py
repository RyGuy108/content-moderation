"""Logo detection against a library of registered marks.

SIFT keypoints from each reference logo (and a few augmented views of it) go into one FLANN
KD-tree. For a query image: every descriptor votes for its nearest mark (retrieval), then the
top few marks are verified by matching their keypoints into the image, Lowe ratio test, and a
RANSAC similarity transform; the score comes from the number of geometrically consistent inliers. Geometry is what separates "the actual mark, warped onto
a mug" from "a few similar-looking corners". No training: registering a new mark is one upload.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

INLIERS_SURE = 30     # inliers at which score reaches 1.0 (calibrated on the dev split)
INLIERS_MIN = 8       # below this, score is 0


def inliers_to_score(n: int) -> float:
    return float(np.clip((n - INLIERS_MIN) / (INLIERS_SURE - INLIERS_MIN), 0.0, 1.0))


def augmented_views(bgr: np.ndarray) -> list[np.ndarray]:
    ink = 1.0 - cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    ink = np.clip(ink * 2.0, 0, 1)[..., None]
    h, w = bgr.shape[:2]
    rng = np.random.default_rng(0)
    noise = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 3)
    texture = np.clip(150 + 40 * noise / (noise.std() + 1e-6), 0, 255)[..., None].repeat(3, 2)
    out = []
    for bg in (np.full_like(bgr, 128, np.float32), np.full_like(bgr, 40, np.float32), texture):
        out.append((bg * (1 - ink) + bgr.astype(np.float32) * ink).astype(np.uint8))
    return out + tilted_views(bgr)


def tilted_views(bgr: np.ndarray) -> list[np.ndarray]:
    """The mark seen from four off-axis angles (one edge foreshortened by 12-15%). SIFT is only
    invariant to similarity transforms; a print on a hanging shirt or a mug is a perspective
    view, so (as in ASIFT) we store a few simulated tilts rather than hoping one frontal view
    matches. On the eval set this alone took recall at 0 false hits from 0.33 to 0.47."""
    h, w = bgr.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    out = []
    for left, right, top, bottom in ((.15, 0, 0, 0), (0, .15, 0, 0), (0, 0, .12, 0), (0, 0, 0, .12)):
        dst = src.copy()
        dst[0, 1] += left * h; dst[3, 1] -= left * h           # noqa: E702  squeeze the left edge
        dst[1, 1] += right * h; dst[2, 1] -= right * h         # noqa: E702  ... the right edge
        dst[0, 0] += top * w; dst[1, 0] -= top * w             # noqa: E702  ... the top edge
        dst[3, 0] += bottom * w; dst[2, 0] -= bottom * w       # noqa: E702  ... the bottom edge
        M = cv2.getPerspectiveTransform(src, dst)
        out.append(cv2.warpPerspective(bgr, M, (w, h), borderValue=(255, 255, 255)))
    return out


@dataclass
class _Ref:
    name: str
    kps: np.ndarray            # (K, 2) keypoint xy
    shape: tuple[int, int]
    desc_range: tuple[int, int] = (0, 0)


@dataclass
class LogoHit:
    name: str
    inliers: int
    score: float
    quad: list[list[int]] = field(default_factory=list)   # projected logo corners in the image


class LogoIndex:
    def __init__(self, max_side: int = 1024, nfeatures: int = 4000, shortlist: int = 8):
        # shortlist: marks verified geometrically per query. Retrieval only has to rank the right
        # mark in the top `shortlist`; with 3 it missed ~1 in 6 true marks on the eval set.
        self.shortlist = shortlist
        self.sift = cv2.SIFT_create(nfeatures=nfeatures)
        self.refs: list[_Ref] = []
        self.owner = np.zeros(0, np.int32)
        self.desc = np.zeros((0, 128), np.float32)
        self.flann = None
        self.max_side = max_side

    def _sift(self, g: np.ndarray):
        kps, desc = self.sift.detectAndCompute(g, None)
        if desc is None:
            return kps, None
        # RootSIFT (Arandjelovic & Zisserman 2012): L1-normalise then sqrt, so Euclidean
        # distance behaves like the Hellinger kernel. Cheap and noticeably better matching.
        desc = desc / (np.abs(desc).sum(axis=1, keepdims=True) + 1e-7)
        return kps, np.sqrt(desc).astype(np.float32)

    def _gray(self, bgr: np.ndarray) -> tuple[np.ndarray, float]:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        h, w = g.shape
        s = min(1.0, self.max_side / max(h, w))
        if s < 1:
            g = cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        return g, s

    def add(self, name: str, bgr: np.ndarray, augment: bool = True) -> int:
        """Register a mark. With augment=True it is also stored as printed on a few different
        surfaces (mid-gray, dark, textured), because a mark drawn on white paper has very
        different gradients at its edges once it is ink on a gray hoodie."""
        views = [bgr] + (augmented_views(bgr) if augment else [])
        n = 0
        for v in views:
            n += self._add_view(name, v)
        return n

    def _add_view(self, name: str, bgr: np.ndarray) -> int:
        g, _ = self._gray(bgr)
        # pad so keypoints near the logo border survive
        g = cv2.copyMakeBorder(g, 32, 32, 32, 32, cv2.BORDER_CONSTANT, value=255)
        kps, desc = self._sift(g)
        if desc is None or len(kps) < 10:
            raise ValueError(f"logo {name!r} has too little texture ({0 if desc is None else len(kps)} keypoints)")
        start = len(self.desc)
        self.refs.append(_Ref(name, np.float32([k.pt for k in kps]), g.shape, (start, start + len(kps))))
        self.desc = np.vstack([self.desc, desc.astype(np.float32)])
        self.owner = np.concatenate([self.owner, np.full(len(kps), len(self.refs) - 1, np.int32)])
        self.flann = None
        return len(kps)

    def _index(self):
        if self.flann is None:
            self.flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 8}, {"checks": 256})
            self.flann.add([self.desc])
            self.flann.train()
        return self.flann

    def detect(self, bgr: np.ndarray, ratio: float = 0.8) -> list[LogoHit]:
        if not self.refs:
            return []
        g, s = self._gray(bgr)
        kps, desc = self._sift(g)
        if desc is None or len(kps) < 8:
            return []
        desc = desc.astype(np.float32)
        # Stage 1, retrieval: every query descriptor votes for the mark that owns its nearest
        # reference descriptor. Cheap, scales to thousands of marks, and only has to put the
        # right mark in the shortlist.
        nn = self._index().knnMatch(desc, k=1)
        tally: dict[str, int] = defaultdict(int)
        for m in nn:
            if m:
                tally[self.refs[self.owner[m[0].trainIdx]].name] += 1
        total = max(1, sum(tally.values()))
        expected = {r.name: 0 for r in self.refs}
        for r in self.refs:
            expected[r.name] += (r.desc_range[1] - r.desc_range[0])
        n_desc = max(1, len(self.desc))
        # rank by votes relative to the share of the index a mark occupies
        ranked = sorted(tally, key=lambda k: -tally[k] / (total * expected[k] / n_desc))[: self.shortlist]
        # Stage 2, verification: match each shortlisted mark's views *into* the query
        # (reference -> query, Lowe ratio against the query's own 2nd-best), then RANSAC.
        bf = cv2.BFMatcher(cv2.NORM_L2)
        hits = []
        for name in ranked:
            best = None
            for ref in (r for r in self.refs if r.name == name):
                rdesc = self.desc[ref.desc_range[0]:ref.desc_range[1]]
                pairs = [(a.queryIdx, a.trainIdx) for a, b in (x for x in bf.knnMatch(rdesc, desc, k=2) if len(x) == 2)
                         if a.distance < ratio * b.distance]
                if len(pairs) < INLIERS_MIN:
                    continue
                src = np.float32([ref.kps[i] for i, _ in pairs])
                dst = np.float32([kps[j].pt for _, j in pairs])
                # A similarity transform (rotation + scale + shift, 4 DoF) rather than a full
                # homography: with 10-20 matches that often sit on one line of text, a
                # homography fit is unstable and gets rejected as degenerate even when the
                # match is real. Print on a surface is close enough to a similarity.
                A, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=6.0,
                                                      maxIters=2000, confidence=0.995)
                if A is None:
                    continue
                n = int(mask.sum())
                scale = float(np.sqrt(abs(np.linalg.det(A[:, :2]))))
                if not (0.05 < scale < 6):
                    continue
                h, w = ref.shape
                corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
                quad = (corners @ A[:, :2].T + A[:, 2]) / s
                if best is None or n > best[0]:
                    best = (n, quad)
            if best and inliers_to_score(best[0]) > 0:
                hits.append(LogoHit(name, best[0], inliers_to_score(best[0]), best[1].astype(int).tolist()))
        hits.sort(key=lambda h: -h.score)
        return hits
