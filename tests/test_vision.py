import cv2
import numpy as np

from modpipe.image.logos import LogoIndex
from modpipe.video import sample_frames


def _mark(text, seed):
    rng = np.random.default_rng(seed)
    im = np.full((240, 520, 3), 255, np.uint8)
    for _ in range(12):
        c = tuple(int(x) for x in rng.integers(0, 160, 3))
        cv2.circle(im, tuple(int(x) for x in rng.integers(20, 220, 2)), int(rng.integers(8, 40)), c, -1)
    cv2.putText(im, text, (200, 140), cv2.FONT_HERSHEY_DUPLEX, 2.0, (20, 20, 120), 4)
    return im


def test_logo_found_under_scale_and_rotation():
    idx = LogoIndex()
    idx.add("acme", _mark("ACME", 1))
    idx.add("zeta", _mark("ZETA", 2))
    scene = np.full((900, 900, 3), 200, np.uint8)
    m = cv2.resize(_mark("ACME", 1), None, fx=0.6, fy=0.6)
    M = cv2.getRotationMatrix2D((m.shape[1] / 2, m.shape[0] / 2), 12, 1.0)
    m = cv2.warpAffine(m, M, (m.shape[1], m.shape[0]), borderValue=(200, 200, 200))
    scene[300:300 + m.shape[0], 200:200 + m.shape[1]] = m
    hits = idx.detect(scene)
    assert hits and hits[0].name == "acme" and hits[0].inliers >= 15
    assert not idx.detect(np.full((600, 600, 3), 200, np.uint8))


def test_sampler_dedupes_static_video(tmp_path):
    p = str(tmp_path / "static.mp4")
    w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 24, (320, 240))
    for i in range(24 * 6):
        f = np.full((240, 320, 3), 90, np.uint8)
        if 72 <= i < 78:                    # a 0.25 s flash of something different
            f[:] = (10, 200, 30)
        w.write(f)
    w.release()
    frames, info = sample_frames(p)
    ts = [round(f.t, 2) for f in frames]
    assert any(3.0 <= t < 3.25 for t in ts)          # caught the 0.25 s event
    assert len(frames) <= 4                           # and skipped the identical frames
