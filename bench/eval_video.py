"""How many frames does a video post-check need to look at?

Takes real Magic Hour clips (earlier generations in this project), splices a short segment
(0.25 s / 0.5 s / 1 s) showing a protected person into a random point of each, and checks
whether each sampling strategy catches it. Clean copies of the clips measure false alarms.

Strategies: every frame, uniform 1 fps, uniform 2 fps, and modpipe's sampler (scene-change
triggered + 1 s max gap + dHash dedupe).
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modpipe.image.faces import FaceEmbedder, IdentityGallery, similarity_to_score  # noqa: E402
from modpipe.video import sample_frames  # noqa: E402

CLIP_GLOBS = ["bio-explainer/out/*/shot*.mp4", "book-scenes/public/books/time-machine/clips/*.mp4",
              "strava-wrapped/v2/clips/*.mp4", "outfit-planner/looks/clips/*.mp4"]
PF = Path("/mnt/project-files")
GEN = ROOT / "data" / "gen" / "faces"


def plant(src: Path, face_img: np.ndarray, start_s: float, dur_s: float, out: Path) -> None:
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    # letterbox the planted still into the clip's frame and slowly push in, like a real shot
    fh, fw = face_img.shape[:2]
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = i / fps
        if start_s <= t < start_s + dur_s:
            z = 1.0 + 0.08 * (t - start_s) / dur_s
            s = min(w / fw, h / fh) * z
            im = cv2.resize(face_img, (int(fw * s), int(fh * s)))
            canvas = np.zeros((h, w, 3), np.uint8)
            y0, x0 = (h - im.shape[0]) // 2, (w - im.shape[1]) // 2
            ys, xs = max(0, y0), max(0, x0)
            iy, ix = max(0, -y0), max(0, -x0)
            hh, ww = min(h - ys, im.shape[0] - iy), min(w - xs, im.shape[1] - ix)
            canvas[ys:ys + hh, xs:xs + ww] = im[iy:iy + hh, ix:ix + ww]
            fr = canvas
        wr.write(fr)
        i += 1
    cap.release()
    wr.release()


def detect(fe, gal, path, strategy) -> tuple[bool, int, float]:
    t0 = time.perf_counter()
    if strategy == "all":
        frames, _ = sample_frames(str(path), uniform=1000, max_frames=10_000)
    elif strategy == "1fps":
        frames, _ = sample_frames(str(path), uniform=1, max_frames=10_000)
    elif strategy == "2fps":
        frames, _ = sample_frames(str(path), uniform=2, max_frames=10_000)
    else:
        frames, _ = sample_frames(str(path))
    hit = False
    for f in frames:
        for face in fe.faces(f.image):
            _, d = gal.match(face.vec)
            if similarity_to_score(d) >= 0.5:
                hit = True
                break
        if hit:
            break
    return hit, len(frames), (time.perf_counter() - t0) * 1000


def main():
    rng = random.Random(5)
    clips = sorted({p for g in CLIP_GLOBS for p in PF.glob(g)})
    fe = FaceEmbedder()
    gal = IdentityGallery()
    for p in sorted((GEN / "enroll").glob("*.png")):
        gal.add(p.name[:12], fe.faces(cv2.imread(str(p)))[0].vec)
    swaps = sorted((GEN / "swaps2").glob("swap2_p*.png"))
    strategies = ["all", "1fps", "2fps", "modpipe"]
    res = {s: {"pos": {}, "neg": [0, 0], "frames": [], "ms": []} for s in strategies}
    with tempfile.TemporaryDirectory() as td:
        for ci, clip in enumerate(clips):
            cap = cv2.VideoCapture(str(clip))
            dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / (cap.get(cv2.CAP_PROP_FPS) or 24)
            cap.release()
            for s in strategies:
                hit, n, ms = detect(fe, gal, clip, s)
                res[s]["neg"][0] += hit
                res[s]["neg"][1] += 1
                res[s]["frames"].append(n)
                res[s]["ms"].append(ms)
            for seg in (0.25, 0.5, 1.0):
                sw = cv2.imread(str(rng.choice(swaps)))
                start = rng.uniform(0.2, max(0.3, dur - seg - 0.2))
                out = Path(td) / f"c{ci}_{seg}.mp4"
                plant(clip, sw, start, seg, out)
                # positive only if the planted still is detectable at all (checked on the still)
                if not any(similarity_to_score(gal.match(f.vec)[1]) >= 0.5 for f in fe.faces(sw)):
                    continue
                for s in strategies:
                    hit, n, ms = detect(fe, gal, out, s)
                    d = res[s]["pos"].setdefault(str(seg), [0, 0])
                    d[0] += hit
                    d[1] += 1
            print(clip.name, {s: res[s]["pos"] for s in strategies})
    summary = {}
    for s in strategies:
        r = res[s]
        summary[s] = {"recall_by_segment_s": {k: round(v[0] / v[1], 3) for k, v in r["pos"].items()},
                      "n_by_segment": {k: v[1] for k, v in r["pos"].items()},
                      "false_alarms_on_clean": f"{r['neg'][0]}/{r['neg'][1]}",
                      "mean_frames_checked": round(float(np.mean(r["frames"])), 1),
                      "mean_ms_per_clip": round(float(np.mean(r["ms"])), 1)}
    json.dump({"clips": len(clips), "strategies": summary}, open(ROOT / "results" / "video.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
