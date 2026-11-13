"""Evaluate the image detectors on the Magic Hour-generated set (bench/make_images.py,
bench/make_logo_set.py). Thresholds are picked on the dev half and reported on the test half.

faces: 10 protected identities enrolled from one photo each. Positives are Magic Hour face swaps
       of those people into new scenes. Negatives are the untouched scenes (other synthetic
       people) and swaps of 6 non-enrolled "impostor" faces. Split by identity / scene parity.
logos: 8 registered marks printed onto 30 product/street scenes; negatives are clean scenes and
       scenes carrying one of 4 unregistered look-alike marks. Split by scene parity.
nsfw:  30 hard negatives (swimwear, gym, statues, bodybuilding...). False-positive rate only:
       this repo deliberately contains no explicit imagery, so NSFW recall is not measured here.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modpipe.image.faces import FaceEmbedder  # noqa: E402
from modpipe.image.logos import LogoIndex  # noqa: E402
from modpipe.image.nsfw import NSFWDetector  # noqa: E402

GEN = ROOT / "data" / "gen"


def pr(scores_pos, scores_neg, thr):
    tp = sum(s >= thr for s in scores_pos)
    fp = sum(s >= thr for s in scores_neg)
    fn = len(scores_pos) - tp
    return {"threshold": round(float(thr), 4), "precision": round(tp / (tp + fp), 4) if tp + fp else 1.0,
            "recall": round(tp / max(1, tp + fn), 4), "fpr": round(fp / max(1, len(scores_neg)), 4),
            "tp": tp, "fp": fp, "fn": fn, "n_pos": len(scores_pos), "n_neg": len(scores_neg)}


def pick_threshold(pos, neg, max_fpr=0.0, higher_is_positive=True):
    """Lowest threshold (most recall) whose dev false-positive rate is <= max_fpr."""
    cands = sorted(set(pos) | set(neg) | {1e9})
    for t in cands:
        if sum(s >= t for s in neg) / max(1, len(neg)) <= max_fpr:
            return t
    return 1e9


def timed(fn, *a):
    t0 = time.perf_counter()
    r = fn(*a)
    return r, (time.perf_counter() - t0) * 1000


def eval_faces():
    fe = FaceEmbedder()
    enroll = {}
    for p in sorted((GEN / "faces" / "enroll").glob("*.png")):
        f = fe.faces(cv2.imread(str(p)))
        assert len(f) == 1, p
        enroll[p.name[:12]] = f[0].vec
    ids = sorted(enroll)
    M = np.stack([enroll[i] for i in ids])
    times = []

    def best(path):
        faces, ms = timed(fe.faces, cv2.imread(str(path)))
        times.append(ms)
        if not faces:
            return None, 9.9, 0
        D = np.linalg.norm(np.stack([f.vec for f in faces])[:, None] - M[None], axis=2)  # (faces, ids)
        i, j = np.unravel_index(D.argmin(), D.shape)
        return ids[j], float(D[i, j]), len(faces), D.min(axis=0)

    pos, neg = [], []
    for p in sorted((GEN / "faces" / "swaps2").glob("swap2_p*.png")):
        k = int(re.search(r"_p(\d+)_", p.name).group(1))
        r = best(p)
        pos.append({"path": p.name, "identity": f"protected_{k:02d}", "pred": r[0], "d": r[1],
                    "faces": r[2], "split": "dev" if k % 2 == 0 else "test",
                    "per_id": (r[3].tolist() if len(r) > 3 else None)})
    # Negatives: people described explicitly (varied age/ethnicity/gender, none overlapping a
    # protected description) plus Magic Hour swaps of non-enrolled impostor faces.
    neg_paths = sorted((GEN / "faces" / "negatives").glob("*.png")) + \
        sorted((GEN / "faces" / "swaps2").glob("swap2_i*.png")) + sorted((GEN / "faces" / "swaps").glob("swap_i*.png"))
    # Stress set, reported separately: "a person ..." prompts, which z-image-turbo renders as
    # nearly the same young East Asian man every time.
    default_face = sorted((GEN / "faces" / "scenes2").glob("*.png"))
    stress = []
    for p in default_face:
        r = best(p)
        stress.append({"path": p.name, "pred": r[0], "d": r[1], "faces": r[2],
                       "per_id": (r[3].tolist() if len(r) > 3 else None)})
    for n, p in enumerate(neg_paths):
        r = best(p)
        neg.append({"path": p.name, "pred": r[0], "d": r[1], "faces": r[2], "split": "dev" if n % 2 == 0 else "test",
                    "per_id": (r[3].tolist() if len(r) > 3 else None)})

    def score(d):
        return -d   # higher = more likely the protected person

    def correct(r):
        return r["pred"] == r["identity"]

    out = {"n_pos": len(pos), "n_neg": len(neg), "no_face_in_positive": sum(r["faces"] == 0 for r in pos),
           "no_face_in_negative": sum(r["faces"] == 0 for r in neg)}
    dev_pos = [score(r["d"]) if correct(r) else -9.9 for r in pos if r["split"] == "dev"]
    dev_neg = [score(r["d"]) for r in neg if r["split"] == "dev"]
    test_pos = [score(r["d"]) if correct(r) else -9.9 for r in pos if r["split"] == "test"]
    test_neg = [score(r["d"]) for r in neg if r["split"] == "test"]
    # global threshold: dlib's default 0.6 vs one tuned on dev for <= 2% false matches
    out["global_default_0.6"] = {"dev": pr(dev_pos, dev_neg, -0.6), "test": pr(test_pos, test_neg, -0.6)}
    t = pick_threshold(dev_pos, dev_neg, 0.02)
    out["test_curve"] = [pr(test_pos, test_neg, -d) for d in (0.36, 0.40, 0.45, 0.50, 0.55, 0.60)]
    allp = [x for x in dev_pos + test_pos]
    alln = [x for x in dev_neg + test_neg]
    out["roc_auc"] = round(float(np.mean([(p > n) + 0.5 * (p == n) for p in allp for n in alln])), 4)
    out["global_tuned"] = {"distance": -t, "dev": pr(dev_pos, dev_neg, t), "test": pr(test_pos, test_neg, t)}

    # Per-identity thresholds from a cohort of dev negatives: some people (in our data, the
    # East Asian identities) sit much closer to random faces in dlib's embedding space, so one
    # global cutoff either misses everyone else or false-matches them constantly.
    cohort = np.array([r["per_id"] for r in neg if r["split"] == "dev" and r["per_id"] is not None])  # (n, ids)
    per_id_thr = {}
    for j, ident in enumerate(ids):
        q = np.quantile(cohort[:, j], 0.02)
        per_id_thr[ident] = float(min(-t, q - 0.02))

    def cohort_hit(r):
        if r["per_id"] is None:
            return None
        margins = [per_id_thr[i] - r["per_id"][j] for j, i in enumerate(ids)]
        j = int(np.argmax(margins))
        return ids[j], margins[j]

    def cohort_scores(rows, positive):
        s = []
        for r in rows:
            h = cohort_hit(r)
            if h is None:
                s.append(-9.9)
            elif positive and h[0] != r["identity"]:
                s.append(-9.9)
            else:
                s.append(h[1])
        return s
    cp_dev = cohort_scores([r for r in pos if r["split"] == "dev"], True)
    cn_dev = cohort_scores([r for r in neg if r["split"] == "dev"], False)
    cp_test = cohort_scores([r for r in pos if r["split"] == "test"], True)
    cn_test = cohort_scores([r for r in neg if r["split"] == "test"], False)
    out["per_identity_cohort"] = {"thresholds": {k: round(v, 4) for k, v in per_id_thr.items()},
                                  "dev": pr(cp_dev, cn_dev, 0.0), "test": pr(cp_test, cn_test, 0.0)}
    st = [r for r in stress if r["faces"]]
    out["default_face_stress_set"] = {
        "n": len(stress), "median_pairwise_note": "median dlib distance between these 'different' people is ~0.47",
        "false_match_rate_global_0.6": round(sum(r["d"] < 0.6 for r in st) / max(1, len(st)), 4),
        "false_match_rate_global_tuned": round(sum(r["d"] < -t for r in st) / max(1, len(st)), 4),
        "false_match_rate_cohort": round(sum((cohort_hit(r) or (None, -1))[1] >= 0 for r in st) / max(1, len(st)), 4),
        "closest_identity_counts": {i: sum(r["pred"] == i for r in st) for i in ids if any(r["pred"] == i for r in st)}}
    out["latency_ms"] = {"p50": round(float(np.median(times)), 1), "p99": round(float(np.percentile(times, 99)), 1)}
    out["hardest_negatives"] = sorted(({"path": r["path"], "closest": r["pred"], "d": round(r["d"], 3)} for r in neg if r["faces"]),
                                      key=lambda x: x["d"])[:8]
    return out


def eval_logos():
    lab = json.load(open(GEN / "logos" / "labels.json"))
    idx = LogoIndex()
    for k in lab["registered"]:
        idx.add(k, cv2.imread(str(GEN / "logos" / "marks" / f"{k}.png")))
    scenes = sorted({r["scene"] for r in lab["images"]})
    split = {s: ("dev" if i % 2 == 0 else "test") for i, s in enumerate(scenes)}
    rows, times = [], []
    for r in lab["images"]:
        hits, ms = timed(idx.detect, cv2.imread(str(ROOT / r["path"])))
        times.append(ms)
        right = [h for h in hits if h.name == r["logo"]]
        wrong = [h for h in hits if h.name != r["logo"]]
        rows.append({**r, "split": split[r["scene"]],
                     "inliers_right": right[0].inliers if right else 0,
                     "inliers_wrong": wrong[0].inliers if wrong else 0})

    def sc(rows, positive):
        # a positive counts only if the *right* mark is found; any hit on a negative is a FP
        return [r["inliers_right"] if positive else max(r["inliers_right"], r["inliers_wrong"]) for r in rows]
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    dp, dn = sc([r for r in dev if r["registered"]], True), sc([r for r in dev if not r["registered"]], False)
    tp_, tn = sc([r for r in test if r["registered"]], True), sc([r for r in test if not r["registered"]], False)
    t = max(10, pick_threshold(dp, dn, 0.0))
    curve = [pr(tp_, tn, k) for k in (8, 10, 12, 15, 20, 25, 30)]
    return {"threshold_inliers": t, "dev": pr(dp, dn, t), "test": pr(tp_, tn, t), "test_curve": curve,
            "latency_ms": {"p50": round(float(np.median(times)), 1), "p99": round(float(np.percentile(times, 99)), 1)},
            "unregistered_lookalike_false_hits": sum(1 for r in rows if r["logo"] and not r["registered"] and r["inliers_wrong"] >= t)}


def eval_nsfw():
    det = NSFWDetector()
    paths = sorted((GEN / "nsfw_neg").glob("*.png"))
    scores, times = [], []
    for p in paths:
        r, ms = timed(det.detect, cv2.imread(str(p)))
        scores.append(r["score"])
        times.append(ms)
    imgs = [cv2.imread(str(p)) for p in paths[:16]]
    _, batch_ms = timed(det.detect_batch, imgs)
    return {"n_hard_negatives": len(paths),
            "fpr_at_review_0.45": round(sum(s >= 0.45 for s in scores) / len(scores), 4),
            "fpr_at_block_0.65": round(sum(s >= 0.65 for s in scores) / len(scores), 4),
            "max_score": round(max(scores), 4),
            "flagged": [p.name for p, s in zip(paths, scores) if s >= 0.45],
            "latency_ms": {"p50_single": round(float(np.median(times)), 1),
                           "per_image_in_batch_of_16": round(batch_ms / 16, 1)}}


def main():
    res = {"faces": eval_faces(), "logos": eval_logos(), "nsfw": eval_nsfw()}
    json.dump(res, open(ROOT / "results" / "images.json", "w"), indent=1)
    f = res["faces"]
    print("faces", {k: f[k]["test"] for k in ("global_default_0.6", "global_tuned", "per_identity_cohort")})
    print("faces no-face", f["no_face_in_positive"], "/", f["n_pos"], "latency", f["latency_ms"])
    print("logos", res["logos"]["test"], res["logos"]["latency_ms"])
    print("nsfw", {k: v for k, v in res["nsfw"].items() if k != "flagged"})


if __name__ == "__main__":
    main()
