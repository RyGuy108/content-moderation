"""results/*.json -> results/summary.png"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1] / "results"
t, im, v = (json.load(open(R / f)) for f in ("text.json", "images.json", "video.json"))
BLUE, GRAY, INK = "#2f5bd3", "#b9b9b3", "#1b1b1a"
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
for a in ax:
    a.spines[["top", "right"]].set_visible(False)

# 1. prompt pre-check vs keyword blocklist
sets = [("natural_final", "hand-written\n(never trained on)"), ("adversarial", "obfuscated\n(leet, homoglyph...)"), ("test", "held-out\ntemplates")]
x = range(len(sets))
for off, who, col in ((-0.2, "baseline", GRAY), (0.2, "ours", BLUE)):
    vals = [t[s]["unsafe" if False else s][who]["unsafe_vs_safe"]["recall"] if False else t[s][who]["unsafe_vs_safe"]["recall"] for s, _ in sets]
    b = ax[0].bar([i + off for i in x], vals, 0.4, color=col, label="keyword blocklist" if who == "baseline" else "modpipe")
    ax[0].bar_label(b, fmt="%.2f", fontsize=9)
ax[0].set_xticks(list(x), [l for _, l in sets], fontsize=9)
ax[0].set_ylim(0, 1.2)
ax[0].set_title("Prompt pre-check: recall on unsafe prompts", fontsize=11, loc="left")
ax[0].legend(frameon=False, fontsize=9, loc="upper left", ncols=2)
fp = t["natural_final"]
ax[0].text(0, -0.32, f"false-positive rate on safe prompts: blocklist {fp['baseline']['unsafe_vs_safe']['false_positive_rate']:.0%}, "
           f"modpipe {fp['ours']['unsafe_vs_safe']['false_positive_rate']:.0%}  ·  {t['latency']['p50_ms']} ms p50",
           transform=ax[0].transAxes, fontsize=8.5, color="#555")

# 2. video sampling
s = v["strategies"]
names = {"1fps": "uniform 1 fps", "2fps": "uniform 2 fps", "modpipe": "modpipe sampler", "all": "every frame"}
for k in ("1fps", "2fps", "modpipe", "all"):
    ax[1].scatter(s[k]["mean_frames_checked"], s[k]["recall_by_segment_s"]["0.25"], s=80, color=BLUE if k == "modpipe" else GRAY, zorder=3)
    ax[1].annotate(names[k], (s[k]["mean_frames_checked"], s[k]["recall_by_segment_s"]["0.25"]), xytext=(8, -4),
                   textcoords="offset points", fontsize=9)
ax[1].set_xscale("log")
ax[1].set_xlabel("frames checked per 5 s clip (log)")
ax[1].set_ylabel("caught a 0.25 s protected-face insert")
ax[1].set_ylim(0, 1.1)
ax[1].set_title("Video post-check: frames vs. recall", fontsize=11, loc="left")

# 3. image detectors operating curves (test split)
fc = im["faces"]["test_curve"]
ax[2].plot([c["fpr"] for c in fc], [c["recall"] for c in fc], "o-", color=BLUE, label=f"protected face (dlib, AUC {im['faces']['roc_auc']:.2f})")
lc = im["logos"]["test_curve"]
ax[2].plot([c["fpr"] for c in lc], [c["recall"] for c in lc], "s-", color="#a86400", label="registered logo (SIFT)")
ax[2].set_xlabel("false-positive rate")
ax[2].set_ylabel("recall")
ax[2].set_xlim(-0.02, 0.6)
ax[2].set_ylim(0, 1.05)
ax[2].legend(frameon=False, fontsize=9, loc="lower right")
ax[2].set_title("Image detectors on Magic Hour test images", fontsize=11, loc="left")
fig.tight_layout()
fig.savefig(R / "summary.png", dpi=160, bbox_inches="tight")
print("wrote", R / "summary.png")
