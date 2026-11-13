"""Build the logo eval set: 8 registered marks + 4 unregistered look-alike marks (all invented),
composited onto Magic Hour product/street scenes with perspective, lighting, blur and JPEG
damage. Positive = a registered mark is visible. Negatives = clean scenes and scenes carrying an
unregistered mark (same visual style, different design), which is what makes precision mean
something.

Writes data/gen/logos/{marks,images}/ and data/gen/logos/labels.json.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "gen" / "logos"
FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]
REGISTERED = [("volta", "VOLTA", "bolt"), ("kestrel", "Kestrel Coffee", "circle"),
              ("nordwind", "NORDWIND", "triangle"), ("pipfern", "Pip & Fern", "leaf"),
              ("halcyon", "HALCYON", "waves"), ("brightloop", "brightloop", "ring"),
              ("tamarack", "TAMARACK", "mountain"), ("orbitex", "Orbitex", "orbit")]
UNREGISTERED = [("vexa", "VEXA", "bolt"), ("kettle", "Kettle Tea Co", "circle"),
                ("nordic", "NORDIQ", "triangle"), ("fernhill", "Fernhill", "leaf")]


def draw_mark(text: str, shape: str, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    W, H = 640, 300
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    col = tuple(rng.randint(0, 150) for _ in range(3))
    acc = tuple(rng.randint(80, 255) for _ in range(3))
    cx, cy, r = 140, 150, 95
    if shape == "bolt":
        d.polygon([(cx - 20, 60), (cx + 50, 60), (cx + 5, 140), (cx + 55, 140), (cx - 40, 245), (cx - 5, 160), (cx - 50, 160)], fill=col)
    elif shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=14)
        d.ellipse([cx - 40, cy - 40, cx + 40, cy + 40], fill=acc)
        d.arc([cx - 70, cy - 70, cx + 70, cy + 70], 200, 340, fill=col, width=10)
    elif shape == "triangle":
        d.polygon([(cx, 50), (cx + 95, 240), (cx - 95, 240)], fill=col)
        d.polygon([(cx, 120), (cx + 45, 210), (cx - 45, 210)], fill=acc)
    elif shape == "leaf":
        d.ellipse([cx - 90, cy - 45, cx + 90, cy + 45], fill=col)
        d.line([(cx - 90, cy), (cx + 90, cy)], fill="white", width=6)
        for k in range(-60, 70, 30):
            d.line([(cx + k, cy), (cx + k + 25, cy - 30)], fill="white", width=4)
    elif shape == "waves":
        for k in range(4):
            d.arc([cx - 90, cy - 70 + k * 30, cx + 90, cy + 10 + k * 30], 200, 340, fill=col if k % 2 else acc, width=12)
    elif shape == "ring":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=26)
        d.ellipse([cx - 20, cy - r - 10, cx + 20, cy - r + 30], fill=acc)
    elif shape == "mountain":
        d.polygon([(cx - 100, 240), (cx - 30, 80), (cx + 10, 160), (cx + 50, 100), (cx + 110, 240)], fill=col)
        d.polygon([(cx - 30, 80), (cx - 50, 125), (cx - 10, 125)], fill="white")
    elif shape == "orbit":
        d.ellipse([cx - 40, cy - 40, cx + 40, cy + 40], fill=col)
        d.ellipse([cx - 100, cy - 35, cx + 100, cy + 35], outline=acc, width=8)
    font = ImageFont.truetype(rng.choice(FONTS), 64 if len(text) < 9 else 46)
    d.text((255, 150), text, fill=col, font=font, anchor="lm")
    d.text((258, 210), "EST. " + str(rng.randint(1950, 2020)), fill=acc, font=ImageFont.truetype(FONTS[1], 24), anchor="lm")
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def composite(scene: np.ndarray, mark: np.ndarray, rng: random.Random) -> np.ndarray:
    H, W = scene.shape[:2]
    mh, mw = mark.shape[:2]
    tw = int(W * rng.uniform(0.18, 0.45))
    th = int(tw * mh / mw)
    # products are roughly centred in these scenes; print the mark near the middle
    x0 = int(np.clip(W / 2 - tw / 2 + rng.gauss(0, W * 0.08), 10, W - tw - 10))
    y0 = int(np.clip(H / 2 - th / 2 + rng.gauss(0, H * 0.08), 10, H - th - 10))
    j = lambda: rng.uniform(-0.12, 0.12)
    dst = np.float32([[x0 + j() * tw, y0 + j() * th], [x0 + tw + j() * tw, y0 + j() * th],
                      [x0 + tw + j() * tw, y0 + th + j() * th], [x0 + j() * tw, y0 + th + j() * th]])
    ang = math.radians(rng.uniform(-15, 15))
    c = dst.mean(0)
    R = np.float32([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    dst = (dst - c) @ R.T + c
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [mw, 0], [mw, mh], [0, mh]]), dst)
    warped = cv2.warpPerspective(mark, M, (W, H))
    # Printed, not stickered: the mark's white paper is transparent, only the ink lands on the
    # surface, and it takes on the surface's own shading.
    ink_amount = 1.0 - cv2.cvtColor(mark, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    ink_amount = np.clip(ink_amount * 2.0, 0, 1)
    mask = cv2.warpPerspective(ink_amount, M, (W, H))
    alpha = (mask * rng.uniform(0.8, 1.0))[..., None]
    shade = cv2.GaussianBlur(cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY), (0, 0), 15)[..., None] / 255.0
    ink = np.clip(warped * (0.45 + 0.7 * shade), 0, 255)
    out = (scene * (1 - alpha) + ink * alpha).astype(np.uint8)
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.6, 1.8))
    out = cv2.convertScaleAbs(out, alpha=rng.uniform(0.8, 1.15), beta=rng.uniform(-20, 20))
    q = rng.randint(35, 90)
    out = cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    return out


def main():
    rng = random.Random(3)
    (OUT / "marks").mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    marks = {}
    for i, (key, text, shape) in enumerate(REGISTERED + UNREGISTERED):
        m = draw_mark(text, shape, i)
        cv2.imwrite(str(OUT / "marks" / f"{key}.png"), m)
        marks[key] = m
    scenes = sorted((OUT / "scenes").glob("*.png"))
    assert scenes, "run bench/make_images.py logos first"
    labels = []
    reg = [k for k, _, _ in REGISTERED]
    unreg = [k for k, _, _ in UNREGISTERED]
    n = 0
    for sp in scenes:
        scene = cv2.imread(str(sp))
        scene = cv2.resize(scene, (1024, 1024))
        # clean copy (negative)
        p = OUT / "images" / f"{n:03d}_clean.jpg"
        cv2.imwrite(str(p), scene, [cv2.IMWRITE_JPEG_QUALITY, 85]); n += 1
        labels.append({"path": str(p.relative_to(ROOT)), "logo": None, "registered": False, "scene": sp.name})
        for key in rng.sample(reg, 2):
            p = OUT / "images" / f"{n:03d}_{key}.jpg"
            cv2.imwrite(str(p), composite(scene, marks[key], rng)); n += 1
            labels.append({"path": str(p.relative_to(ROOT)), "logo": key, "registered": True, "scene": sp.name})
        key = rng.choice(unreg)
        p = OUT / "images" / f"{n:03d}_{key}.jpg"
        cv2.imwrite(str(p), composite(scene, marks[key], rng)); n += 1
        labels.append({"path": str(p.relative_to(ROOT)), "logo": key, "registered": False, "scene": sp.name})
    json.dump({"registered": reg, "unregistered": unreg, "images": labels}, open(OUT / "labels.json", "w"), indent=1)
    print(len(labels), "images")


if __name__ == "__main__":
    main()
