"""Generate the labeled image eval set with the real Magic Hour API.

Everything is synthetic: no real person's face or real company's logo is used.

  faces/   10 "protected" identities (think: talent who opted in to likeness protection) and
           6 impostors, face-swapped (Magic Hour face_swap_photo) into 40 everyday scenes.
           Positive = a protected face appears. Negatives = the original scenes and impostor swaps.
  logos/   product/street scenes from Magic Hour; bench/make_logo_set.py composites 8 invented
           marks into them. A few are re-rendered by ai_image_editor so the mark is part of the
           image, not pasted on top.
  nsfw_neg/ hard negatives for the NSFW detector: swimwear, gym, beach, classical statues.

Every project id is appended to data/gen/ledger.jsonl before downloading, so a crash never
loses a paid render. Re-running skips finished items.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from magic_hour import Client

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "data" / "gen"
LEDGER = GEN / "ledger.jsonl"

PROTECTED = [
    "a woman in her 30s with curly dark hair and freckles",
    "a man in his 50s with a gray beard and round glasses",
    "a young east asian man with short spiky hair",
    "a black woman in her 40s with short natural hair and gold earrings",
    "a red-haired man in his 20s with a thin mustache",
    "a south asian woman in her 20s with long straight black hair",
    "an older white woman in her 70s with silver bob haircut",
    "a latino man in his 30s with a shaved head and goatee",
    "a blonde woman in her 20s with a nose ring",
    "a middle eastern man in his 40s with thick eyebrows and stubble",
]
IMPOSTORS = [
    "a woman in her 30s with wavy brown hair",
    "a man in his 50s with a white beard",
    "a young korean man with a bowl haircut",
    "a black man in his 30s with short dreadlocks",
    "a woman in her 60s with gray curly hair",
    "a man in his 20s with blond hair and blue eyes",
]
SCENES = [
    "standing at a farmers market", "laughing in a coffee shop", "walking down a city street at night with neon lights",
    "sitting on a park bench in autumn", "hiking on a mountain trail", "working on a laptop in an office",
    "at a wedding reception, holding a glass", "on a beach at sunset, wind in hair", "in a kitchen cooking",
    "on a subway train, looking out the window", "at a concert, stage lights behind", "riding a bicycle in amsterdam",
    "in a library between bookshelves", "at a basketball game in the stands", "in the rain with an umbrella",
    "in a black and white portrait, dramatic lighting", "at a gym lifting weights", "in a recording studio with headphones",
    "at a campfire at night", "in a business suit at a press podium",
]
STYLE = ", candid photo, face visible, natural light, 35mm"
# Round 1 asked for "a person <doing X>" and about half came back as backs of heads or cropped
# faces, which makes the face swap a no-op. Round 2 asks for medium shots facing the camera.
SCENES2 = [
    "medium shot of a person smiling at the camera at a street food stall",
    "waist-up photo of a person looking at the camera on a ferry deck",
    "a person facing the camera in a grocery store aisle",
    "selfie of a person at a theme park",
    "a person at a desk looking up at the camera, office background",
    "a person in a winter coat facing the camera in snowfall",
    "a person on a sofa looking at the camera, living room",
    "a person at a bus stop facing the camera, overcast",
    "a person in a hard hat facing the camera at a construction site",
    "a person holding a coffee and smiling at the camera in a cafe",
    "a person in a graduation gown facing the camera",
    "a person in a chef jacket facing the camera in a restaurant kitchen",
    "head and shoulders of a person in a crowd at a stadium, looking at camera",
    "a person facing the camera in a dim bar, colored lights",
    "a person facing the camera on a city rooftop at golden hour",
    "a person in sunglasses pushed up on their head, facing the camera at the beach",
    "a person in a lab coat facing the camera in a laboratory",
    "a person facing the camera in a car, driver seat",
    "a person with a backpack facing the camera at an airport",
    "a person in a hoodie facing the camera on a skate ramp",
]
# Round 2 showed z-image-turbo renders "a person" as nearly the same young East Asian man every
# time (median dlib distance between those "different" people: 0.47, under dlib's 0.6
# same-person cutoff). Negatives therefore describe who is in the picture, and none of the
# descriptions overlap a protected identity.
NEG_PEOPLE = [
    "a heavyset white man in his 60s with a bald head", "a nigerian woman in her 30s with braids",
    "a teenage-looking adult latina woman with bangs, 20s", "an indian man in his 70s with a white mustache",
    "a japanese woman in her 50s with short gray hair", "a scandinavian man in his 30s with a long blond beard",
    "a filipino man in his 40s with a buzz cut", "an irish woman in her 40s with red curly hair and glasses",
    "a somali man in his 20s, tall and slim", "a vietnamese grandmother in her 80s",
    "a brazilian man in his 30s with long wavy hair", "a korean woman in her 30s with a bob and bangs",
    "a turkish man in his 50s with a thick black mustache", "a polish woman in her 20s with platinum hair",
    "an ethiopian woman in her 50s with a headscarf", "a mexican man in his 60s with a white beard",
    "a chinese man in his 50s with glasses and receding hairline", "an australian woman in her 30s with freckles and a sunhat",
    "a pakistani man in his 30s with a full beard", "a ghanaian man in his 40s with a shaved head and glasses",
    "a russian woman in her 60s with a bun", "a thai woman in her 20s with long hair",
    "a greek man in his 40s with curly salt-and-pepper hair", "a native american woman in her 40s with long braids",
    "a french man in his 20s with a mullet",
]
LOGO_SCENES = [
    "a plain white t-shirt on a hanger against a brick wall", "a blank coffee cup on a cafe table",
    "a delivery van parked on a street, plain side panel", "a shopping bag held by a person, plain paper bag",
    "a storefront with a blank sign above the door", "a baseball cap on a wooden table",
    "a soda can on a picnic table, plain silver can", "a billboard on a highway, empty white billboard",
    "a laptop lid on a desk, plain gray", "a cardboard shipping box on a porch", "a hoodie worn by a person walking",
    "a water bottle at a gym", "a food truck at a festival", "a sneaker on a white pedestal",
    "a phone case on a marble table",
]
NSFW_NEG = [
    "a woman in a one-piece swimsuit at a public pool", "a shirtless man running on a beach",
    "a man and a woman doing yoga in sportswear", "a marble statue of a greek god in a museum",
    "people sunbathing on a crowded beach, swimwear", "a bodybuilder posing on stage in trunks",
    "a woman in a sports bra doing crunches", "a ballet dancer in a leotard mid jump",
    "a renaissance oil painting of a mythological scene, draped figures", "a man in boxer shorts in a messy bedroom, comedy",
    "a couple kissing on a park bench, clothed", "a lingerie set flat lay on a bed, no person",
    "a surfer in a wetsuit pulling it down to the waist", "a woman in a bikini top and shorts at a music festival",
    "a mother holding her baby on a hospital bed",
]


def client() -> Client:
    return Client(token=os.environ["MAGIC_HOUR_API_KEY"])


def done() -> dict[str, dict]:
    if not LEDGER.exists():
        return {}
    out = {}
    for line in LEDGER.read_text().splitlines():
        r = json.loads(line)
        out[r["key"]] = r
    return out


def log(rec: dict) -> None:
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


def gen_image(c, key, prompt, outdir: Path, count=1):
    if key in DONE and all(Path(p).exists() for p in DONE[key].get("paths", [])) and DONE[key].get("paths"):
        return DONE[key]["paths"]
    outdir.mkdir(parents=True, exist_ok=True)
    r = c.v1.ai_image_generator.create(image_count=count, style={"prompt": prompt}, model="z-image-turbo",
                                       resolution="1k", aspect_ratio="1:1", name=key)
    log({"key": key, "id": r.id, "kind": "image", "prompt": prompt, "credits": r.credits_charged})
    return fetch(c, key, r.id, outdir, "image")


def swap(c, key, src: str, dst: str, outdir: Path):
    if key in DONE and DONE[key].get("paths") and all(Path(p).exists() for p in DONE[key]["paths"]):
        return DONE[key]["paths"]
    outdir.mkdir(parents=True, exist_ok=True)
    r = c.v1.face_swap_photo.create(assets={"face_swap_mode": "all-faces",
                                            "source_file_path": c.v1.files.upload_file(src),
                                            "target_file_path": c.v1.files.upload_file(dst)}, name=key)
    log({"key": key, "id": r.id, "kind": "swap", "src": src, "dst": dst, "credits": r.credits_charged})
    return fetch(c, key, r.id, outdir, "image")


def fetch(c, key, pid, outdir: Path, kind):
    import httpx
    for _ in range(120):
        p = c.v1.image_projects.get(id=pid)
        if p.status == "complete":
            paths = []
            for i, d in enumerate(p.downloads):
                path = outdir / f"{key}_{i}.png"
                path.write_bytes(httpx.get(d.url, timeout=60).content)
                paths.append(str(path.relative_to(ROOT)))
            log({"key": key, "id": pid, "paths": paths, "status": "complete"})
            return paths
        if p.status in ("error", "canceled"):
            log({"key": key, "id": pid, "status": p.status, "error": str(getattr(p, "error", ""))})
            return []
        time.sleep(3)
    return []


DONE: dict = {}


def main(stage: str):
    global DONE
    GEN.mkdir(parents=True, exist_ok=True)
    DONE = done()
    c = client()
    pool = ThreadPoolExecutor(6)
    if stage in ("people", "all"):
        futs = [pool.submit(gen_image, c, f"protected_{i:02d}", f"passport-style portrait photo of {d}, facing camera, plain background, sharp focus", GEN / "faces" / "enroll")
                for i, d in enumerate(PROTECTED)]
        futs += [pool.submit(gen_image, c, f"impostor_{i:02d}", f"portrait photo of {d}, facing camera, plain background, sharp focus", GEN / "faces" / "impostor_src")
                 for i, d in enumerate(IMPOSTORS)]
        futs += [pool.submit(gen_image, c, f"scene_{i:02d}", f"a person {s}{STYLE}", GEN / "faces" / "scenes", 2)
                 for i, s in enumerate(SCENES)]
        [f.result() for f in futs]
    if stage in ("swaps", "all"):
        DONE = done()
        scenes = sorted((GEN / "faces" / "scenes").glob("*.png"))
        futs = []
        for i in range(len(PROTECTED)):
            src = GEN / "faces" / "enroll" / f"protected_{i:02d}_0.png"
            for k in range(4):
                dst = scenes[(i * 4 + k) % len(scenes)]
                futs.append(pool.submit(swap, c, f"swap_p{i:02d}_{k}", str(src), str(dst), GEN / "faces" / "swaps"))
        for i in range(len(IMPOSTORS)):
            src = GEN / "faces" / "impostor_src" / f"impostor_{i:02d}_0.png"
            for k in range(3):
                dst = scenes[(i * 7 + k * 3 + 1) % len(scenes)]
                futs.append(pool.submit(swap, c, f"swap_i{i:02d}_{k}", str(src), str(dst), GEN / "faces" / "swaps"))
        [f.result() for f in futs]
    if stage == "round2":
        futs = [pool.submit(gen_image, c, f"scene2_{i:02d}", s + ", realistic photo, sharp face", GEN / "faces" / "scenes2", 2)
                for i, s in enumerate(SCENES2)]
        [f.result() for f in futs]
        DONE = done()
        scenes = sorted((GEN / "faces" / "scenes2").glob("*.png"))
        futs = []
        for i in range(len(PROTECTED)):
            src = GEN / "faces" / "enroll" / f"protected_{i:02d}_0.png"
            for k in range(3):
                dst = scenes[(i * 3 + k) % len(scenes)]
                futs.append(pool.submit(swap, c, f"swap2_p{i:02d}_{k}", str(src), str(dst), GEN / "faces" / "swaps2"))
        for i in range(len(IMPOSTORS)):
            src = GEN / "faces" / "impostor_src" / f"impostor_{i:02d}_0.png"
            for k in range(2):
                dst = scenes[(30 + i * 2 + k) % len(scenes)]
                futs.append(pool.submit(swap, c, f"swap2_i{i:02d}_{k}", str(src), str(dst), GEN / "faces" / "swaps2"))
        [f.result() for f in futs]
    if stage == "negatives":
        futs = [pool.submit(gen_image, c, f"neg3_{i:02d}", f"{who}, {SCENES2[i % len(SCENES2)].split(' a person ', 1)[-1].replace('a person ', '')}, realistic photo, sharp face",
                            GEN / "faces" / "negatives", 2) for i, who in enumerate(NEG_PEOPLE)]
        [f.result() for f in futs]
    if stage in ("logos", "all"):
        futs = [pool.submit(gen_image, c, f"logoscene_{i:02d}", f"{s}, product photography, realistic", GEN / "logos" / "scenes", 2)
                for i, s in enumerate(LOGO_SCENES)]
        futs += [pool.submit(gen_image, c, f"nsfwneg_{i:02d}", f"{s}, realistic photo", GEN / "nsfw_neg", 2)
                 for i, s in enumerate(NSFW_NEG)]
        [f.result() for f in futs]
    spent = sum(r.get("credits") or 0 for r in map(json.loads, LEDGER.read_text().splitlines()))
    print("credits spent so far:", spent)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
