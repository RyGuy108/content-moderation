"""The keyword-blocklist filter most teams ship first. It gets the same term lists, celebrity
names, brands and characters as the real pipeline, and a minor+sexual co-occurrence rule, so
the comparison is about method, not about who wrote a longer list."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modpipe.text import lexicon as lx  # noqa: E402

MINOR_TERMS = ["child", "kid", "teen", "teenage", "minor", "schoolgirl", "little girl",
               "little boy", "toddler", "preteen", "underage", "year old", "yo"]


def _rx(terms):
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")\b")


RX = {k: _rx(v) for k, v in lx.BLOCKLIST.items()}
RX["real_person"] = _rx(lx.BLOCKLIST["real_person"] + lx.PUBLIC_FIGURES)
RX["trademark"] = _rx(lx.BLOCKLIST["trademark"] + lx.BRANDS + lx.CHARACTERS)
RX_MINOR = _rx(MINOR_TERMS)


def baseline_labels(text: str) -> set[str]:
    t = text.lower()
    out = {k for k, rx in RX.items() if rx.search(t)}
    if "sexual" in out and RX_MINOR.search(t):
        out.add("minor_sexual")
    return out
