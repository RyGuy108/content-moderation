"""Text normalization that undoes the common ways people dodge prompt filters.

A keyword filter sees "n.u.d.e", "nüde", "nսde" (Armenian 'ս'), "nu​de" and "nud3" as five
different strings. This module folds all of them to "nude" and reports *that* it had to, because
an obfuscated prompt is itself a signal worth routing to review.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Characters that render like Latin letters. NFKC folds fullwidth and math-alphanumeric forms
# but deliberately leaves other scripts alone, so we map the usual lookalikes ourselves.
_HOMOGLYPHS = {
    # Cyrillic
    "а": "a", "в": "b", "е": "e", "ё": "e", "к": "k", "м": "m", "н": "h", "о": "o", "р": "p",
    "с": "c", "т": "t", "у": "y", "х": "x", "і": "i", "ї": "i", "ј": "j", "ѕ": "s", "ԁ": "d",
    "ԛ": "q", "ԝ": "w", "ɡ": "g", "һ": "h", "ӏ": "l",
    # Greek
    "α": "a", "β": "b", "ε": "e", "η": "n", "ι": "i", "κ": "k", "ν": "v", "ο": "o", "ρ": "p",
    "τ": "t", "υ": "u", "χ": "x", "ω": "w",
    # Armenian / misc
    "ս": "u", "օ": "o", "ց": "g", "ո": "n", "ɑ": "a", "ı": "i", "ℓ": "l",
}
_LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a",
         "$": "s", "!": "i", "|": "l", "€": "e", "+": "t"}

_ZERO_WIDTH = re.compile("[​-‏⁠-⁤﻿­͏᠎]")
# "n u d e", "n.u.d.e", "n-u-d-e", "n_u_d_e" -> "nude". Needs 3+ single characters in a row so
# ordinary text ("a b test", "x-ray") is left alone.
_SPACED = re.compile(r"(?<![\w])((?:[^\W_][\s.\-_*·•]+){2,}[^\W_])(?![\w])")
_REPEAT = re.compile(r"(\w)\1{2,}")
_LEET_TOKEN = re.compile(r"[\w@$!|€+]+")
_WS = re.compile(r"\s+")


@dataclass
class Normalized:
    raw: str
    text: str                 # case-folded, deobfuscated text the classifier reads
    flags: set[str] = field(default_factory=set)

    @property
    def obfuscated(self) -> bool:
        return bool(self.flags & {"homoglyph", "zero_width", "spaced", "leet"})


def _deleet_token(tok: str) -> str:
    # Only rewrite digits/symbols inside a token that is mostly letters, so "4k", "1080p",
    # "24fps", "$20" and "2025" survive but "nud3" and "p0rn" do not.
    letters = sum(c.isalpha() for c in tok)
    subs = sum(c in _LEET for c in tok)
    if subs == 0 or letters < 2 or letters < subs:
        return tok
    if re.fullmatch(r"\d+[a-z]{1,3}", tok):   # 4k, 1080p, 60fps, 3d
        return tok
    return "".join(_LEET.get(c, c) for c in tok)


def normalize(raw: str) -> Normalized:
    flags: set[str] = set()
    s = raw
    if _ZERO_WIDTH.search(s):
        flags.add("zero_width")
        s = _ZERO_WIDTH.sub("", s)
    folded = unicodedata.normalize("NFKC", s)
    if folded != s and any(ord(c) > 0xFF00 or 0x1D400 <= ord(c) <= 0x1D7FF for c in s):
        flags.add("homoglyph")       # fullwidth / math-alphanumeric letters
    s = folded
    # strip diacritics: nüde -> nude
    decomposed = unicodedata.normalize("NFD", s)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    if stripped != decomposed:
        flags.add("diacritics")
    s = stripped.lower()
    mapped = "".join(_HOMOGLYPHS.get(c, c) for c in s)
    if mapped != s:
        flags.add("homoglyph")
    s = mapped

    def _join(m: re.Match) -> str:
        flags.add("spaced")
        return re.sub(r"[\s.\-_*·•]+", "", m.group(1))
    s = _SPACED.sub(_join, s)

    def _leet(m: re.Match) -> str:
        out = _deleet_token(m.group(0))
        if out != m.group(0):
            flags.add("leet")
        return out
    s = _LEET_TOKEN.sub(_leet, s)
    s = _REPEAT.sub(r"\1", s)
    s = _WS.sub(" ", s).strip()
    return Normalized(raw=raw, text=s, flags=flags)
