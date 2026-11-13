"""Prompt -> hashed n-gram features.

Pipeline: normalize (undo obfuscation) -> replace gazetteer hits with placeholders -> word
uni/bi-grams + char 3-5-grams, hashed into a fixed number of buckets. Hashing keeps the model a
fixed 2^20 x d table no matter how much vocabulary it sees, and lets us featurize in pure Python
in ~30 us per prompt.
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass

from . import lexicon as lx
from .normalize import normalize

N_BUCKETS = 1 << 20
_WORD = re.compile(r"<[a-z_]+>|[a-z0-9][a-z0-9'\-]*")


def _gaz_regex(terms):
    terms = sorted({t for t in terms}, key=len, reverse=True)
    def pat(t):
        t = t.replace("'", "")
        return "'?".join(re.escape(ch) if ch not in " -" else r"[\s\-]?" for ch in t)
    alts = "|".join(pat(t) for t in terms)
    return re.compile(rf"(?<![a-z0-9])(?:{alts})(?:'s)?(?![a-z0-9])")


class Gazetteer:
    """Longest-match replacement of names/brands/characters with placeholder tokens."""

    def __init__(self, people=None, people_ambig=None, brands=None, brands_ambig=None, chars=None):
        people = people if people is not None else lx.PUBLIC_FIGURES
        people_ambig = set(people_ambig if people_ambig is not None else lx.AMBIGUOUS_FIGURES)
        brands = brands if brands is not None else lx.BRANDS
        brands_ambig = set(brands_ambig if brands_ambig is not None else lx.AMBIGUOUS_BRANDS)
        chars = chars if chars is not None else lx.CHARACTERS
        self.kind: dict[str, str] = {}
        # Order matters: unambiguous multi-word entries first, ambiguous single words last.
        for t in brands:
            self.kind[t] = "<brand>"
        for t in brands_ambig:
            self.kind[t] = "<brand_ambig>"
        for t in chars:
            self.kind[t] = "<character>"
        for t in people:
            self.kind[t] = "<person>"
        for t in people_ambig:
            # a word that is both an ambiguous brand and an ambiguous person ("jordan", "lincoln")
            self.kind[t] = "<person_ambig>" if t not in brands_ambig else "<name_ambig>"
        self.rx = _gaz_regex(self.kind)
        self._compact = {re.sub(r"[\s\-']", "", k): v for k, v in self.kind.items()}

    def sub(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        hits: list[tuple[str, str]] = []

        def rep(m):
            key = re.sub(r"[\s\-']", "", m.group(0))
            kind = self._compact.get(key) or self._compact[key[:-1]]   # possessive "'s"
            hits.append((kind, m.group(0)))
            return f" {kind} "
        return self.rx.sub(rep, text), hits


_CAPNAME = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b")
_DEFAULT_GAZ: Gazetteer | None = None


def default_gazetteer() -> Gazetteer:
    global _DEFAULT_GAZ
    if _DEFAULT_GAZ is None:
        _DEFAULT_GAZ = Gazetteer()
    return _DEFAULT_GAZ


_LEX = {k: re.compile(r"(?<![a-z])(?:" + "|".join(re.escape(t) for t in sorted(v, key=len, reverse=True)) + r")(?![a-z])")
        for k, v in {**lx.BLOCKLIST, **lx.CONCEPTS}.items()}


@dataclass
class Featurized:
    special: list[int]        # placeholders (<person>, <brand>...), obfuscation flags, blocklist hits
    ids: list[int]            # plain word tokens
    bigrams: list[int]
    chars: list[int]
    text: str                         # normalized + placeholder text (what the model saw)
    flags: set[str]
    hits: list[tuple[str, str]]


def _h(s: str) -> int:
    return zlib.crc32(s.encode()) % N_BUCKETS


def featurize(raw: str, gaz: Gazetteer | None = None) -> Featurized:
    gaz = gaz or default_gazetteer()
    norm = normalize(raw)
    text, hits = gaz.sub(norm.text)
    words = _WORD.findall(text)
    toks = [w for w in words if w.startswith("<")]
    # Capitalized multi-word spans in the raw prompt that the gazetteer missed are probably
    # names of real people ("Pete Davidson"); give the model a hint without the name itself.
    if any(_CAPNAME.finditer(raw)) and not hits:
        toks.append("<capname>")
    toks += [f"<flag_{f}>" for f in sorted(norm.flags)]
    # blocklist hits on the *normalized* text: the baseline's knowledge, minus its blind spots
    toks += [f"<lex_{k}>" for k, rx in _LEX.items() if rx.search(norm.text)]
    special = [_h("w:" + t) for t in toks] or [_h("w:<none>")]
    ids = [_h("w:" + w) for w in words if not w.startswith("<")] or [_h("w:<empty>")]
    bigrams = [_h("b:" + a + " " + b) for a, b in zip(words, words[1:])] or [_h("b:<empty>")]
    chars = []
    for w in words:
        if w.startswith("<"):
            continue
        w2 = f"^{w}$"
        for n in (3, 4, 5):
            chars += [_h("c:" + w2[i:i + n]) for i in range(max(1, len(w2) - n + 1))]
    chars = chars or [_h("c:<empty>")]
    return Featurized(special=special, ids=ids, bigrams=bigrams, chars=chars, text=" ".join(words),
                      flags=norm.flags, hits=hits)
