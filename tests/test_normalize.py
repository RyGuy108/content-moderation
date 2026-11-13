import pytest

from modpipe.text.features import featurize
from modpipe.text.normalize import normalize


@pytest.mark.parametrize("raw,expect,flag", [
    ("n.u.d.e woman", "nude woman", "spaced"),
    ("N U D E", "nude", "spaced"),
    ("nսde beach", "nude beach", "homoglyph"),     # Armenian 'ս'
    ("ｎｕｄｅ", "nude", "homoglyph"),               # fullwidth
    ("nu​de", "nude", "zero_width"),
    ("nud3 p0rn", "nude porn", "leet"),
    ("t0pl3ss", "topless", "leet"),
])
def test_deobfuscation(raw, expect, flag):
    n = normalize(raw)
    assert n.text == expect
    assert flag in n.flags


@pytest.mark.parametrize("raw", ["4k 1080p 60fps cinematic", "a b test x-ray", "$20 bill", "3d render of a cat"])
def test_leaves_normal_text_alone(raw):
    n = normalize(raw)
    assert n.text == raw.lower()
    assert not n.obfuscated


def test_gazetteer_placeholders():
    assert featurize("Taylor Swift's face").text == "<person> face"
    assert featurize("coca cola bottle").text == "<brand> bottle"
    assert featurize("the mcdonald's arches").text == "the <brand> arches"
    assert featurize("apple pie").text == "<brand_ambig> pie"
    assert featurize("t@yl0r sw1ft").text == "<person>"
