from __future__ import annotations

import re
from typing import Final, Literal


_DASH_MAP: Final[dict[str, str]] = {
    "～": "-",
    "~": "-",
    "—": "-",
    "–": "-",
    "－": "-",
    "‒": "-",
    "−": "-",
}


def _to_halfwidth_ascii(ch: str) -> str:
    """
    Offset-preserving conversion for full-width ASCII variants.
    - U+FF01..U+FF5E map to ASCII by -0xFEE0
    - U+3000 (full-width space) -> ' '
    """
    o = ord(ch)
    if o == 0x3000:
        return " "
    if 0xFF01 <= o <= 0xFF5E:
        return chr(o - 0xFEE0)
    return ch


def normalize_unicode(text: str) -> str:
    # 1-to-1 replacements only (safe for offsets).
    out = []
    for ch in text:
        ch2 = _to_halfwidth_ascii(ch)
        ch2 = _DASH_MAP.get(ch2, ch2)
        out.append(ch2)
    return "".join(out)


_RE_DIGIT_CONTEXT_O: Final[re.Pattern[str]] = re.compile(r"(?<=\d)O(?=\d)")
_RE_DIGIT_CONTEXT_I: Final[re.Pattern[str]] = re.compile(r"(?<=\d)[Il|](?=\d)")
_RE_QH_1: Final[re.Pattern[str]] = re.compile(r"([QH])([Il|])", re.IGNORECASE)
_RE_CHN_QUARTER: Final[re.Pattern[str]] = re.compile(r"第([一二三四])季")
_CHN_NUM_MAP: Final[dict[str, str]] = {"一": "1", "二": "2", "三": "3", "四": "4"}


def normalize_ocr(text: str) -> str:
    """
    OCR-like normalization with 1-to-1 replacements only.
    Keep this conservative to avoid changing non-time words.
    """
    t = text
    # O -> 0 in digit context: e.g. 2O25 -> 2025
    t = _RE_DIGIT_CONTEXT_O.sub("0", t)
    # I/l/| -> 1 in digit context: e.g. 20I5 -> 2015
    t = _RE_DIGIT_CONTEXT_I.sub("1", t)
    # QI/Ql/HI/Hl -> Q1/H1
    t = _RE_QH_1.sub(lambda m: m.group(1) + "1", t)
    # 第三季 (Chinese numeral) -> 第3季 (safe 1-to-1 numeral replacement)
    t = _RE_CHN_QUARTER.sub(lambda m: "第" + _CHN_NUM_MAP.get(m.group(1), m.group(1)) + "季", t)
    return t


def preprocess(text: str, mode: Literal["offset_preserving", "compact"] = "offset_preserving") -> str:
    """
    Shared preprocessing.

    - offset_preserving: MUST keep string length identical (raw offsets stay valid).
    - compact: MAY trim/collapse whitespace for nicer normalizer inputs.
    """
    if mode not in ("offset_preserving", "compact"):
        raise ValueError(f"Unknown preprocess mode: {mode}")

    t = normalize_unicode(text)
    t = normalize_ocr(t)

    if mode == "compact":
        # Compact whitespace (length-changing allowed).
        t = re.sub(r"\s+", " ", t).strip()
    return t

