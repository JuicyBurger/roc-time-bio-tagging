#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic zh_TW finance-style prompt generator for time span extraction/normalization.

Outputs JSONL records:
- raw prompts: {"id": "...", "text": "..."}
- optionally with spans: {"id": "...", "text": "...", "spans":[{"start":..,"end":..,"label":"TIME"}, ...], "meta": {...}}

Design goals:
- prompts "sound human" via templates + constrained time grammar
- supports ROC/AD/relative years, quarter/half, month ranges, rolling windows, as-of, YTD/QTD/MTD, comparisons (同期)
- noise injection (OCR-ish): full-width, dash variants, OCR confusions, spacing/brackets (with offset mapping)
"""

from __future__ import annotations
import argparse
import json
import random
import re
from typing import List, Tuple, Dict

# Words that connect two time expressions but should NOT be labeled as TIME spans.
GLUE_WORDS = (
    "與",
    "与",
    "和",
    "及",
    "以及",
    "跟",
    "vs",
    "VS",
    "V.S.",
    "v.s.",
    "對比",
    "相比",
)
GLUE_WORD_SET = set(GLUE_WORDS)

FW_DIGITS = str.maketrans({
    '0':'０','1':'１','2':'２','3':'３','4':'４','5':'５','6':'６','7':'７','8':'８','9':'９',
    'Q':'Ｑ','H':'Ｈ','-':'－'
})

DASH_VARIANTS = ['-', '－', '–', '—', '～', '~']

OCR_CONFUSIONS = [
    ('Q1','Ql'), ('Q1','QI'), ('Q2','QZ'), ('H1','Hl'), ('H2','HZ'),
    ('2025','2O25'), ('2024','2O24'), ('2023','2O23'), ('114','l14'), ('113','l13'),
]

def rand_year_ad(rng: random.Random, lo=2019, hi=2027) -> int:
    return rng.randint(lo, hi)

def rand_year_roc(rng: random.Random, lo=108, hi=116) -> int:
    return rng.randint(lo, hi)

_REL_YEAR_SURFACES = {
    "this_year": ["今年", "本年", "本年度", "當年", "当年", "當年度", "当年度", "該年", "该年"],
    "last_year": ["去年", "上年度"],
    "two_years_ago": ["前年"],
    "next_year": ["明年", "次年"],
}

def year_expr(rng: random.Random) -> Tuple[str, Dict]:
    choice = rng.random()
    if choice < 0.45:
        y = rand_year_ad(rng)
        # Western era markers are valid but rarer in typical finance prompts than plain "YYYY年".
        surface = rng.choices([f"{y}年", f"西元{y}年", f"公元{y}年"], [85, 10, 5], k=1)[0]
        return surface, {"year_type": "ad", "year": y, "surface": surface}
    if choice < 0.75:
        ry = rand_year_roc(rng)
        tpl = rng.choice(["民國{y}年", "民国{y}年", "{y}年"])
        surface = tpl.format(y=ry)
        assumed = tpl == "{y}年"
        return surface, {"year_type": "roc", "roc_year": ry, "assumed": assumed, "surface": surface}

    rel = rng.choices(
        ["this_year", "last_year", "two_years_ago", "next_year"],
        [40, 25, 20, 15],
        k=1,
    )[0]
    surface = rng.choice(_REL_YEAR_SURFACES[rel])
    return surface, {"year_type": "relative", "rel": rel, "surface": surface}


def explicit_year_expr(rng: random.Random) -> Tuple[str, Dict]:
    """AD/ROC only (useful for things like '{YEAR}同期')."""
    if rng.random() < 0.62:
        y = rand_year_ad(rng)
        surface = rng.choices([f"{y}年", f"西元{y}年", f"公元{y}年"], [85, 10, 5], k=1)[0]
        return surface, {"year_type": "ad", "year": y, "surface": surface}

    ry = rand_year_roc(rng)
    tpl = rng.choice(["民國{y}年", "民国{y}年", "{y}年"])
    surface = tpl.format(y=ry)
    assumed = tpl == "{y}年"
    return surface, {"year_type": "roc", "roc_year": ry, "assumed": assumed, "surface": surface}

def quarter_expr(rng: random.Random) -> Tuple[str, Dict]:
    q = rng.randint(1,4)
    cn = ['一','二','三','四'][q-1]
    style = rng.choice([
        f"Q{q}", f"Ｑ{q}", f"{q}Q",
        f"第{q}季", f"第{cn}季",
        f"第{q} 季", f"第 {q} 季",
        f"第{q}季度", f"第{cn}季度",
        f"第{q} 季度", f"第 {q} 季度",
        f"第一季" if q==1 else f"第{cn}季",
        f"第一季度" if q==1 else f"第{cn}季度",
    ])
    return style, {"period":"quarter", "q":q}

def half_expr(rng: random.Random) -> Tuple[str, Dict]:
    h = rng.choice([1,2])
    style = rng.choice([
        "上半年" if h==1 else "下半年",
        "上半年度" if h==1 else "下半年度",
        f"H{h}", f"Ｈ{h}",
        "上 半年" if h==1 else "下 半年",
    ])
    return style, {"period":"half", "h":h}

def q1_q3_expr(rng: random.Random) -> Tuple[str, Dict]:
    style = rng.choice([
        "前三季", "前3季", "截至第三季", "截至第3季", "截至第三季度",
        "截至Q3", "截至Ｑ3", "截至3Q",
        "截止第三季", "截止Q3", "截止Ｑ3",
        "Q1-Q3", "Q1–Q3", "Q1~Q3", "Q1到Q3", "Q1至Q3",
        "第一季至第三季", "第一季到第三季",
        "Q1－Q3", "Ｑ1－Ｑ3",
    ])
    return style, {"period":"q1_q3"}

def month_range_expr(rng: random.Random) -> Tuple[str, Dict]:
    m1 = rng.randint(1, 11)
    m2 = rng.randint(m1, min(12, m1+rng.randint(1,3)))
    dash = rng.choice(DASH_VARIANTS)
    style = rng.choice([
        f"{m1}{dash}{m2}月",
        f"{m1}月{rng.choice(['到','至','-','～','~'])}{m2}月",
        f"{m1}{dash}{m2} 月",
    ])
    return style, {"period":"month_range", "m1":m1, "m2":m2}

def month_expr(rng: random.Random) -> Tuple[str, Dict]:
    m = rng.randint(1, 12)
    style = rng.choice([f"{m}月", f"{m} 月", f"{m}月份", f"{m} 月份"])
    return style, {"period": "month", "m": m}


_REL_PERIOD_SURFACES = {
    "this_month": ["本月", "這個月", "这个月", "當月", "当月"],
    "last_month": ["上月", "上個月", "上个月"],
    "next_month": ["下月", "下個月", "下个月"],
    "this_quarter": ["本季", "本季度"],
    "last_quarter": ["上季", "上季度"],
    "next_quarter": ["下季", "下季度"],
}


def rel_period_expr(rng: random.Random) -> Tuple[str, Dict]:
    kind = rng.choices(
        ["this_month", "last_month", "next_month", "this_quarter", "last_quarter", "next_quarter"],
        [22, 12, 10, 24, 18, 14],
        k=1,
    )[0]
    surface = rng.choice(_REL_PERIOD_SURFACES[kind])
    return surface, {"period": "rel_period", "kind": kind, "surface": surface}


def week_rel_expr(rng: random.Random) -> Tuple[str, Dict]:
    kind = rng.choices(["this", "last", "next"], [45, 30, 25], k=1)[0]
    if kind == "this":
        surface = rng.choice(["本週", "本周", "這週", "这周", "本星期", "本禮拜", "本礼拜"])
    elif kind == "last":
        surface = rng.choice(["上週", "上周", "上星期", "上禮拜", "上礼拜"])
    else:
        surface = rng.choice(["下週", "下周", "下星期", "下禮拜", "下礼拜"])
    return surface, {"period": "week_rel", "kind": kind, "surface": surface}


def rolling_weeks_expr(rng: random.Random) -> Tuple[str, Dict]:
    n = rng.choice([1, 2, 4, 8, 12])
    unit = rng.choice(["週", "周"])
    surface = rng.choice(
        [
            f"近{n}{unit}",
            f"最近{n}{unit}",
            f"過去{n}{unit}",
            f"过去{n}{unit}",
            f"前{n}{unit}",
        ]
    )
    return surface, {"period": "rolling_weeks", "n": n, "surface": surface}


def rolling_days_expr(rng: random.Random) -> Tuple[str, Dict]:
    n = rng.choice([7, 14, 30, 60, 90])
    surface = rng.choice([f"近{n}天", f"最近{n}天", f"過去{n}天", f"过去{n}天", f"前{n}天"])
    return surface, {"period": "rolling_days", "n": n, "surface": surface}


def _year_for_date_expr(rng: random.Random) -> str:
    # Keep this simple/natural; avoid "本年度" style surfaces for day-level dates.
    if rng.random() < 0.62:
        return explicit_year_expr(rng)[0]
    return rng.choice(["今年", "去年", "前年", "明年"])


def date_expr(rng: random.Random) -> Tuple[str, Dict]:
    y = _year_for_date_expr(rng) if rng.random() < 0.72 else ""
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)  # safe day across all months
    style = rng.choice(
        [
            f"{m}月{d}日",
            f"{m}月{d}號",
            f"{m}月{d}号",
            f"{m}/{d}",
            f"{m}-{d}",
        ]
    )
    surface = f"{y}{style}" if y else style
    return surface, {"period": "date", "surface": surface}


def date_range_expr(rng: random.Random) -> Tuple[str, Dict]:
    y = _year_for_date_expr(rng) if rng.random() < 0.78 else ""
    m = rng.randint(1, 12)
    d1 = rng.randint(1, 25)
    d2 = rng.randint(d1, min(28, d1 + rng.randint(1, 6)))
    sep = rng.choice(["-", "～", "~", "至", "到", "–", "—", "－"])

    # Prefer forms that people actually type in finance/email/chat.
    if rng.random() < 0.55:
        # Chinese form with explicit 月/日 (same month)
        if rng.random() < 0.55:
            inner = f"{m}月{d1}日{sep}{d2}日"
        else:
            inner = f"{m}月{d1}日{sep}{m}月{d2}日"
    else:
        # Slash form
        inner = f"{m}/{d1}{sep}{m}/{d2}"

    surface = f"{y}{inner}" if y else inner
    return surface, {"period": "date_range", "surface": surface}


def month_boundary_expr(rng: random.Random) -> Tuple[str, Dict]:
    # These are common natural-language fragments, but downstream parsing may treat them as ambiguous.
    choice = rng.random()
    if choice < 0.55:
        # Relative month boundary
        if rng.random() < 0.18:
            base2 = rng.choice(["本月", "上月", "下月"])
            half = rng.choice(["上半月", "下半月"])
            surface = f"{base2}{half}"
            return surface, {"period": "month_boundary", "surface": surface}

        base = rng.choice(["本", "上", "下"])
        tail = rng.choice(["初", "中", "底", "末"])
        if tail == "底":
            surface = f"{base}月底"
        else:
            surface = f"{base}月{tail}"
        return surface, {"period": "month_boundary", "surface": surface}

    if choice < 0.85:
        # Explicit month boundary / x旬
        y = _year_for_date_expr(rng) if rng.random() < 0.45 else ""
        m = rng.randint(1, 12)
        tail = rng.choice(["月初", "月中", "月底", "月末", "上旬", "中旬", "下旬"])
        if tail.startswith("月"):
            inner = f"{m}{tail}"
        else:
            inner = f"{m}月{tail}"
        surface = f"{y}{inner}" if y else inner
        return surface, {"period": "month_boundary", "surface": surface}

    # Year/quarter boundary words
    surface = rng.choice(["年初", "年中", "年底", "年末", "季初", "季中", "季末"])
    return surface, {"period": "boundary", "surface": surface}


def year_only_expr(rng: random.Random) -> Tuple[str, Dict]:
    surface, meta = year_expr(rng)
    # Explicit years often appear as "YYYY年度"/"民國NNN年度" in finance reporting.
    if meta.get("year_type") in {"ad", "roc"} and rng.random() < 0.35 and surface.endswith("年"):
        surface = surface[:-1] + "年度"
    return surface, {"period": "year_only", **meta}


def rolling_expr(rng: random.Random) -> Tuple[str, Dict]:
    kind = rng.choices(["months", "quarters", "years"], [40, 20, 40], k=1)[0]

    if kind == "months":
        n = rng.choice([1, 2, 3, 6, 9, 12])
        if n == 6 and rng.random() < 0.30:
            return "近半年", {"period": "rolling", "unit": "months", "n": 6}
        style = rng.choice(
            [
                f"近{n}個月",
                f"近{n}个月",
                f"近{n}月",
                f"最近{n}個月",
                f"過去{n}個月",
                f"过去{n}个月",
            ]
        )
        return style, {"period": "rolling", "unit": "months", "n": n}

    if kind == "quarters":
        n = rng.choice([2, 3, 4, 6, 8])
        style = rng.choice(
            [
                f"近{n}季",
                f"近{n}季度",
                f"近{n}個季度",
                f"最近{n}季",
                f"過去{n}季",
                f"过去{n}季",
            ]
        )
        return style, {"period": "rolling", "unit": "quarters", "n": n}

    n = rng.choice([1, 2, 3, 4, 5])
    variants = [f"近{n}年", f"最近{n}年", f"過去{n}年", f"过去{n}年", f"前{n}年"]
    if n == 2:
        variants.extend(["近兩年", "近二年"])
    if n == 3:
        variants.extend(["近三年", "過去三年", "过去三年", "前三年"])
    style = rng.choice(variants)
    return style, {"period": "rolling", "unit": "years", "n": n}

def ytd_qtd_mtd_expr(rng: random.Random) -> Tuple[str, Dict]:
    style = rng.choice(
        [
            "年初至今",
            "年初迄今",
            "年初到目前為止",
            "年初到目前为止",
            "本季至今",
            "本季度至今",
            "本月至今",
            "半年度至今",
            "半年度迄今",
            "YTD",
            "QTD",
            "MTD",
            "HTD",
        ]
    )
    key = {
        "年初至今": "ytd",
        "年初迄今": "ytd",
        "年初到目前為止": "ytd",
        "年初到目前为止": "ytd",
        "本季至今": "qtd",
        "本季度至今": "qtd",
        "本月至今": "mtd",
        "半年度至今": "htd",
        "半年度迄今": "htd",
        "YTD": "ytd",
        "QTD": "qtd",
        "MTD": "mtd",
        "HTD": "htd",
    }[style]
    return style, {"period": key}

def compare_glue(rng: random.Random) -> str:
    return rng.choice(list(GLUE_WORDS))

def make_time_parts(rng: random.Random, *, profile: str = "pipeline") -> List[str]:
    if profile not in {"pipeline", "broad"}:
        raise ValueError(f"Unknown profile: {profile}")

    if profile == "pipeline":
        intent = rng.choices(
            [
                "year_quarter",
                "year_half",
                "year_q1q3",
                "year_month_range",
                "year_month",
                "year_only",
                "month_only",
                "rel_period",
                "rolling_only",
                "ytdqtdmtd",
                "compare_yoy",
                "compare_two",
            ],
            [19, 10, 7, 7, 8, 8, 4, 8, 8, 8, 9, 4],
            k=1,
        )[0]
    else:
        # Broad profile adds week/day/date-like spans. These are useful for extractor robustness but
        # may not be fully supported by the downstream normalizer/resolver yet.
        intent = rng.choices(
            [
                "year_quarter",
                "year_half",
                "year_q1q3",
                "year_month_range",
                "year_month",
                "year_only",
                "month_only",
                "rel_period",
                "rolling_only",
                "ytdqtdmtd",
                "compare_yoy",
                "compare_two",
                "week_rel",
                "rolling_weeks",
                "rolling_days",
                "date",
                "date_range",
                "month_boundary",
            ],
            [16, 8, 6, 6, 7, 7, 4, 7, 7, 7, 8, 3, 3, 2, 2, 3, 2, 2],
            k=1,
        )[0]

    if intent == "year_quarter":
        y, _ = year_expr(rng); q, _ = quarter_expr(rng)
        return [f"{y}{q}"]
    if intent == "year_half":
        y, _ = year_expr(rng); h, _ = half_expr(rng)
        return [f"{y}{h}"]
    if intent == "year_q1q3":
        y, _ = year_expr(rng); p, _ = q1_q3_expr(rng)
        if p.startswith(("截至", "截止")):
            # Prefer "截至{YEAR}Q3" over "{YEAR}截至Q3"
            return [f"{p[:2]}{y}{p[2:]}"]
        return [f"{y}{p}"]
    if intent == "year_month_range":
        y, _ = year_expr(rng)
        m_surface, m_meta = month_range_expr(rng)
        # Add common range framing ("自/從...起...至...") sometimes.
        if rng.random() < 0.22:
            prefix = rng.choice(["自", "從", "从"])
            sep = rng.choice(["到", "至"])
            m1 = m_meta["m1"]
            m2 = m_meta["m2"]
            if rng.random() < 0.45:
                return [f"{prefix}{y}{m1}月起{sep}{m2}月"]
            return [f"{prefix}{y}{m1}月{sep}{m2}月"]
        return [f"{y}{m_surface}"]
    if intent == "year_month":
        y, _ = year_expr(rng); m, _ = month_expr(rng)
        return [f"{y}{m}"]
    if intent == "year_only":
        y, _ = year_only_expr(rng)
        return [y]
    if intent == "month_only":
        m, _ = month_expr(rng)
        return [m]
    if intent == "rel_period":
        p, _ = rel_period_expr(rng)
        return [p]
    if intent == "rolling_only":
        r, _ = rolling_expr(rng)
        return [r]
    if intent == "ytdqtdmtd":
        t, _ = ytd_qtd_mtd_expr(rng)
        if rng.random() < 0.45:
            y, _ = year_expr(rng)
            return [f"{y}{t}"]
        return [t]
    if intent == "compare_yoy":
        # anchor + last-year same period
        left = make_time_parts(rng, profile=profile)[0]
        glue = compare_glue(rng)
        right = rng.choice(["去年同期", "前年同期", "上年同期", "去年 同期"])
        if rng.random() < 0.35:
            y, _ = explicit_year_expr(rng)
            right = f"{y}同期"
        return [left, glue, right]
    if intent == "compare_two":
        left = f"{year_expr(rng)[0]}{quarter_expr(rng)[0]}"
        right = f"{year_expr(rng)[0]}{quarter_expr(rng)[0]}"
        glue = compare_glue(rng)
        return [left, glue, right]

    if intent == "week_rel":
        w, _ = week_rel_expr(rng)
        return [w]
    if intent == "rolling_weeks":
        w, _ = rolling_weeks_expr(rng)
        return [w]
    if intent == "rolling_days":
        d, _ = rolling_days_expr(rng)
        return [d]
    if intent == "date":
        d, _ = date_expr(rng)
        return [d]
    if intent == "date_range":
        d, _ = date_range_expr(rng)
        return [d]
    if intent == "month_boundary":
        b, _ = month_boundary_expr(rng)
        return [b]
    raise RuntimeError("unknown intent")

DEPTS = ["植保事業群各事業部","台灣植保部","植保事業群","植保事業部","植保事業群-台灣區","台灣區植保事業群"]
MEASURES = ["年終獎金","獎金","薪資","營收","毛利","費用"]
QUALIFIERS = [
    "",
    "實際與預算",
    "實際 vs 預算",
    "各月拆分",
    "各季彙總",
    "分月比較",
    "同比分析",
    "同比",
    "年增",
    "年減",
    "年减",
    "年增率",
    "年增減",
    "累計",
    "累计",
    "彙總",
    "汇总",
    "合計",
    "合计",
    "小計",
    "小计",
    "累積",
    "累积",
    "按月",
    "按季",
    "按年",
    "分月",
    "分季",
    "分年",
    "月度彙總",
    "月度汇总",
    "季度彙總",
    "季度汇总",
    "年度彙總",
    "年度汇总",
]
VERBS = ["請彙整","我想查","麻煩拉一下","幫我看","需要分析","請提供","請整理"]
TEMPLATES = [
    "{verb}：{dept}{time}{measure}{qual}",
    "{dept}{time}{measure}{qual}",
    "{verb} {dept}{time}{measure}{qual}",
    "【速報】{dept}{time}{measure}{qual}",
    "{dept}：{time}{measure}{qual}",
    "針對{dept}，{time}{measure}{qual}",
]

# Patterns to find time-like phrases that might appear in qualifiers or elsewhere in the prompt.
# These are additional to the explicit time_parts spans.
# Strategy: Match time qualifiers as separate spans (e.g., "年度" separate from "彙總").
_TIME_PHRASE_PATTERNS = [
    # Year qualifiers: 年度, 年終 (as standalone time words)
    re.compile(r"年度"),
    re.compile(r"年終"),
    # Month qualifiers: 月度, 各月, 分月
    re.compile(r"月度"),
    re.compile(r"各月"),
    re.compile(r"分月"),
    # Quarter qualifiers: 季度, 各季, 分季
    re.compile(r"季度"),
    re.compile(r"各季"),
    re.compile(r"分季"),
    # Week qualifiers: 週度, 周度, 各週, 各周, 分週, 分周
    re.compile(r"週度"),
    re.compile(r"周度"),
    re.compile(r"各週"),
    re.compile(r"各周"),
    re.compile(r"分週"),
    re.compile(r"分周"),
    # Time comparison/analysis: 同比, 年增, 年減, 年增率, etc.
    re.compile(r"同比"),
    re.compile(r"年增(?!率|減|减)"),
    re.compile(r"年增率"),
    re.compile(r"年增減"),
    re.compile(r"年減(?!率)"),
    re.compile(r"年减(?!率)"),
    re.compile(r"年減率"),
    re.compile(r"年减率"),
]


def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merge overlapping or adjacent spans.
    Returns sorted, non-overlapping spans.
    """
    if not spans:
        return []
    # Sort by start, then by end
    sorted_spans = sorted(spans)
    merged = [sorted_spans[0]]
    for start, end in sorted_spans[1:]:
        last_start, last_end = merged[-1]
        # Merge if overlapping or adjacent (within 1 char)
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def find_all_time_phrases(text: str, existing_spans: List[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
    """
    Find all time-like phrases in the text using regex patterns.
    Returns list of (start, end) spans that are NOT already covered by existing_spans.
    
    Args:
        text: Full text to search
        existing_spans: Spans already found from time_parts (to avoid double-labeling)
    """
    if existing_spans is None:
        existing_spans = []
    
    found_spans: List[Tuple[int, int]] = []
    for pattern in _TIME_PHRASE_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            # Skip if this span is entirely within a glue word (shouldn't happen but be safe)
            span_text = text[start:end]
            if span_text in GLUE_WORD_SET:
                continue
            
            # Skip if this span is entirely contained within an existing span
            # (e.g., if "年度" is part of "2024年度", don't label it separately)
            is_contained = False
            for ex_start, ex_end in existing_spans:
                if ex_start <= start and end <= ex_end:
                    is_contained = True
                    break
            if is_contained:
                continue
            
            found_spans.append((start, end))
    return found_spans


def assemble(rng: random.Random, *, profile: str = "pipeline") -> Tuple[str, List[Tuple[int,int]], Dict]:
    dept = rng.choice(DEPTS)
    measure = rng.choice(MEASURES)
    qual = rng.choice(QUALIFIERS)
    verb = rng.choice(VERBS)
    parts = make_time_parts(rng, profile=profile)

    # join the time surface form
    time_text = "".join(parts)

    # wrap sometimes
    wrap = rng.random()
    if wrap < 0.20:
        time_surface = f"({time_text})"
    elif wrap < 0.30:
        time_surface = f"【{time_text}】"
    else:
        time_surface = time_text

    if qual:
        qual = (" " + qual) if rng.random() < 0.6 else qual

    text = rng.choice(TEMPLATES).format(verb=verb, dept=dept, time=time_surface, measure=measure, qual=qual)

    # compute spans for TIME pieces (not glue words)
    spans: List[Tuple[int,int]] = []
    time_offset = text.find(time_surface)
    if time_offset < 0:
        raise RuntimeError("internal error: time_surface not found in assembled text")

    surface_prefix_len = 0
    if time_surface.startswith("(") and time_surface.endswith(")"):
        surface_prefix_len = 1
    elif time_surface.startswith("【") and time_surface.endswith("】"):
        surface_prefix_len = 1

    cursor = 0
    for p in parts:
        if p in GLUE_WORD_SET:
            cursor += len(p)
            continue
        start = time_offset + surface_prefix_len + cursor
        end = start + len(p)
        spans.append((start, end))
        cursor += len(p)

    # Find all additional time-like phrases in the full text (e.g., "年度" in "年度彙總")
    # Pass existing spans to avoid double-labeling parts already covered
    additional_spans = find_all_time_phrases(text, existing_spans=spans)
    
    # Merge all spans (from time_parts + additional finds)
    all_spans = spans + additional_spans
    merged_spans = _merge_spans(all_spans)

    meta = {"dept":dept, "measure":measure, "qual":qual.strip(), "time_text":time_text, "time_parts":parts}
    return text, merged_spans, meta

def apply_noise(rng: random.Random, text: str, level: str) -> Tuple[str, List[int], List[int], Dict]:
    if level == "clean":
        ident = list(range(len(text) + 1))
        return text, ident, ident, {"level": "clean", "ops": []}

    probs = {
        "mild":   {"fullwidth":0.25, "dash":0.35, "ocr":0.15, "spaces":0.10, "wrap":0.05},
        "medium": {"fullwidth":0.55, "dash":0.55, "ocr":0.35, "spaces":0.25, "wrap":0.10},
        "heavy":  {"fullwidth":0.75, "dash":0.65, "ocr":0.55, "spaces":0.45, "wrap":0.20},
    }[level]

    ops = []
    prefix = ""
    suffix = ""
    if rng.random() < probs["wrap"]:
        prefix = rng.choice(["【","[","「"])
        suffix = {"【":"】","[":"]","「":"」"}[prefix]
        ops.append("wrap")

    out = []
    # We insert whitespace BEFORE a character. That means span starts need a different projection than span ends:
    # - `pos_end[k]`: original offset k -> noisy offset BEFORE inserted whitespace for character k
    # - `pos_start[k]`: original offset k -> noisy offset AFTER inserted whitespace for character k
    # Project a base span [a,b) as: noisy_start = pos_start[a], noisy_end = pos_end[b].
    pos_end = [0] * (len(text) + 1)
    pos_start = [0] * (len(text) + 1)
    j = 0
    pos_end[0] = 0

    for i, ch in enumerate(text):
        pos_end[i] = j
        if rng.random() < probs["spaces"]:
            out.append(rng.choice([" ","　"]))
            j += 1
            ops.append("ins_space")
        pos_start[i] = j

        ch2 = ch
        if ch2 in "-－–—～~" and rng.random() < probs["dash"]:
            ch2 = rng.choice(DASH_VARIANTS)
            ops.append("dash")
        if ch2 in "0123456789QH-" and rng.random() < probs["fullwidth"]:
            ch2 = ch2.translate(FW_DIGITS)
            ops.append("fullwidth")

        out.append(ch2)
        j += 1
        pos_end[i + 1] = j
    pos_start[len(text)] = pos_end[len(text)]

    noisy = prefix + "".join(out) + suffix
    if prefix:
        shift = len(prefix)
        pos_end = [p + shift for p in pos_end]
        pos_start = [p + shift for p in pos_start]

    # same-length OCR swaps
    if rng.random() < probs["ocr"]:
        for a,b in OCR_CONFUSIONS:
            if len(a)==len(b) and a in noisy and rng.random() < 0.5:
                noisy = noisy.replace(a,b,1)
                ops.append(f"ocr:{a}->{b}")

    return noisy, pos_start, pos_end, {"level": level, "ops": ops}

def generate_records(n: int, seed: int, noise: str, with_spans: bool, *, profile: str = "pipeline") -> List[Dict]:
    rng = random.Random(seed)
    recs = []
    for i in range(1, n+1):
        base_text, base_spans, meta = assemble(rng, profile=profile)
        noisy_text, pos_start, pos_end, noise_meta = apply_noise(rng, base_text, noise)

        rec = {"id": f"s{i:06d}", "text": noisy_text}
        if with_spans:
            rec["spans"] = [
                {"start": pos_start[a], "end": pos_end[b], "label": "TIME"} for a, b in base_spans
            ]
            # rec["meta"] = {"noise": noise_meta, "base_text": base_text, "base_spans": base_spans, **meta}
        recs.append(rec)
    return recs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise", choices=["clean","mild","medium","heavy"], default="medium")
    ap.add_argument("--out", type=str, default="raw_prompts.jsonl")
    ap.add_argument("--with_spans", action="store_true")
    ap.add_argument(
        "--profile",
        choices=["pipeline", "broad"],
        default="pipeline",
        help="Pattern coverage profile. 'pipeline' stays within current DSL/resolver scope; "
        "'broad' adds week/day/date-like spans (useful for extractor robustness).",
    )
    args = ap.parse_args()

    recs = generate_records(args.n, args.seed, args.noise, args.with_spans, profile=args.profile)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(recs)} records to {args.out}")

if __name__ == "__main__":
    main()
