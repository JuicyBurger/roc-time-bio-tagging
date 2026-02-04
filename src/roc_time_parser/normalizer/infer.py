from __future__ import annotations

from datetime import date
import re
from typing import Callable

from roc_time_parser.config import Settings
from roc_time_parser.normalizer.ollama_client import query_ollama
from roc_time_parser.preprocess import preprocess
from roc_time_parser.schema import parse_dsl


_RE_HAS_QUARTER = re.compile(r"(Q[1-4]|第[1-4]季|第[一二三四]季|第一季|第二季|第三季|第四季|季度)", re.IGNORECASE)
_RE_HAS_HALF = re.compile(r"(H[12]|上半年|下半年)", re.IGNORECASE)
_RE_HAS_MONTH = re.compile(r"(\d{1,2}\s*月)", re.IGNORECASE)
_RE_HAS_MONTH_RANGE = re.compile(r"(\d{1,2}\s*[-~～至到]\s*\d{1,2}\s*月)", re.IGNORECASE)
_RE_HAS_Q1_Q3 = re.compile(r"(Q1\s*-\s*Q3|Q1\s*~\s*Q3|Q1\s*～\s*Q3|Q1\s*至\s*Q3)", re.IGNORECASE)

_RE_AD_YEAR = re.compile(r"(?<!\d)(\d{4})\s*年")
_RE_ROC_YEAR = re.compile(r"(?<!\d)(\d{2,3})\s*年")


def _has_explicit_period(span: str) -> bool:
    s = span
    return bool(
        _RE_HAS_QUARTER.search(s)
        or _RE_HAS_HALF.search(s)
        or _RE_HAS_MONTH_RANGE.search(s)
        or _RE_HAS_MONTH.search(s)
        or _RE_HAS_Q1_Q3.search(s)
    )


def _year_from_span_for_anchor(span: str) -> str:
    """
    Pick a YEAR=... string for anchor-dependent 同期 spans.
    Priority:
    - explicit AD(YYYY) via 4-digit year
    - explicit ROC via 民國 marker or 2-3 digit year (assumed=true)
    - relative last-year for 去年
    - else REL(same_period_last_year) (generic 同期 fallback)
    """
    s = span
    if "去年" in s:
        return "REL(same_period_last_year)"

    m = _RE_AD_YEAR.search(s)
    if m:
        return f"AD({int(m.group(1))})"

    # If 民國 appears, treat as ROC and assumed=false
    if "民國" in s:
        m2 = re.search(r"民國\s*(\d{2,3})\s*年", s)
        if m2:
            return f"ROC({int(m2.group(1))},assumed=false)"

    m = _RE_ROC_YEAR.search(s)
    if m:
        roc = int(m.group(1))
        # Treat as ROC; mark assumed=true since marker might be omitted.
        return f"ROC({roc},assumed=true)"

    return "REL(same_period_last_year)"


def _build_prompt(span: str, refdate: date) -> str:
    # Keep this compact but explicit: we need deterministic one-line outputs.
    return (
        "You are a strict date-span normalizer.\n"
        "Output ONLY one single line in DSL_TIME_V1. No explanations.\n"
        "\n"
        "DSL_TIME_V1 format:\n"
        "T1: YEAR=<AD(YYYY)|ROC(NNN,assumed=true|false)|REL(this_year|last_year|two_years_ago|next_year|same_period_last_year)|RANGE(YYYY-YYYY)>; "
        "PERIOD=<YEAR|Q1|Q2|Q3|Q4|H1|H2|Q1_Q3|MONTH(m)|MONTH_RANGE(m1,m2)|YTD|QTD|HTD|MTD|REL(this_quarter|last_quarter|next_quarter|this_half|last_half|next_half|this_month|last_month|next_month)|"
        "ROLLING(months=n)|ROLLING(quarters=n)|ROLLING(years=n)|ANCHOR>; "
        "[FLAGS=...]\n"
        "\n"
        f"REFDATE={refdate.isoformat()}\n"
        f"SPAN={span}\n"
    )


def _clean_model_output(s: str) -> str:
    # Keep only first line; strip code fences if any.
    t = s.strip()
    t = re.sub(r"^```.*?$", "", t, flags=re.MULTILINE).strip()
    line = t.splitlines()[0].strip() if t else ""
    return line


def normalize_span(
    span_text: str,
    *,
    refdate: date,
    settings: Settings,
    query_fn: Callable[..., object] = query_ollama,
) -> tuple[str, float]:
    """
    Normalize a span into DSL_TIME_V1.
    Returns (dsl, confidence).
    """
    span = preprocess(span_text, mode="compact")

    # Special rule: 同期 requires anchor unless explicit period is present.
    if "同期" in span and not _has_explicit_period(span):
        year_expr = _year_from_span_for_anchor(span)
        dsl = f"T1: YEAR={year_expr}; PERIOD=ANCHOR; FLAGS=NEEDS_ANCHOR"
        return dsl, 0.6

    prompt = _build_prompt(span, refdate)

    r1 = query_fn(prompt, model_name=settings.model_name, ollama_url=settings.ollama_url)
    text1 = getattr(r1, "response", "") if r1 is not None else ""
    dsl1 = _clean_model_output(str(text1))
    try:
        spec = parse_dsl(dsl1)
        # Confidence heuristic
        if "NEEDS_CLARIFICATION" in spec.flags:
            return dsl1, 0.0
        if "NEEDS_ANCHOR" in spec.flags:
            return dsl1, 0.5
        return dsl1, 0.9
    except Exception:
        # Retry with stricter prefix
        prompt2 = "OUTPUT ONLY DSL_TIME_V1.\n" + prompt
        r2 = query_fn(prompt2, model_name=settings.model_name, ollama_url=settings.ollama_url)
        text2 = getattr(r2, "response", "") if r2 is not None else ""
        dsl2 = _clean_model_output(str(text2))
        try:
            spec2 = parse_dsl(dsl2)
            if "NEEDS_CLARIFICATION" in spec2.flags:
                return dsl2, 0.0
            if "NEEDS_ANCHOR" in spec2.flags:
                return dsl2, 0.45
            return dsl2, 0.7
        except Exception:
            return "T1: FLAGS=NEEDS_CLARIFICATION", 0.0

