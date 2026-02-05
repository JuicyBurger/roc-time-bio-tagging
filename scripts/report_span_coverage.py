#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick coverage report for generated span datasets.

Reads JSONL records:
  {"id": "...", "text": "...", "spans":[{"start":..,"end":..,"label":"TIME"}, ...]}

Counts how often important time-expression patterns appear INSIDE labeled spans.
Useful for sanity-checking prompt generator coverage.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class Record:
    rec_id: str
    text: str
    spans: list[tuple[int, int]]


def iter_records(path: str) -> Iterable[Record]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip("\n")
            if not line.strip():
                continue
            r: Dict[str, Any] = json.loads(line)
            rec_id = str(r.get("id", f"line{line_no:06d}"))
            text = r.get("text", "")
            spans_raw = r.get("spans") or []
            spans: list[tuple[int, int]] = []
            for sp in spans_raw:
                if not isinstance(sp, dict):
                    continue
                try:
                    a = int(sp["start"])
                    b = int(sp["end"])
                except Exception:
                    continue
                if 0 <= a < b <= len(text):
                    spans.append((a, b))
            yield Record(rec_id=rec_id, text=text, spans=spans)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default="spans_labeled.jsonl")
    args = ap.parse_args()

    # Patterns: aim for high signal over perfection.
    pats: list[tuple[str, re.Pattern[str]]] = [
        ("rel_month", re.compile(r"(本月|上月|下月|上個月|上个月|下個月|下个月|這個月|这个月|當月|当月)")),
        ("rel_quarter", re.compile(r"(本季|上季|下季|本季度|上季度|下季度)")),
        ("year_markers", re.compile(r"(西元|公元|民國|民国)")),
        # Don't use word boundaries: Chinese characters are \w in Python regex, so "年1Q" would not match \b.
        ("quarter_1Q", re.compile(r"(?i)[1-4]Q")),
        ("quarter_Q1", re.compile(r"(?i)Q[1-4]|Ｑ[1-4]")),
        ("quarter_chinese", re.compile(r"(第\s*[1-4]\s*季|第[一二三四]季|第一季|第二季|第三季|第四季|季度)")),
        ("half", re.compile(r"(?i)(H[12]|Ｈ[12]|上半年|下半年|上半年度|下半年度)")),
        ("month_range", re.compile(r"(\d{1,2}\s*[-~～至到]\s*\d{1,2}\s*月)")),
        ("month_any", re.compile(r"(\d{1,2}\s*月)")),
        (
            "to_date",
            re.compile(
                r"(年初至今|年初迄今|本季至今|本季度至今|本月至今|半年度至今|半年度迄今|YTD|QTD|MTD|HTD)",
                re.IGNORECASE,
            ),
        ),
        ("as_of", re.compile(r"(截至|截止)")),
        ("rolling_years", re.compile(r"(近\d+\s*年|最近\d+\s*年|過去\d+\s*年|过去\d+\s*年|前\d+\s*年)")),
        ("rolling_months", re.compile(r"(近\d+\s*(月|個月|个月)|最近\d+\s*(月|個月|个月))")),
        ("rolling_quarters", re.compile(r"(近\d+\s*(季|季度)|最近\d+\s*季|過去\d+\s*季|过去\d+\s*季)")),
        ("q1_q3", re.compile(r"(前三季|前3季|Q1\s*[-–—－~～至到]\s*Q3|Q1\s*[-–—－~～]\s*Q3|截至.*Q3|截至.*第三季|截止.*Q3|截止.*第三季)")),
        ("week_rel", re.compile(r"(本週|本周|上週|上周|下週|下周|這週|这周|本星期|上星期|下星期|本禮拜|上禮拜|下禮拜|本礼拜|上礼拜|下礼拜)")),
        ("rolling_weeks", re.compile(r"(近\d+\s*(週|周)|最近\d+\s*(週|周)|過去\d+\s*(週|周)|过去\d+\s*(週|周)|前\d+\s*(週|周))")),
        ("rolling_days", re.compile(r"(近\d+\s*天|最近\d+\s*天|過去\d+\s*天|过去\d+\s*天|前\d+\s*天)")),
        ("date_any", re.compile(r"(\d{1,2}月\d{1,2}(日|號|号)|\d{1,2}[/-]\d{1,2}|\d{4}/\d{1,2}/\d{1,2})")),
        ("boundary_terms", re.compile(r"(月初|月中|月底|月末|上旬|中旬|下旬|上半月|下半月|年初|年中|年底|年末|季初|季中|季末)")),
    ]

    glue_words = {"與", "与", "和", "及", "以及", "跟", "vs", "VS", "V.S.", "v.s.", "對比", "相比"}

    n_recs = 0
    n_spans = 0
    hit_counts = Counter()
    glue_labeled = 0

    for rec in iter_records(args.input):
        n_recs += 1
        for a, b in rec.spans:
            n_spans += 1
            s = rec.text[a:b]
            if s in glue_words:
                glue_labeled += 1
            for name, rx in pats:
                if rx.search(s):
                    hit_counts[name] += 1

    print(f"records={n_recs} spans={n_spans}")
    if glue_labeled:
        print(f"WARNING: glue_labeled_spans={glue_labeled} (should be 0)")

    if n_spans == 0:
        return

    print("hits (spans containing pattern):")
    for name, _ in pats:
        c = hit_counts.get(name, 0)
        pct = 100.0 * c / n_spans
        print(f"  {name:14s} {c:6d}  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()

