#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate span offsets in a JSONL dataset of the form:
  {"id": "...", "text": "...", "spans":[{"start":..,"end":..,"label":"TIME"}, ...], ...}

Focus: catch offset bugs that silently poison training (OOB, empty, leading/trailing whitespace).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class SpanIssue:
    kind: str
    rec_id: str
    line_no: int
    span_idx: int
    start: int
    end: int
    excerpt: str


def _is_ws(ch: str) -> bool:
    # `str.isspace()` covers ASCII space, ideographic space (U+3000), tabs, newlines, etc.
    return ch.isspace()


def _excerpt(text: str, start: int, end: int, window: int = 18) -> str:
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    prefix = "…" if lo > 0 else ""
    suffix = "…" if hi < len(text) else ""
    return f"{prefix}{text[lo:hi]}{suffix}"


def iter_jsonl(path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip("\n")
            if not line.strip():
                continue
            yield line_no, json.loads(line)


def validate_record(line_no: int, rec: Dict[str, Any]) -> List[SpanIssue]:
    rec_id = str(rec.get("id", f"line{line_no:06d}"))
    text = rec.get("text")
    spans = rec.get("spans") or []

    issues: List[SpanIssue] = []

    if not isinstance(text, str):
        issues.append(
            SpanIssue(
                kind="MISSING_TEXT",
                rec_id=rec_id,
                line_no=line_no,
                span_idx=-1,
                start=-1,
                end=-1,
                excerpt=repr(text),
            )
        )
        return issues

    if not isinstance(spans, list):
        issues.append(
            SpanIssue(
                kind="BAD_SPANS_TYPE",
                rec_id=rec_id,
                line_no=line_no,
                span_idx=-1,
                start=-1,
                end=-1,
                excerpt=repr(spans),
            )
        )
        return issues

    for i, s in enumerate(spans):
        if not isinstance(s, dict):
            issues.append(
                SpanIssue(
                    kind="BAD_SPAN_OBJ",
                    rec_id=rec_id,
                    line_no=line_no,
                    span_idx=i,
                    start=-1,
                    end=-1,
                    excerpt=repr(s),
                )
            )
            continue

        start = s.get("start")
        end = s.get("end")

        if not isinstance(start, int) or not isinstance(end, int):
            issues.append(
                SpanIssue(
                    kind="NON_INT_OFFSETS",
                    rec_id=rec_id,
                    line_no=line_no,
                    span_idx=i,
                    start=-1 if not isinstance(start, int) else start,
                    end=-1 if not isinstance(end, int) else end,
                    excerpt=_excerpt(text, 0, min(len(text), 1)),
                )
            )
            continue

        if start < 0 or end < 0 or start > len(text) or end > len(text):
            issues.append(
                SpanIssue(
                    kind="OOB",
                    rec_id=rec_id,
                    line_no=line_no,
                    span_idx=i,
                    start=start,
                    end=end,
                    excerpt=_excerpt(text, max(0, min(start, len(text))), max(0, min(end, len(text)))),
                )
            )
            continue

        if start >= end:
            issues.append(
                SpanIssue(
                    kind="EMPTY_OR_INVERTED",
                    rec_id=rec_id,
                    line_no=line_no,
                    span_idx=i,
                    start=start,
                    end=end,
                    excerpt=_excerpt(text, start, end),
                )
            )
            continue

        sub = text[start:end]
        if sub and _is_ws(sub[0]):
            issues.append(
                SpanIssue(
                    kind="LEADING_WS",
                    rec_id=rec_id,
                    line_no=line_no,
                    span_idx=i,
                    start=start,
                    end=end,
                    excerpt=_excerpt(text, start, end),
                )
            )
        if sub and _is_ws(sub[-1]):
            issues.append(
                SpanIssue(
                    kind="TRAILING_WS",
                    rec_id=rec_id,
                    line_no=line_no,
                    span_idx=i,
                    start=start,
                    end=end,
                    excerpt=_excerpt(text, start, end),
                )
            )

        # Very common “poison label”: spans accidentally include wrappers.
        if sub and sub[0] in "([【「":
            issues.append(
                SpanIssue(
                    kind="LEADING_WRAPPER",
                    rec_id=rec_id,
                    line_no=line_no,
                    span_idx=i,
                    start=start,
                    end=end,
                    excerpt=_excerpt(text, start, end),
                )
            )
        if sub and sub[-1] in ")]】」":
            issues.append(
                SpanIssue(
                    kind="TRAILING_WRAPPER",
                    rec_id=rec_id,
                    line_no=line_no,
                    span_idx=i,
                    start=start,
                    end=end,
                    excerpt=_excerpt(text, start, end),
                )
            )

    return issues


def main() -> None:
    # Windows consoles may default to a legacy code page. Avoid crashing on CJK output.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default="spans_labeled.jsonl", help="Path to JSONL to validate")
    ap.add_argument("--max-print", type=int, default=40, help="Max issues to print")
    args = ap.parse_args()

    counts = Counter()
    all_issues: List[SpanIssue] = []
    n_recs = 0
    n_spans = 0

    for line_no, rec in iter_jsonl(args.input):
        n_recs += 1
        n_spans += len(rec.get("spans") or [])
        issues = validate_record(line_no, rec)
        for it in issues:
            counts[it.kind] += 1
        all_issues.extend(issues)

    print(f"records={n_recs} spans={n_spans} issues={len(all_issues)}")
    if counts:
        print("by_kind:")
        for k, v in counts.most_common():
            print(f"  {k}: {v}")

    if all_issues:
        print("\nexamples:")
        for it in all_issues[: args.max_print]:
            print(
                f"- {it.kind} id={it.rec_id} line={it.line_no} span#{it.span_idx} "
                f"[{it.start},{it.end}): {it.excerpt}"
            )


if __name__ == "__main__":
    main()

