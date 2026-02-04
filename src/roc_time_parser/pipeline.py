from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

from roc_time_parser.policy import Policy
from roc_time_parser.resolver import ResolvedRange, resolve, resolve_with_anchor
from roc_time_parser.schema import PeriodAnchor, ValidatedSpec, parse_dsl, to_json


@dataclass(frozen=True)
class Models:
    extractor_model: Any
    extractor_tokenizer: Any
    normalizer_settings: Any  # roc_time_parser.config.Settings


def _range_to_json(r: ResolvedRange) -> dict[str, Any]:
    return {
        "start": r.start.isoformat(),
        "end": r.end.isoformat(),
        "grain": r.grain,
        "warnings": list(r.warnings),
        "source_text": r.source_text,
    }


def _pick_anchor_index(
    spans_out: list[dict[str, Any]],
    i_target: int,
) -> Optional[int]:
    """
    Deterministic anchor selection:
    - prefer nearest resolved span to the left
    - tie: higher extractor score; then higher normalizer confidence
    - if none left: consider right side with same tie-breakers
    """
    target = spans_out[i_target]
    t_start = int(target["start"])

    candidates: list[tuple[int, int, float, float]] = []
    for i, s in enumerate(spans_out):
        if i == i_target:
            continue
        ranges = s.get("ranges") or []
        if not ranges:
            continue
        # Only consider successfully resolved spans as anchors.
        dist = abs(int(s["start"]) - t_start)
        score = float(s.get("score") or 0.0)
        nconf = float(s.get("dsl_confidence") or 0.0)
        candidates.append((i, dist, score, nconf))

    if not candidates:
        return None

    left = [c for c in candidates if int(spans_out[c[0]]["start"]) < t_start]
    right = [c for c in candidates if int(spans_out[c[0]]["start"]) > t_start]

    def key_fn(t: tuple[int, int, float, float]):
        _, dist, score, nconf = t
        return (dist, -score, -nconf)

    if left:
        left.sort(key=key_fn)
        return left[0][0]
    right.sort(key=key_fn)
    return right[0][0]


def parse_prompt(
    text: str,
    *,
    refdate: date,
    models: Models,
    policy: Policy,
    extractor_threshold: float = 0.5,
) -> dict[str, Any]:
    """
    End-to-end parsing of a raw prompt.
    Returns JSON-serializable dict with raw offsets and resolved ranges.
    """
    from roc_time_parser.extractor.infer import extract_time_spans
    from roc_time_parser.normalizer.infer import normalize_span

    raw_text = text
    spans = extract_time_spans(
        raw_text,
        model=models.extractor_model,
        tokenizer=models.extractor_tokenizer,
        threshold=extractor_threshold,
    )

    out_spans: list[dict[str, Any]] = []
    span_specs: list[Optional[ValidatedSpec]] = []

    for sp in spans:
        span_warnings: list[str] = []
        dsl, conf = normalize_span(sp.text, refdate=refdate, settings=models.normalizer_settings)

        spec: Optional[ValidatedSpec] = None
        spec_json: Optional[dict[str, Any]] = None
        ranges: list[ResolvedRange] = []

        try:
            spec = parse_dsl(dsl)
            spec_json = to_json(spec)
        except Exception:
            span_warnings.append("INVALID_DSL")
            dsl = "T1: FLAGS=NEEDS_CLARIFICATION"
            conf = 0.0

        if spec is not None:
            try:
                ranges = resolve(spec, refdate=refdate, policy=policy)
            except Exception as e:  # noqa: BLE001
                span_warnings.append(f"RESOLVE_ERROR:{type(e).__name__}")
                ranges = []

        # Attach source_text to resolved ranges
        for r in ranges:
            r.source_text = sp.text

        out_spans.append(
            {
                "start": sp.start,
                "end": sp.end,
                "text": sp.text,
                "score": sp.score,
                "dsl": dsl,
                "dsl_confidence": conf,
                "spec": spec_json,
                "ranges": [_range_to_json(r) for r in ranges],
                "warnings": span_warnings,
            }
        )
        span_specs.append(spec)

    # Anchor pass
    for i, spec in enumerate(span_specs):
        if spec is None or spec.period is None:
            continue
        if not isinstance(spec.period, PeriodAnchor):
            continue
        if out_spans[i].get("ranges"):
            continue

        anchor_idx = _pick_anchor_index(out_spans, i)
        if anchor_idx is None:
            out_spans[i]["warnings"].append("NEEDS_ANCHOR")
            continue

        anchor_range_json = out_spans[anchor_idx]["ranges"][0]
        anchor_range = ResolvedRange(
            start=date.fromisoformat(anchor_range_json["start"]),
            end=date.fromisoformat(anchor_range_json["end"]),
            grain=str(anchor_range_json["grain"]),
            warnings=list(anchor_range_json.get("warnings") or []),
            source_text=str(anchor_range_json.get("source_text") or out_spans[anchor_idx]["text"]),
        )

        resolved = resolve_with_anchor(spec, anchor=anchor_range, refdate=refdate)
        for r in resolved:
            r.source_text = out_spans[i]["text"]
        if resolved:
            out_spans[i]["ranges"] = [_range_to_json(r) for r in resolved]
        else:
            out_spans[i]["warnings"].append("ANCHOR_RESOLVE_FAILED")

    top_warnings: list[str] = []
    if not out_spans:
        top_warnings.append("NO_TIME_FOUND")
    elif len(out_spans) > 1:
        top_warnings.append("MULTIPLE_TIME_SPANS")

    return {
        "reference_date": refdate.isoformat(),
        "spans": out_spans,
        "warnings": top_warnings,
    }

