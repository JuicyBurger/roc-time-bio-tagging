from datetime import date

import pytest

from roc_time_parser.config import Settings
from roc_time_parser.normalizer.infer import normalize_span


def test_normalize_span_same_period_last_year_needs_anchor_no_call() -> None:
    settings = Settings(model_name="x", ollama_host="", ollama_url="http://invalid")

    def boom(*args, **kwargs):  # noqa: ANN001, D401
        raise AssertionError("query_fn should not be called for anchor-only rule")

    dsl, conf = normalize_span("去年同期", refdate=date(2026, 2, 4), settings=settings, query_fn=boom)
    assert "PERIOD=ANCHOR" in dsl
    assert "REL(same_period_last_year)" in dsl
    assert conf > 0


def test_normalize_span_with_period_calls_model() -> None:
    settings = Settings(model_name="x", ollama_host="", ollama_url="http://invalid")

    class R:
        response = "T1: YEAR=AD(2025); PERIOD=Q1"

    dsl, conf = normalize_span("2025年Q1", refdate=date(2026, 2, 4), settings=settings, query_fn=lambda *a, **k: R())
    assert dsl.startswith("T1:")
    assert conf >= 0.7


def test_normalize_span_invalid_output_falls_back() -> None:
    settings = Settings(model_name="x", ollama_host="", ollama_url="http://invalid")

    class R:
        response = "not dsl"

    dsl, conf = normalize_span("2025年Q1", refdate=date(2026, 2, 4), settings=settings, query_fn=lambda *a, **k: R())
    assert dsl == "T1: FLAGS=NEEDS_CLARIFICATION"
    assert conf == 0.0

