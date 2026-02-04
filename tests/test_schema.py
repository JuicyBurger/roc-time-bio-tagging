import pytest

from roc_time_parser.schema import (
    PeriodAnchor,
    PeriodQuarter,
    ValidatedSpec,
    YearROC,
    parse_dsl,
    to_json,
)


def test_parse_basic_roc_q1() -> None:
    spec = parse_dsl("T1: YEAR=ROC(114,assumed=true); PERIOD=Q1")
    assert isinstance(spec.year, YearROC)
    assert spec.year.roc == 114
    assert spec.year.assumed is True
    assert isinstance(spec.period, PeriodQuarter)
    assert spec.period.q == 1


def test_parse_flags_only_allowed_for_needs_clarification() -> None:
    spec = parse_dsl("T1: FLAGS=NEEDS_CLARIFICATION")
    assert isinstance(spec, ValidatedSpec)
    assert spec.year is None
    assert spec.period is None
    assert "NEEDS_CLARIFICATION" in spec.flags


def test_parse_anchor() -> None:
    spec = parse_dsl("T1: YEAR=REL(same_period_last_year); PERIOD=ANCHOR; FLAGS=NEEDS_ANCHOR")
    assert isinstance(spec.period, PeriodAnchor)


def test_invalid_missing_t1() -> None:
    with pytest.raises(ValueError):
        parse_dsl("YEAR=AD(2025); PERIOD=Q1")


def test_invalid_multiline() -> None:
    with pytest.raises(ValueError):
        parse_dsl("T1: YEAR=AD(2025); PERIOD=Q1\nEXTRA=oops")


def test_to_json_roundtrip_shapes() -> None:
    spec = parse_dsl("T1: YEAR=ROC(114,assumed=false); PERIOD=Q4; FLAGS=A|B")
    j = to_json(spec)
    assert j["year"]["type"] == "ROC"
    assert j["period"]["type"] == "QUARTER"
    assert set(j["flags"]) == {"A", "B"}

