from datetime import date

from roc_time_parser.policy import Policy
from roc_time_parser.resolver import resolve, resolve_with_anchor, roc_to_ad_year
from roc_time_parser.schema import parse_dsl


def test_roc_to_ad_year() -> None:
    assert roc_to_ad_year(114) == 2025


def test_resolve_ad_q1() -> None:
    spec = parse_dsl("T1: YEAR=AD(2025); PERIOD=Q1")
    r = resolve(spec, refdate=date(2026, 2, 4), policy=Policy())
    assert len(r) == 1
    assert r[0].start == date(2025, 1, 1)
    assert r[0].end == date(2025, 4, 1)


def test_resolve_rel_last_year_h1() -> None:
    spec = parse_dsl("T1: YEAR=REL(last_year); PERIOD=H1")
    r = resolve(spec, refdate=date(2026, 2, 4), policy=Policy())
    assert (r[0].start, r[0].end) == (date(2025, 1, 1), date(2025, 7, 1))


def test_resolve_range_years_expands() -> None:
    spec = parse_dsl("T1: YEAR=RANGE(2022-2024); PERIOD=YEAR")
    r = resolve(spec, refdate=date(2026, 2, 4), policy=Policy())
    assert [(x.start, x.end) for x in r] == [
        (date(2022, 1, 1), date(2023, 1, 1)),
        (date(2023, 1, 1), date(2024, 1, 1)),
        (date(2024, 1, 1), date(2025, 1, 1)),
    ]


def test_resolve_range_q1_expands() -> None:
    spec = parse_dsl("T1: YEAR=RANGE(2022-2024); PERIOD=Q1")
    r = resolve(spec, refdate=date(2026, 2, 4), policy=Policy())
    assert [(x.start, x.end) for x in r] == [
        (date(2022, 1, 1), date(2022, 4, 1)),
        (date(2023, 1, 1), date(2023, 4, 1)),
        (date(2024, 1, 1), date(2024, 4, 1)),
    ]


def test_resolve_rolling_months_month_boundary() -> None:
    spec = parse_dsl("T1: YEAR=REL(this_year); PERIOD=ROLLING(months=3)")
    r = resolve(spec, refdate=date(2026, 2, 4), policy=Policy(rolling_end="tomorrow"))
    assert (r[0].start, r[0].end) == (date(2025, 12, 1), date(2026, 2, 5))


def test_resolve_ytd_end_exclusive() -> None:
    spec = parse_dsl("T1: YEAR=REL(this_year); PERIOD=YTD")
    r = resolve(spec, refdate=date(2026, 2, 4), policy=Policy(rolling_end="refdate"))
    assert (r[0].start, r[0].end) == (date(2026, 1, 1), date(2026, 2, 4))


def test_resolve_rel_last_quarter_cross_year() -> None:
    spec = parse_dsl("T1: YEAR=REL(this_year); PERIOD=REL(last_quarter)")
    r = resolve(spec, refdate=date(2026, 1, 15), policy=Policy())
    assert (r[0].start, r[0].end) == (date(2025, 10, 1), date(2026, 1, 1))


def test_anchor_shift_same_period_last_year() -> None:
    anchor = resolve(parse_dsl("T1: YEAR=AD(2025); PERIOD=Q1"), refdate=date(2026, 2, 4), policy=Policy())[0]
    spec = parse_dsl("T1: YEAR=REL(same_period_last_year); PERIOD=ANCHOR; FLAGS=NEEDS_ANCHOR")
    r = resolve_with_anchor(spec, anchor=anchor, refdate=date(2026, 2, 4))
    assert (r[0].start, r[0].end) == (date(2024, 1, 1), date(2024, 4, 1))


def test_anchor_map_to_explicit_year() -> None:
    anchor = resolve(parse_dsl("T1: YEAR=AD(2025); PERIOD=Q1"), refdate=date(2026, 2, 4), policy=Policy())[0]
    spec = parse_dsl("T1: YEAR=ROC(113,assumed=false); PERIOD=ANCHOR; FLAGS=NEEDS_ANCHOR")
    r = resolve_with_anchor(spec, anchor=anchor, refdate=date(2026, 2, 4))
    assert (r[0].start, r[0].end) == (date(2024, 1, 1), date(2024, 4, 1))

