from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from roc_time_parser.policy import Policy
from roc_time_parser.schema import (
    PeriodAnchor,
    PeriodHalf,
    PeriodMonth,
    PeriodMonthRange,
    PeriodQ1Q3,
    PeriodQuarter,
    PeriodRel,
    PeriodRolling,
    PeriodSpec,
    PeriodToDate,
    PeriodYear,
    ValidatedSpec,
    YearAD,
    YearRANGE,
    YearREL,
    YearROC,
    YearSpec,
)


@dataclass
class ResolvedRange:
    start: date
    end: date  # end-exclusive
    grain: str
    warnings: list[str]
    source_text: str = ""


def roc_to_ad_year(roc: int) -> int:
    return roc + 1911


def _end_exclusive(refdate: date, policy: Policy) -> date:
    if policy.rolling_end == "refdate":
        return refdate
    return refdate + timedelta(days=1)


def _first_day_of_month(y: int, m: int) -> date:
    return date(y, m, 1)


def _first_day_next_month(y: int, m: int) -> date:
    if m == 12:
        return date(y + 1, 1, 1)
    return date(y, m + 1, 1)


def _add_months(y: int, m: int, delta: int) -> tuple[int, int]:
    """Add delta months to (y,m) returning (new_y,new_m)."""
    idx = (y * 12 + (m - 1)) + delta
    ny = idx // 12
    nm = (idx % 12) + 1
    return ny, nm


def _quarter_start_month(m: int) -> int:
    return ((m - 1) // 3) * 3 + 1


def _start_of_quarter(d: date) -> date:
    sm = _quarter_start_month(d.month)
    return date(d.year, sm, 1)


def _start_of_next_quarter(d: date) -> date:
    sm = _quarter_start_month(d.month)
    nm = sm + 3
    if nm <= 12:
        return date(d.year, nm, 1)
    return date(d.year + 1, nm - 12, 1)


def _start_of_half(d: date) -> date:
    return date(d.year, 1, 1) if d.month <= 6 else date(d.year, 7, 1)


def _start_of_next_half(d: date) -> date:
    if d.month <= 6:
        return date(d.year, 7, 1)
    return date(d.year + 1, 1, 1)


def _shift_year_safe(d: date, years: int) -> date:
    try:
        return date(d.year + years, d.month, d.day)
    except ValueError:
        # Leap day handling (rare for our month-boundary ranges but keep safe).
        if d.month == 2 and d.day == 29:
            return date(d.year + years, 2, 28)
        raise


def _year_candidates(year: YearSpec, refdate: date) -> list[int]:
    if isinstance(year, YearAD):
        return [year.year]
    if isinstance(year, YearROC):
        return [roc_to_ad_year(year.roc)]
    if isinstance(year, YearREL):
        if year.rel == "this_year":
            return [refdate.year]
        if year.rel == "last_year":
            return [refdate.year - 1]
        if year.rel == "two_years_ago":
            return [refdate.year - 2]
        if year.rel == "next_year":
            return [refdate.year + 1]
        if year.rel == "same_period_last_year":
            # Only meaningful when PERIOD=ANCHOR, but keep a deterministic fallback.
            return [refdate.year - 1]
        raise ValueError(f"Unknown REL year: {year.rel}")
    if isinstance(year, YearRANGE):
        return list(range(year.start, year.end + 1))
    raise TypeError(f"Unknown YearSpec: {type(year)}")


def _resolve_fixed_period_in_year(period: PeriodSpec, year: int) -> tuple[date, date, str]:
    """
    Resolve fixed periods that are defined within an explicit calendar year.
    Returns (start,end,grain).
    """
    if isinstance(period, PeriodYear):
        return date(year, 1, 1), date(year + 1, 1, 1), "year"

    if isinstance(period, PeriodQuarter):
        start_month = 1 + (period.q - 1) * 3
        start = date(year, start_month, 1)
        end_month = start_month + 3
        if end_month <= 12:
            end = date(year, end_month, 1)
        else:
            end = date(year + 1, end_month - 12, 1)
        return start, end, "quarter"

    if isinstance(period, PeriodHalf):
        if period.h == 1:
            return date(year, 1, 1), date(year, 7, 1), "half"
        return date(year, 7, 1), date(year + 1, 1, 1), "half"

    if isinstance(period, PeriodQ1Q3):
        return date(year, 1, 1), date(year, 10, 1), "quarter_range"

    if isinstance(period, PeriodMonth):
        start = _first_day_of_month(year, period.m)
        end = _first_day_next_month(year, period.m)
        return start, end, "month"

    if isinstance(period, PeriodMonthRange):
        start = _first_day_of_month(year, period.m1)
        end = _first_day_next_month(year, period.m2)
        return start, end, "month_range"

    raise ValueError(f"Period is not a fixed-in-year period: {period}")


def _resolve_period_relative(period: PeriodRel, refdate: date) -> tuple[date, date, str]:
    k = period.kind
    if k == "this_month":
        start = date(refdate.year, refdate.month, 1)
        end = _first_day_next_month(refdate.year, refdate.month)
        return start, end, "month"
    if k == "last_month":
        py, pm = _add_months(refdate.year, refdate.month, -1)
        start = date(py, pm, 1)
        end = _first_day_next_month(py, pm)
        return start, end, "month"
    if k == "next_month":
        ny, nm = _add_months(refdate.year, refdate.month, 1)
        start = date(ny, nm, 1)
        end = _first_day_next_month(ny, nm)
        return start, end, "month"

    if k == "this_quarter":
        start = _start_of_quarter(refdate)
        end = _start_of_next_quarter(refdate)
        return start, end, "quarter"
    if k == "last_quarter":
        # Go to first day of this quarter, then subtract 1 day to land in last quarter.
        this_q_start = _start_of_quarter(refdate)
        d = this_q_start - timedelta(days=1)
        start = _start_of_quarter(d)
        end = _start_of_next_quarter(d)
        return start, end, "quarter"
    if k == "next_quarter":
        next_q_start = _start_of_next_quarter(refdate)
        start = next_q_start
        end = _start_of_next_quarter(next_q_start)
        return start, end, "quarter"

    if k == "this_half":
        start = _start_of_half(refdate)
        end = _start_of_next_half(refdate)
        return start, end, "half"
    if k == "last_half":
        this_h_start = _start_of_half(refdate)
        d = this_h_start - timedelta(days=1)
        start = _start_of_half(d)
        end = _start_of_next_half(d)
        return start, end, "half"
    if k == "next_half":
        next_h_start = _start_of_next_half(refdate)
        start = next_h_start
        end = _start_of_next_half(next_h_start)
        return start, end, "half"

    raise ValueError(f"Unknown relative period: {k}")


def _resolve_period_to_date(period: PeriodToDate, refdate: date, policy: Policy) -> tuple[date, date, str]:
    end = _end_exclusive(refdate, policy)
    if period.kind == "YTD":
        return date(refdate.year, 1, 1), end, "ytd"
    if period.kind == "QTD":
        return _start_of_quarter(refdate), end, "qtd"
    if period.kind == "HTD":
        return _start_of_half(refdate), end, "htd"
    if period.kind == "MTD":
        return date(refdate.year, refdate.month, 1), end, "mtd"
    raise ValueError(f"Unknown to-date period: {period.kind}")


def _resolve_period_rolling(period: PeriodRolling, refdate: date, policy: Policy) -> tuple[date, date, str]:
    end = _end_exclusive(refdate, policy)
    if period.unit == "months":
        if period.n == 6 and policy.recent_half_year_mode == "last_full_months":
            # End at the first day of current month (exclude partial current month).
            end2 = date(refdate.year, refdate.month, 1)
            sy, sm = _add_months(end2.year, end2.month, -period.n)
            start = date(sy, sm, 1)
            return start, end2, "rolling_months"

        # Start at the first day of the month that is (n-1) months before refdate's month.
        sy, sm = _add_months(refdate.year, refdate.month, -(period.n - 1))
        start = date(sy, sm, 1)
        return start, end, "rolling_months"

    if period.unit == "quarters":
        # Quarter-boundary based: start at start of quarter (n-1 quarters before current quarter).
        this_q_start = _start_of_quarter(refdate)
        # Move back (n-1) quarters by subtracting 3*(n-1) months from quarter start.
        back_months = 3 * (period.n - 1)
        sy, sm = _add_months(this_q_start.year, this_q_start.month, -back_months)
        start = date(sy, sm, 1)
        return start, end, "rolling_quarters"

    if period.unit == "years":
        # Year-boundary based: start at Jan1 of (n-1) years before current year.
        start = date(refdate.year - (period.n - 1), 1, 1)
        return start, end, "rolling_years"

    raise ValueError(f"Unknown rolling unit: {period.unit}")


def resolve(spec: ValidatedSpec, refdate: date, policy: Policy) -> list[ResolvedRange]:
    """
    Deterministic resolver.
    Returns 0..N ranges. For anchor-dependent specs, returns [] with warnings.
    """
    warnings: list[str] = []

    if spec.period is None:
        return []

    # Anchor-dependent: do not resolve here.
    if isinstance(spec.period, PeriodAnchor):
        warnings.append("NEEDS_ANCHOR")
        return []

    # Relative periods ignore YEAR field (YEAR treated as hint only).
    if isinstance(spec.period, PeriodRel):
        start, end, grain = _resolve_period_relative(spec.period, refdate)
        return [ResolvedRange(start=start, end=end, grain=grain, warnings=warnings)]

    # To-date ignores YEAR field.
    if isinstance(spec.period, PeriodToDate):
        start, end, grain = _resolve_period_to_date(spec.period, refdate, policy)
        return [ResolvedRange(start=start, end=end, grain=grain, warnings=warnings)]

    # Rolling ignores YEAR field.
    if isinstance(spec.period, PeriodRolling):
        start, end, grain = _resolve_period_rolling(spec.period, refdate, policy)
        return [ResolvedRange(start=start, end=end, grain=grain, warnings=warnings)]

    # Fixed-in-year periods require a YEAR.
    if spec.year is None:
        warnings.append("MISSING_YEAR")
        return []

    years = _year_candidates(spec.year, refdate)

    ranges: list[ResolvedRange] = []
    for y in years:
        start, end, grain = _resolve_fixed_period_in_year(spec.period, y)
        ranges.append(ResolvedRange(start=start, end=end, grain=grain, warnings=list(warnings)))
    return ranges


def resolve_with_anchor(spec: ValidatedSpec, anchor: ResolvedRange, refdate: date) -> list[ResolvedRange]:
    """
    Resolve a PERIOD=ANCHOR spec given an anchor range (already resolved).

    Supported:
    - YEAR=REL(same_period_last_year): shift anchor by -1 year
    - YEAR=AD/ROC/REL(this/last/...): map anchor grain into the target year by shifting year
      while keeping month/day boundaries.
    """
    if not isinstance(spec.period, PeriodAnchor) or spec.year is None:
        return []

    warnings: list[str] = []
    y = spec.year
    if isinstance(y, YearREL) and y.rel == "same_period_last_year":
        start = _shift_year_safe(anchor.start, -1)
        end = _shift_year_safe(anchor.end, -1)
        return [ResolvedRange(start=start, end=end, grain=anchor.grain, warnings=warnings)]

    # For explicit target year, shift anchor so its start.year becomes target_year.
    # This assumes anchor is within one calendar year boundary (true for our month/quarter/half ranges).
    if isinstance(y, YearAD):
        target_year = y.year
    elif isinstance(y, YearROC):
        target_year = roc_to_ad_year(y.roc)
    elif isinstance(y, YearREL):
        # map to a concrete year (relative to refdate)
        if y.rel == "this_year":
            target_year = refdate.year
        elif y.rel == "last_year":
            target_year = refdate.year - 1
        elif y.rel == "two_years_ago":
            target_year = refdate.year - 2
        elif y.rel == "next_year":
            target_year = refdate.year + 1
        else:
            # already handled same_period_last_year above
            return []
    elif isinstance(y, YearRANGE):
        # For range + anchor, pick first year deterministically (caller may do something smarter).
        target_year = y.start
        warnings.append("RANGE_WITH_ANCHOR_PICKED_START")
    else:
        return []

    delta = target_year - anchor.start.year
    start = _shift_year_safe(anchor.start, delta)
    end = _shift_year_safe(anchor.end, delta)
    return [ResolvedRange(start=start, end=end, grain=anchor.grain, warnings=warnings)]

