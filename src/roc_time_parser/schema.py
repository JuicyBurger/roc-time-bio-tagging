from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final, Literal, Optional


# -----------------------------
# Spec structures
# -----------------------------

RelYear = Literal[
    "this_year",
    "last_year",
    "two_years_ago",
    "next_year",
    "same_period_last_year",
]

RelPeriod = Literal[
    "this_quarter",
    "last_quarter",
    "next_quarter",
    "this_half",
    "last_half",
    "next_half",
    "this_month",
    "last_month",
    "next_month",
]

ToDatePeriod = Literal["YTD", "QTD", "HTD", "MTD"]

RollingUnit = Literal["months", "quarters", "years"]


@dataclass(frozen=True)
class YearAD:
    year: int


@dataclass(frozen=True)
class YearROC:
    roc: int
    assumed: bool


@dataclass(frozen=True)
class YearREL:
    rel: RelYear


@dataclass(frozen=True)
class YearRANGE:
    start: int
    end: int


YearSpec = YearAD | YearROC | YearREL | YearRANGE


@dataclass(frozen=True)
class PeriodYear:
    pass


@dataclass(frozen=True)
class PeriodQuarter:
    q: int  # 1..4


@dataclass(frozen=True)
class PeriodHalf:
    h: int  # 1..2


@dataclass(frozen=True)
class PeriodQ1Q3:
    pass


@dataclass(frozen=True)
class PeriodMonth:
    m: int  # 1..12


@dataclass(frozen=True)
class PeriodMonthRange:
    m1: int
    m2: int


@dataclass(frozen=True)
class PeriodToDate:
    kind: ToDatePeriod


@dataclass(frozen=True)
class PeriodRel:
    kind: RelPeriod


@dataclass(frozen=True)
class PeriodRolling:
    unit: RollingUnit
    n: int


@dataclass(frozen=True)
class PeriodAnchor:
    pass


PeriodSpec = (
    PeriodYear
    | PeriodQuarter
    | PeriodHalf
    | PeriodQ1Q3
    | PeriodMonth
    | PeriodMonthRange
    | PeriodToDate
    | PeriodRel
    | PeriodRolling
    | PeriodAnchor
)


@dataclass(frozen=True)
class ValidatedSpec:
    year: Optional[YearSpec]
    period: Optional[PeriodSpec]
    flags: tuple[str, ...] = ()
    extra: Optional[str] = None


# -----------------------------
# Parsing
# -----------------------------

_RE_AD: Final[re.Pattern[str]] = re.compile(r"^AD\((\d{4})\)$", re.IGNORECASE)
_RE_ROC: Final[re.Pattern[str]] = re.compile(
    r"^ROC\((\d{1,3}),\s*assumed=(true|false)\)$", re.IGNORECASE
)
_RE_REL: Final[re.Pattern[str]] = re.compile(r"^REL\(([^)]+)\)$", re.IGNORECASE)
_RE_RANGE: Final[re.Pattern[str]] = re.compile(r"^RANGE\((\d{4})-(\d{4})\)$", re.IGNORECASE)

_RE_MONTH: Final[re.Pattern[str]] = re.compile(r"^MONTH\((\d{1,2})\)$", re.IGNORECASE)
_RE_MONTH_RANGE: Final[re.Pattern[str]] = re.compile(
    r"^MONTH_RANGE\((\d{1,2}),\s*(\d{1,2})\)$", re.IGNORECASE
)
_RE_ROLLING: Final[re.Pattern[str]] = re.compile(
    r"^ROLLING\(\s*(months|quarters|years)\s*=\s*(\d{1,3})\s*\)$", re.IGNORECASE
)


_ALLOWED_REL_YEAR: Final[set[str]] = {
    "this_year",
    "last_year",
    "two_years_ago",
    "next_year",
    "same_period_last_year",
}
_ALLOWED_REL_PERIOD: Final[set[str]] = {
    "this_quarter",
    "last_quarter",
    "next_quarter",
    "this_half",
    "last_half",
    "next_half",
    "this_month",
    "last_month",
    "next_month",
}
_ALLOWED_TODATE: Final[set[str]] = {"YTD", "QTD", "HTD", "MTD"}


def _parse_year(value: str) -> YearSpec:
    v = value.strip()
    m = _RE_AD.match(v)
    if m:
        year = int(m.group(1))
        if not (1000 <= year <= 9999):
            raise ValueError("AD year out of range")
        return YearAD(year=year)

    m = _RE_ROC.match(v)
    if m:
        roc = int(m.group(1))
        assumed = m.group(2).lower() == "true"
        if not (1 <= roc <= 999):
            raise ValueError("ROC year out of range")
        return YearROC(roc=roc, assumed=assumed)

    m = _RE_REL.match(v)
    if m:
        rel = m.group(1).strip().lower()
        if rel not in _ALLOWED_REL_YEAR:
            raise ValueError(f"Unknown REL year: {rel}")
        return YearREL(rel=rel)  # type: ignore[arg-type]

    m = _RE_RANGE.match(v)
    if m:
        y1 = int(m.group(1))
        y2 = int(m.group(2))
        if not (1000 <= y1 <= 9999 and 1000 <= y2 <= 9999):
            raise ValueError("RANGE year out of range")
        if y1 > y2:
            raise ValueError("RANGE start > end")
        return YearRANGE(start=y1, end=y2)

    raise ValueError(f"Invalid YEAR: {value}")


def _parse_period(value: str) -> PeriodSpec:
    v = value.strip().upper()
    if v == "YEAR":
        return PeriodYear()
    if v in {"Q1", "Q2", "Q3", "Q4"}:
        return PeriodQuarter(q=int(v[1]))
    if v in {"H1", "H2"}:
        return PeriodHalf(h=int(v[1]))
    if v == "Q1_Q3":
        return PeriodQ1Q3()
    if v in _ALLOWED_TODATE:
        return PeriodToDate(kind=v)  # type: ignore[arg-type]
    if v == "ANCHOR":
        return PeriodAnchor()

    m = _RE_MONTH.match(value.strip())
    if m:
        m1 = int(m.group(1))
        if not (1 <= m1 <= 12):
            raise ValueError("MONTH out of range")
        return PeriodMonth(m=m1)

    m = _RE_MONTH_RANGE.match(value.strip())
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        if not (1 <= a <= 12 and 1 <= b <= 12):
            raise ValueError("MONTH_RANGE out of range")
        if a > b:
            raise ValueError("MONTH_RANGE requires m1<=m2")
        return PeriodMonthRange(m1=a, m2=b)

    # REL(period)
    m = _RE_REL.match(value.strip())
    if m:
        rel = m.group(1).strip().lower()
        if rel not in _ALLOWED_REL_PERIOD:
            raise ValueError(f"Unknown REL period: {rel}")
        return PeriodRel(kind=rel)  # type: ignore[arg-type]

    m = _RE_ROLLING.match(value.strip())
    if m:
        unit = m.group(1).lower()
        n = int(m.group(2))
        if not (1 <= n <= 120):
            raise ValueError("ROLLING n out of range")
        return PeriodRolling(unit=unit, n=n)  # type: ignore[arg-type]

    raise ValueError(f"Invalid PERIOD: {value}")


def parse_dsl(dsl: str) -> ValidatedSpec:
    s = dsl.strip()
    if not s:
        raise ValueError("Empty DSL")
    if "\n" in s or "\r" in s:
        raise ValueError("DSL must be single-line")

    # Accept only per-span single line "T1: ..."
    if not re.match(r"^T1\s*:", s, flags=re.IGNORECASE):
        raise ValueError("DSL must start with 'T1:'")

    # Split "T1:" prefix
    _, rest = s.split(":", 1)
    rest = rest.strip()
    if not rest:
        raise ValueError("Missing spec body after T1:")

    # Split by semicolons
    parts = [p.strip() for p in rest.split(";") if p.strip()]
    kv: dict[str, str] = {}
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Invalid part (expected KEY=VALUE): {p}")
        k, v = p.split("=", 1)
        k = k.strip().upper()
        v = v.strip()
        if k in kv:
            raise ValueError(f"Duplicate key: {k}")
        kv[k] = v

    allowed_keys = {"YEAR", "PERIOD", "FLAGS", "EXTRA"}
    unknown = set(kv) - allowed_keys
    if unknown:
        raise ValueError(f"Unknown keys: {sorted(unknown)}")

    flags: tuple[str, ...] = ()
    if "FLAGS" in kv:
        raw = kv["FLAGS"]
        toks = re.split(r"[|,\s]+", raw.strip())
        flags = tuple(t.upper() for t in toks if t)

    extra = kv.get("EXTRA")

    year: Optional[YearSpec] = None
    period: Optional[PeriodSpec] = None

    if "YEAR" in kv:
        year = _parse_year(kv["YEAR"])
    if "PERIOD" in kv:
        period = _parse_period(kv["PERIOD"])

    # Strictness rules:
    # - If NEEDS_CLARIFICATION: YEAR/PERIOD may be missing.
    if "NEEDS_CLARIFICATION" not in flags:
        if year is None:
            raise ValueError("Missing YEAR")
        if period is None:
            raise ValueError("Missing PERIOD")

    # - If PERIOD=ANCHOR and not clarifying, YEAR must be present.
    if isinstance(period, PeriodAnchor) and "NEEDS_CLARIFICATION" not in flags and year is None:
        raise ValueError("PERIOD=ANCHOR requires YEAR")

    return ValidatedSpec(year=year, period=period, flags=flags, extra=extra)


def to_json(spec: ValidatedSpec) -> dict:
    def year_json(y: Optional[YearSpec]) -> Optional[dict]:
        if y is None:
            return None
        if isinstance(y, YearAD):
            return {"type": "AD", "year": y.year}
        if isinstance(y, YearROC):
            return {"type": "ROC", "roc": y.roc, "assumed": y.assumed}
        if isinstance(y, YearREL):
            return {"type": "REL", "rel": y.rel}
        if isinstance(y, YearRANGE):
            return {"type": "RANGE", "start": y.start, "end": y.end}
        raise TypeError(f"Unknown YearSpec: {type(y)}")

    def period_json(p: Optional[PeriodSpec]) -> Optional[dict]:
        if p is None:
            return None
        if isinstance(p, PeriodYear):
            return {"type": "YEAR"}
        if isinstance(p, PeriodQuarter):
            return {"type": "QUARTER", "q": p.q}
        if isinstance(p, PeriodHalf):
            return {"type": "HALF", "h": p.h}
        if isinstance(p, PeriodQ1Q3):
            return {"type": "Q1_Q3"}
        if isinstance(p, PeriodMonth):
            return {"type": "MONTH", "m": p.m}
        if isinstance(p, PeriodMonthRange):
            return {"type": "MONTH_RANGE", "m1": p.m1, "m2": p.m2}
        if isinstance(p, PeriodToDate):
            return {"type": "TODATE", "kind": p.kind}
        if isinstance(p, PeriodRel):
            return {"type": "REL", "kind": p.kind}
        if isinstance(p, PeriodRolling):
            return {"type": "ROLLING", "unit": p.unit, "n": p.n}
        if isinstance(p, PeriodAnchor):
            return {"type": "ANCHOR"}
        raise TypeError(f"Unknown PeriodSpec: {type(p)}")

    return {
        "year": year_json(spec.year),
        "period": period_json(spec.period),
        "flags": list(spec.flags),
        "extra": spec.extra,
    }

