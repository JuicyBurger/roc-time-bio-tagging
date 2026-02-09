"""
Request/response schemas for the time-parser API.

Granularity values align with the resolver and generate_prompts.py period types:
  month, quarter, half, year, month_range, quarter_range,
  ytd, qtd, mtd, htd, rolling_months, rolling_quarters, rolling_years
"""

from __future__ import annotations

from datetime import date
from pydantic import BaseModel, Field


class TimeRange(BaseModel):
    """A single DB-friendly date range (end-inclusive)."""

    start_time: str = Field(..., description="Start date YYYY-MM-DD (inclusive)")
    end_time: str = Field(..., description="End date YYYY-MM-DD (inclusive)")
    granularity: str = Field(
        ...,
        description="Period type: month, quarter, half, year, month_range, quarter_range, "
        "ytd, qtd, mtd, htd, rolling_months, rolling_quarters, rolling_years",
    )


class TimeExpression(BaseModel):
    """One extracted time expression with its resolved ranges."""

    source_text: str = Field(..., description="Original span text from the prompt")
    ranges: list[TimeRange] = Field(default_factory=list, description="Resolved date ranges")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Normalizer confidence 0–1")
    flags: list[str] = Field(default_factory=list, description="e.g. NEEDS_CLARIFICATION, NEEDS_ANCHOR")


class ParseResponse(BaseModel):
    """Structured result of parsing a prompt."""

    prompt: str = Field(..., description="Echo of the input prompt")
    reference_date: str = Field(..., description="Reference date YYYY-MM-DD used for resolution")
    time_expressions: list[TimeExpression] = Field(
        default_factory=list,
        description="Extracted time expressions and their ranges",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Top-level warnings e.g. NO_TIME_FOUND, MULTIPLE_TIME_SPANS",
    )


class ParseRequest(BaseModel):
    """Request body for POST /parse."""

    prompt: str = Field(..., min_length=1, description="Raw user prompt to parse")
    reference_date: str | None = Field(
        None,
        description="Reference date YYYY-MM-DD; default is today",
    )


def parse_refdate(s: str | None) -> date:
    if not s:
        return date.today()
    return date.fromisoformat(s)
