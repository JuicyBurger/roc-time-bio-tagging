from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Policy:
    # Don’t guess ROC unless explicitly marked (safe default).
    assume_bare_roc_year: bool = False
    # End-exclusive choice for rolling/to-date windows.
    rolling_end: Literal["refdate", "tomorrow"] = "tomorrow"
    # Only applies to the phrase “近半年” if we map it to rolling months.
    recent_half_year_mode: Literal["rolling", "last_full_months"] = "rolling"

