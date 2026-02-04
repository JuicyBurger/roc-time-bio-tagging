from __future__ import annotations

from typing import Final


LABELS: Final[list[str]] = ["O", "B-TIME", "I-TIME"]
LABEL2ID: Final[dict[str, int]] = {l: i for i, l in enumerate(LABELS)}
ID2LABEL: Final[dict[int, str]] = {i: l for i, l in enumerate(LABELS)}

