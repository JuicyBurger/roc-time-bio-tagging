from __future__ import annotations

import sys
from pathlib import Path


# Allow `import roc_time_parser` without requiring `pip install -e .`
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

