from __future__ import annotations

from pathlib import Path
import os

from transformers import AutoModelForTokenClassification, AutoTokenizer

from roc_time_parser.extractor.labels import ID2LABEL, LABEL2ID


def _repo_root() -> Path:
    # src/roc_time_parser/extractor/model.py -> repo root is 4 parents up
    return Path(__file__).resolve().parents[3]


def load_extractor(model_dir: str | Path | None = None, base_model: str = "microsoft/Multilingual-MiniLM-L12-H384"):
    """
    Load a trained extractor model from `model_dir`.

    If `model_dir` is None, tries (in order):
    - $EXTRACTOR_MODEL_DIR
    - <repo>/artifacts/extractor
    - <repo>/models/extractor
    """
    if model_dir is None:
        model_dir = os.environ.get("EXTRACTOR_MODEL_DIR")
    candidates: list[Path] = []
    if model_dir:
        candidates.append(Path(model_dir))
    root = _repo_root()
    candidates.extend([
        root / "artifacts" / "extractor_runA",
        root / "artifacts" / "extractor",
        root / "models" / "extractor",
    ])

    for c in candidates:
        if c.exists() and (c / "config.json").exists():
            tok = AutoTokenizer.from_pretrained(str(c), use_fast=True)
            model = AutoModelForTokenClassification.from_pretrained(str(c))
            return model, tok

    raise FileNotFoundError(
        "Extractor model not found. Train it first, e.g.:\n"
        "  python -m roc_time_parser.extractor.train --train data/spans_labeled.jsonl "
        "--dev data/spans_labeled.jsonl --output-dir artifacts/extractor\n"
        "Then re-run `roc-time-parser extract` / `roc-time-parser parse`.\n"
        "Alternatively set EXTRACTOR_MODEL_DIR to a trained model directory."
    )


def init_extractor_from_base(base_model: str = "microsoft/Multilingual-MiniLM-L12-H384"):
    """
    Create an untrained token-classifier head on top of a base model.
    Useful for training script initialization.
    """
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return model, tok

