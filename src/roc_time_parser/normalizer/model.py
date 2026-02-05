from __future__ import annotations

from pathlib import Path
import os

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def _repo_root() -> Path:
    # src/roc_time_parser/normalizer/model.py -> repo root is 4 parents up
    return Path(__file__).resolve().parents[3]


def load_normalizer(
    model_dir: str | Path | None = None,
    base_model: str = "google/mt5-small",
):
    """
    Load a trained seq2seq normalizer model from `model_dir`.

    If `model_dir` is None, tries (in order):
    - $NORMALIZER_MODEL_DIR
    - <repo>/artifacts/normalizer
    - <repo>/models/normalizer
    """
    if model_dir is None:
        model_dir = os.environ.get("NORMALIZER_MODEL_DIR")
    candidates: list[Path] = []
    if model_dir:
        candidates.append(Path(model_dir))
    root = _repo_root()
    candidates.extend([
        root / "artifacts" / "normalizer",
        root / "models" / "normalizer",
    ])

    for c in candidates:
        if c.exists() and (c / "config.json").exists():
            tok = AutoTokenizer.from_pretrained(str(c), use_fast=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(str(c))
            return model, tok

    raise FileNotFoundError(
        "Normalizer model not found. Train it first, e.g.:\n"
        "  python -m roc_time_parser.normalizer.train --train data/normalizer_train.jsonl "
        "--dev data/normalizer_dev.jsonl --output-dir artifacts/normalizer\n"
        "Then re-run `roc-time-parser normalize` / `roc-time-parser parse`.\n"
        "Alternatively set NORMALIZER_MODEL_DIR to a trained model directory."
    )


def init_normalizer_from_base(base_model: str = "google/mt5-small"):
    """
    Create a seq2seq model + tokenizer from a base model name.
    Useful for training initialization.
    """
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    return model, tok

