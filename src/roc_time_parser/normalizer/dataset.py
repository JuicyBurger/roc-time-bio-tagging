from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING
import json

from roc_time_parser.preprocess import preprocess


if TYPE_CHECKING:  # pragma: no cover
    from datasets import Dataset


def make_input(span: str, refdate: str) -> str:
    """
    Build a deterministic input string for the normalizer model.

    Format:
    REFDATE=YYYY-MM-DD
    SPAN=...
    """
    span_compact = preprocess(span, mode="compact")
    return f"REFDATE={refdate}\nSPAN={span_compact}"


def load_normalizer_jsonl(path: str | Path) -> Dataset:
    """
    Load JSONL with rows:
      {"span":"...", "refdate":"YYYY-MM-DD", "target":"T1: ..."}
    """
    from datasets import Dataset  # noqa: WPS433 (runtime import)

    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def prepare_seq2seq_features(
    examples: dict[str, list[Any]],
    tokenizer,
    *,
    max_source_length: int = 128,
    max_target_length: int = 128,
) -> dict[str, Any]:
    """
    HF datasets.map batched function for seq2seq normalizer.
    """
    spans: list[str] = examples["span"]
    refdates: list[str] = examples["refdate"]
    targets: list[str] = examples["target"]

    inputs = [make_input(s, r) for s, r in zip(spans, refdates, strict=True)]

    # Fixed-length padding so every batch has same dimensions (avoids CUDA cuBLAS issues)
    model_inputs = tokenizer(
        inputs,
        max_length=max_source_length,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        text_target=targets,
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
    )
    # Loss ignores positions with -100; mask padded label positions
    pad_id = tokenizer.pad_token_id
    label_ids = labels["input_ids"]
    if pad_id is not None:
        label_ids = [
            [-100 if tid == pad_id else tid for tid in row]
            for row in label_ids
        ]
    model_inputs["labels"] = label_ids
    return model_inputs

