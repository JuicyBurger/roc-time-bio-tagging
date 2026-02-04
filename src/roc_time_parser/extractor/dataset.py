from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING
import random

import json

from roc_time_parser.extractor.labels import LABEL2ID
from roc_time_parser.preprocess import preprocess


_FULLWIDTH_DIGITS = str.maketrans("0123456789", "０１２３４５６７８９")
_DIGIT_CONFUSIONS = {
    "0": "O",
    "1": "I",
}
_DASH_CONFUSIONS = {
    "-": "～",
}


def _augment_ocr_noise(text: str, rng: random.Random, prob: float) -> str:
    """
    Offset-preserving noise injection for model input text.
    Only 1-to-1 replacements (length stays identical).
    """
    out = []
    for ch in text:
        r = rng.random()
        if ch.isdigit() and r < prob:
            # Mix full-width digits and simple confusions
            if rng.random() < 0.5:
                out.append(ch.translate(_FULLWIDTH_DIGITS))
            else:
                out.append(_DIGIT_CONFUSIONS.get(ch, ch))
            continue
        if ch in _DASH_CONFUSIONS and r < prob:
            out.append(_DASH_CONFUSIONS[ch])
            continue
        out.append(ch)
    return "".join(out)


if TYPE_CHECKING:  # pragma: no cover
    from datasets import Dataset


def load_spans_jsonl(path: str | Path) -> Dataset:
    # Import lazily to avoid triggering multiprocessing resource tracker on module import
    # (can produce noisy shutdown warnings on some Windows/Python combinations).
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


def char_bio_labels(text: str, spans: Iterable[dict[str, Any]]) -> list[str]:
    labels = ["O"] * len(text)
    for sp in spans:
        start = int(sp["start"])
        end = int(sp["end"])
        lab = sp.get("label", "TIME")
        if lab != "TIME":
            continue
        if not (0 <= start < end <= len(text)):
            raise ValueError(f"Span out of bounds: {start}:{end} for len={len(text)}")
        # Overlap check
        if any(labels[i] != "O" for i in range(start, end)):
            raise ValueError(f"Overlapping TIME span at {start}:{end}")
        labels[start] = "B-TIME"
        for i in range(start + 1, end):
            labels[i] = "I-TIME"
    return labels


def tokenize_and_align_labels(
    examples: dict[str, list[Any]],
    tokenizer,
    *,
    augment_ocr: bool = False,
    augment_prob: float = 0.15,
    augment_seed: int = 42,
) -> dict[str, Any]:
    """
    HF datasets.map batched function.
    - compute char BIO on raw text using raw offsets
    - feed the model `offset_preserving` preprocessed text (length identical)
    - align token labels using token offset_mapping start indices
    """
    texts_raw: list[str] = examples["text"]
    spans_list: list[list[dict[str, Any]]] = examples.get("spans") or [[] for _ in texts_raw]

    texts_model: list[str] = []
    char_labels_list: list[list[str]] = []
    rng = random.Random(augment_seed)
    for raw, spans in zip(texts_raw, spans_list, strict=True):
        model_text = preprocess(raw, mode="offset_preserving")
        if len(model_text) != len(raw):
            raise ValueError("offset_preserving preprocess changed length; offsets would break")
        if augment_ocr:
            model_text = _augment_ocr_noise(model_text, rng=rng, prob=augment_prob)
        texts_model.append(model_text)
        char_labels_list.append(char_bio_labels(raw, spans))

    tokenized = tokenizer(
        texts_model,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        truncation=True,
    )

    labels_out: list[list[int]] = []
    for offsets, special_mask, char_labels in zip(
        tokenized["offset_mapping"], tokenized["special_tokens_mask"], char_labels_list, strict=True
    ):
        label_ids: list[int] = []
        for (start, end), is_special in zip(offsets, special_mask, strict=True):
            if is_special or (start == 0 and end == 0):
                label_ids.append(-100)
                continue
            if start >= len(char_labels):
                label_ids.append(LABEL2ID["O"])
                continue
            label_ids.append(LABEL2ID[char_labels[start]])
        labels_out.append(label_ids)

    # Remove offsets from training features (kept in tokenized for alignment only).
    tokenized.pop("offset_mapping", None)
    tokenized["labels"] = labels_out
    return tokenized

