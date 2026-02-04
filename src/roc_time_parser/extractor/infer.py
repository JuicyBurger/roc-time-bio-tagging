from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F

from roc_time_parser.preprocess import preprocess


@dataclass
class Span:
    start: int
    end: int
    text: str
    score: float


_MERGE_GAP_ALLOWED: set[str] = {
    "-",
    "–",
    "—",
    "～",
    "－",
    "至",
}


def _merge_adjacent_spans(
    spans: list[Span],
    raw_text: str,
    *,
    merge_score_threshold: float,
) -> list[Span]:
    if not spans:
        return spans
    merged: list[Span] = [spans[0]]
    for sp in spans[1:]:
        prev = merged[-1]
        if sp.start <= prev.end:
            # overlap; merge defensively
            new_start = min(prev.start, sp.start)
            new_end = max(prev.end, sp.end)
            txt = raw_text[new_start:new_end]
            score = (prev.score + sp.score) / 2.0
            merged[-1] = Span(start=new_start, end=new_end, text=txt, score=score)
            continue

        gap = raw_text[prev.end : sp.start]
        if (
            0 < len(gap) <= 2
            and all(ch in _MERGE_GAP_ALLOWED for ch in gap)
            and prev.score >= merge_score_threshold
            and sp.score >= merge_score_threshold
        ):
            new_end = sp.end
            txt = raw_text[prev.start:new_end]
            score = (prev.score + sp.score) / 2.0
            merged[-1] = Span(start=prev.start, end=new_end, text=txt, score=score)
        else:
            merged.append(sp)
    return merged


def extract_time_spans(
    text: str,
    model,
    tokenizer,
    threshold: float = 0.5,
    merge_score_threshold: float = 0.8,
) -> list[Span]:
    """
    Extract TIME spans from raw text and return raw offsets (end-exclusive).
    """
    raw_text = text
    model_text = preprocess(raw_text, mode="offset_preserving")
    if len(model_text) != len(raw_text):
        raise ValueError("offset_preserving preprocess changed length; offsets would break")

    enc = tokenizer(
        model_text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
    )
    offsets = enc.pop("offset_mapping")[0].tolist()

    model.eval()
    with torch.no_grad():
        out = model(**enc)
        probs = F.softmax(out.logits, dim=-1)[0]  # (seq_len, num_labels)
        pred_ids = torch.argmax(probs, dim=-1).tolist()
        pred_scores = torch.max(probs, dim=-1).values.tolist()

    id2label = getattr(model.config, "id2label", None) or {0: "O", 1: "B-TIME", 2: "I-TIME"}

    tokens: list[tuple[int, int, str, float]] = []
    for (start, end), lid, score in zip(offsets, pred_ids, pred_scores, strict=True):
        if start == 0 and end == 0:
            continue  # special tokens
        label = str(id2label[int(lid)])
        if label in {"B-TIME", "I-TIME"} and float(score) < threshold:
            label = "O"
        tokens.append((int(start), int(end), label, float(score)))

    # Decode BIO into spans
    spans_raw: list[Span] = []
    cur_start = None
    cur_end = None
    cur_scores: list[float] = []

    def flush():
        nonlocal cur_start, cur_end, cur_scores
        if cur_start is None or cur_end is None or cur_end <= cur_start:
            cur_start, cur_end, cur_scores = None, None, []
            return
        s = int(cur_start)
        e = int(cur_end)
        if 0 <= s < e <= len(raw_text):
            spans_raw.append(Span(start=s, end=e, text=raw_text[s:e], score=sum(cur_scores) / len(cur_scores)))
        cur_start, cur_end, cur_scores = None, None, []

    for start, end, label, score in tokens:
        if label == "B-TIME":
            flush()
            cur_start, cur_end = start, end
            cur_scores = [score]
        elif label == "I-TIME":
            if cur_start is None:
                cur_start, cur_end = start, end
                cur_scores = [score]
            else:
                cur_end = end
                cur_scores.append(score)
        else:
            flush()

    flush()
    spans_raw.sort(key=lambda s: (s.start, s.end))
    return _merge_adjacent_spans(
        spans_raw,
        raw_text,
        merge_score_threshold=merge_score_threshold,
    )

