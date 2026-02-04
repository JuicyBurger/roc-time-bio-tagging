import pytest

from roc_time_parser.extractor.dataset import char_bio_labels
from roc_time_parser.extractor.infer import Span


def test_char_bio_labels_single_span() -> None:
    text = "abc114年Q1xyz"
    spans = [{"start": 3, "end": 9, "label": "TIME"}]
    labels = char_bio_labels(text, spans)
    assert labels[0:3] == ["O", "O", "O"]
    assert labels[3] == "B-TIME"
    assert all(l == "I-TIME" for l in labels[4:9])


def test_char_bio_labels_overlap_raises() -> None:
    text = "abcdef"
    spans = [{"start": 1, "end": 4, "label": "TIME"}, {"start": 3, "end": 5, "label": "TIME"}]
    with pytest.raises(ValueError):
        char_bio_labels(text, spans)


def test_span_dataclass() -> None:
    s = Span(start=1, end=3, text="xx", score=0.9)
    assert s.start == 1 and s.end == 3

