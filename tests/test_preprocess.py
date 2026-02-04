from roc_time_parser.preprocess import normalize_ocr, normalize_unicode, preprocess


def test_offset_preserving_keeps_length() -> None:
    raw = "  植保事業群２０２５年Ｑ１～Ｑ３年終獎金  "
    out = preprocess(raw, mode="offset_preserving")
    assert len(out) == len(raw)


def test_compact_may_change_length() -> None:
    raw = "  A   B  "
    out = preprocess(raw, mode="compact")
    assert out == "A B"


def test_fullwidth_to_halfwidth() -> None:
    assert normalize_unicode("Ｑ１") == "Q1"
    assert normalize_unicode("１２３４") == "1234"


def test_dash_normalization() -> None:
    assert normalize_unicode("Q1～Q3") == "Q1-Q3"


def test_ocr_digit_context() -> None:
    assert normalize_ocr("2O25") == "2025"
    assert normalize_ocr("20I5") == "2015"


def test_ocr_qh_one() -> None:
    assert normalize_ocr("Ql") == "Q1"
    assert normalize_ocr("HI") == "H1"


def test_chinese_quarter_numeral() -> None:
    assert normalize_ocr("第ㄧ季".replace("ㄧ", "一")) == "第1季"

