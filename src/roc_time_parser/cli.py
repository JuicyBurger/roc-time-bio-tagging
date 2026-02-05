from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date
import json
from typing import Any

from roc_time_parser.config import load_settings


def _parse_date(s: str) -> date:
    try:
        y, m, d = s.split("-")
        return date(int(y), int(m), int(d))
    except Exception as e:  # noqa: BLE001
        raise argparse.ArgumentTypeError(f"Invalid date '{s}', expected YYYY-MM-DD") from e


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="roc-time-parser")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract", help="Extract time spans (Stage A).")
    p_extract.add_argument("--text", required=True)
    p_extract.add_argument("--threshold", type=float, default=0.5)

    p_norm = sub.add_parser("normalize", help="Normalize a span to DSL (Stage B).")
    p_norm.add_argument("--span", required=True)
    p_norm.add_argument("--refdate", type=_parse_date, required=True)
    p_norm.add_argument("--normalizer-dir", default=None, help="Seq2seq normalizer model directory.")

    p_parse = sub.add_parser("parse", help="Run full pipeline on a prompt.")
    p_parse.add_argument("--text", required=True)
    p_parse.add_argument("--refdate", type=_parse_date, required=True)
    p_parse.add_argument("--threshold", type=float, default=0.5)
    p_parse.add_argument("--normalizer-dir", default=None, help="Seq2seq normalizer model directory.")

    args = parser.parse_args(argv)

    # Lazy imports so `pip install -e .` works even before models are trained.
    if args.cmd == "extract":
        from roc_time_parser.extractor.model import load_extractor
        from roc_time_parser.extractor.infer import extract_time_spans

        model, tokenizer = load_extractor()
        spans = extract_time_spans(args.text, model=model, tokenizer=tokenizer, threshold=float(args.threshold))
        for s in spans:
            print(json.dumps(asdict(s), ensure_ascii=False))
        return 0

    if args.cmd == "normalize":
        from roc_time_parser.normalizer.infer import normalize_span

        if args.normalizer_dir:
            from roc_time_parser.normalizer.model import load_normalizer

            n_model, n_tok = load_normalizer(model_dir=args.normalizer_dir)
            dsl, conf = normalize_span(
                args.span,
                refdate=args.refdate,
                normalizer_model=n_model,
                normalizer_tokenizer=n_tok,
            )
        else:
            settings = load_settings()
            dsl, conf = normalize_span(args.span, refdate=args.refdate, settings=settings)
        print(json.dumps({"dsl": dsl, "confidence": conf}, ensure_ascii=False))
        return 0

    if args.cmd == "parse":
        from roc_time_parser.pipeline import parse_prompt
        from roc_time_parser.pipeline import Models
        from roc_time_parser.policy import Policy
        from roc_time_parser.extractor.model import load_extractor

        settings = None
        normalizer_model = None
        normalizer_tokenizer = None
        if args.normalizer_dir:
            from roc_time_parser.normalizer.model import load_normalizer

            normalizer_model, normalizer_tokenizer = load_normalizer(model_dir=args.normalizer_dir)
        else:
            settings = load_settings()
        ex_model, ex_tok = load_extractor()
        models = Models(
            extractor_model=ex_model,
            extractor_tokenizer=ex_tok,
            normalizer_settings=settings,
            normalizer_model=normalizer_model,
            normalizer_tokenizer=normalizer_tokenizer,
        )
        out: dict[str, Any] = parse_prompt(
            args.text,
            refdate=args.refdate,
            models=models,
            policy=Policy(),
            extractor_threshold=float(args.threshold),
        )
        print(json.dumps(out, ensure_ascii=False))
        return 0

    parser.error("Unknown command")
    return 2

