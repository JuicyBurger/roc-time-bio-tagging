from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import json
import os
from pathlib import Path
from typing import Any, Optional


def _parse_date(s: str) -> date:
    y, m, d = s.split("-")
    return date(int(y), int(m), int(d))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float


def _span_prf(gold: list[tuple[int, int]], pred: list[tuple[int, int]]) -> PRF:
    gold_set = list(gold)
    pred_set = list(pred)
    matched_gold: set[int] = set()
    matched_pred: set[int] = set()

    for i, p in enumerate(pred_set):
        for j, g in enumerate(gold_set):
            if j in matched_gold:
                continue
            if p == g:
                matched_pred.add(i)
                matched_gold.add(j)
                break

    tp = len(matched_pred)
    fp = len(pred_set) - tp
    fn = len(gold_set) - tp
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return PRF(precision=precision, recall=recall, f1=f1)


def main(argv: list[str] | None = None) -> int:
    # Load .env first so EXTRACTOR_MODEL_DIR / NORMALIZER_MODEL_DIR are available for defaults
    from roc_time_parser.config import load_dotenv_into_env

    load_dotenv_into_env()

    ap = argparse.ArgumentParser(prog="evaluate_pipeline.py")
    ap.add_argument("--input", default="data/spans_labeled.jsonl")
    ap.add_argument("--refdate", default=date.today().isoformat())
    ap.add_argument("--stage", choices=["a", "b", "e2e", "all"], default="all")
    ap.add_argument("--threshold", type=float, default=0.5, help="Extractor threshold.")
    ap.add_argument(
        "--extractor-dir",
        default=os.environ.get("EXTRACTOR_MODEL_DIR"),
        help="Extractor model directory (default: EXTRACTOR_MODEL_DIR from env or .env).",
    )
    ap.add_argument(
        "--normalizer-dir",
        default=os.environ.get("NORMALIZER_MODEL_DIR"),
        help="Normalizer model directory (default: NORMALIZER_MODEL_DIR from env or .env).",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="artifacts/eval_results.jsonl")
    args = ap.parse_args(argv)

    inp = Path(args.input)
    rows = _read_jsonl(inp)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    ref = _parse_date(args.refdate)

    results_rows: list[dict[str, Any]] = []

    # Lazy imports to avoid heavy deps when not needed.
    do_a = args.stage in {"a", "all"}
    do_b = args.stage in {"b", "all"}
    do_e2e = args.stage in {"e2e", "all"}

    extractor = None
    tokenizer = None
    if do_a or do_e2e:
        try:
            from roc_time_parser.extractor.model import load_extractor

            extractor, tokenizer = load_extractor(model_dir=args.extractor_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] extractor not available: {e}")
            extractor, tokenizer = None, None
            if args.stage == "a":
                return 2

    settings = None
    normalizer_model = None
    normalizer_tokenizer = None
    if do_b or do_e2e:
        try:
            from roc_time_parser.normalizer.model import load_normalizer

            normalizer_model, normalizer_tokenizer = load_normalizer(model_dir=args.normalizer_dir)
        except FileNotFoundError:
            normalizer_model, normalizer_tokenizer = None, None
            try:
                from roc_time_parser.config import load_settings

                settings = load_settings()
            except Exception as e:  # noqa: BLE001
                print(f"[warn] normalizer model/settings not available: {e}")
                settings = None
                if args.stage == "b":
                    return 2
        except Exception as e:  # noqa: BLE001
            print(f"[warn] normalizer model not available: {e}")
            normalizer_model, normalizer_tokenizer = None, None
            try:
                from roc_time_parser.config import load_settings

                settings = load_settings()
            except Exception as e2:  # noqa: BLE001
                print(f"[warn] normalizer settings not available: {e2}")
                settings = None
            if args.stage == "b":
                return 2

    # Stage A metrics accumulator
    a_tp = a_fp = a_fn = 0

    # Stage B counters
    b_total = b_needs_clar = b_needs_anchor = 0

    # End-to-end counters
    e_total = e_no_time = e_multi = 0

    for r in rows:
        rid = r.get("id")
        text = str(r.get("text") or "")
        gold_spans = [(int(s["start"]), int(s["end"])) for s in (r.get("spans") or [])]

        row_out: dict[str, Any] = {"id": rid, "text": text, "gold_spans": gold_spans}

        # Stage A evaluation (exact span match)
        pred_spans: list[tuple[int, int]] = []
        if do_a and extractor is not None and tokenizer is not None:
            from roc_time_parser.extractor.infer import extract_time_spans

            preds = extract_time_spans(text, model=extractor, tokenizer=tokenizer, threshold=args.threshold)
            pred_spans = [(p.start, p.end) for p in preds]
            prf = _span_prf(gold_spans, pred_spans)

            # accumulate TP/FP/FN using counts
            # recompute TP/FP/FN from PRF isn't stable, so count directly via greedy match
            # (reuse _span_prf matching logic by direct matching again)
            matched = set()
            matched_gold = set()
            for i, p in enumerate(pred_spans):
                for j, g in enumerate(gold_spans):
                    if j in matched_gold:
                        continue
                    if p == g:
                        matched.add(i)
                        matched_gold.add(j)
                        break
            tp = len(matched)
            fp = len(pred_spans) - tp
            fn = len(gold_spans) - tp
            a_tp += tp
            a_fp += fp
            a_fn += fn

            row_out["pred_spans"] = pred_spans
            row_out["stage_a"] = {"precision": prf.precision, "recall": prf.recall, "f1": prf.f1}

        # Stage B evaluation (using gold spans)
        if do_b and (settings is not None or normalizer_model is not None):
            from roc_time_parser.normalizer.infer import normalize_span
            from roc_time_parser.schema import parse_dsl

            b_rows: list[dict[str, Any]] = []
            for (s0, s1) in gold_spans:
                span_text = text[s0:s1]
                dsl, conf = normalize_span(
                    span_text,
                    refdate=ref,
                    settings=settings,
                    normalizer_model=normalizer_model,
                    normalizer_tokenizer=normalizer_tokenizer,
                )
                spec = parse_dsl(dsl)
                flags = set(spec.flags)
                b_total += 1
                if "NEEDS_CLARIFICATION" in flags:
                    b_needs_clar += 1
                if "NEEDS_ANCHOR" in flags:
                    b_needs_anchor += 1
                b_rows.append({"span": span_text, "dsl": dsl, "confidence": conf, "flags": list(flags)})
            row_out["stage_b"] = b_rows

        # End-to-end evaluation (smoke)
        if do_e2e and extractor is not None and tokenizer is not None and (settings is not None or normalizer_model is not None):
            from roc_time_parser.pipeline import Models, parse_prompt
            from roc_time_parser.policy import Policy

            e_total += 1
            out = parse_prompt(
                text,
                refdate=ref,
                models=Models(
                    extractor_model=extractor,
                    extractor_tokenizer=tokenizer,
                    normalizer_settings=settings,
                    normalizer_model=normalizer_model,
                    normalizer_tokenizer=normalizer_tokenizer,
                ),
                policy=Policy(),
                extractor_threshold=args.threshold,
            )
            warns = set(out.get("warnings") or [])
            if "NO_TIME_FOUND" in warns:
                e_no_time += 1
            if "MULTIPLE_TIME_SPANS" in warns:
                e_multi += 1
            row_out["e2e"] = out

        results_rows.append(row_out)

    # Summaries
    summary: dict[str, Any] = {"input": str(inp), "n": len(rows), "refdate": ref.isoformat(), "stage": args.stage}

    if do_a and extractor is not None and tokenizer is not None:
        prec = a_tp / (a_tp + a_fp) if (a_tp + a_fp) else 1.0
        rec = a_tp / (a_tp + a_fn) if (a_tp + a_fn) else 1.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        summary["stage_a_span_exact"] = {
            "tp": a_tp,
            "fp": a_fp,
            "fn": a_fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

    if do_b and settings is not None:
        summary["stage_b"] = {
            "n_spans": b_total,
            "needs_clarification_rate": (b_needs_clar / b_total) if b_total else 0.0,
            "needs_anchor_rate": (b_needs_anchor / b_total) if b_total else 0.0,
        }

    if do_e2e and extractor is not None and tokenizer is not None and settings is not None:
        summary["e2e"] = {
            "n_prompts": e_total,
            "no_time_rate": (e_no_time / e_total) if e_total else 0.0,
            "multi_span_rate": (e_multi / e_total) if e_total else 0.0,
        }

    out_path = Path(args.out)
    _write_jsonl(out_path, [{"_summary": summary}] + results_rows)
    print(json.dumps(summary, ensure_ascii=False))
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

