# ROC Time Parser (Python)

End-to-end ROC/AD time parsing pipeline for noisy (OCR-like) prompts.

## What it does
- **Stage A (Extractor)**: BIO/NER token classifier extracts 1+ time spans from raw text and returns **raw offsets**.
- **Stage B (Normalizer)**: normalizes each extracted span into **DSL_TIME_V1** (single-line `T1: ...`).
- **Resolver**: deterministic conversion from DSL to one-or-more `[start, end)` date ranges (end-exclusive).

## Environment
Create `.env` at repo root:
```
MODEL_NAME=gemma3:12b
OLLAMA_HOST=http://<host>:11434
OLLAMA_URL=http://<host>:11434/api/generate
```

## Install (venv recommended)
```bash
pip install -U -e .
```

## Train extractor (Stage A)
Split weak-labeled spans into train/dev:
```bash
python scripts/split_data.py --input data/spans_labeled.jsonl --train-out data/extractor_train.jsonl --dev-out data/extractor_dev.jsonl --dev-ratio 0.1
```

Train and save to `artifacts/extractor`:
```bash
python -m roc_time_parser.extractor.train --train data/extractor_train.jsonl --dev data/extractor_dev.jsonl --output-dir artifacts/extractor --epochs 3 --batch-size 8
```

## Quickstart (CLI)
```bash
roc-time-parser parse --text "植保事業群各事業部114年第一季年終獎金" --refdate 2026-02-04
```

## Evaluate (smoke + metrics)
```bash
python scripts/evaluate_pipeline.py --input data/spans_labeled.jsonl --refdate 2026-02-04 --stage a --threshold 0.5
```

## Data
Expected JSONL formats live under `data/`:
- `raw_prompts.jsonl`: `{id,text}`
- `spans_labeled.jsonl`: `{id,text,spans:[{start,end,label:"TIME"}], ...}`

## Notes
- Offsets in datasets and pipeline output are always **relative to the original raw prompt**.
- Any preprocessing used before extraction must be **offset-preserving** (no length-changing operations).
