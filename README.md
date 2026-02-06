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

# Optional: default model directories (used when CLI/eval omit --extractor-dir / --normalizer-dir)
EXTRACTOR_MODEL_DIR=artifacts/extractor
NORMALIZER_MODEL_DIR=artifacts/normalizer
```

## Install (venv recommended)
```bash
pip install -U -e .
```

## Generate synthetic training data
Generate synthetic zh_TW finance-style prompts with time span labels:
```bash
python scripts/generate_prompts.py --n 1000 --seed 40 --noise clean --profile broad --out artifacts/gen_broad.jsonl --with_spans
```

Options:
- `--n`: Number of records to generate
- `--seed`: Random seed for reproducibility
- `--noise`: Noise level (`clean`, `mild`, `medium`, `heavy`) - simulates OCR errors
- `--profile`: `pipeline` (DSL-compatible only) or `broad` (includes weeks/days/dates)
- `--out`: Output JSONL file path
- `--with_spans`: Include time span labels

Validate generated spans:
```bash
python scripts/validate_spans_jsonl.py artifacts/gen_broad.jsonl
python scripts/report_span_coverage.py artifacts/gen_broad.jsonl
```

## Train extractor (Stage A)
Split labeled spans into train/dev (use your labeled data or generated synthetic data):
```bash
python scripts/split_data.py --input artifacts/gen_broad.jsonl --train-out data/extractor_train.jsonl --dev-out data/extractor_dev.jsonl --dev-ratio 0.1
```

Or use existing labeled data:
```bash
python scripts/split_data.py --input data/spans_labeled.jsonl --train-out data/extractor_train.jsonl --dev-out data/extractor_dev.jsonl --dev-ratio 0.1
```

Train and save to `artifacts/extractor`:
```bash
python -m roc_time_parser.extractor.train --train data/extractor_train.jsonl --dev data/extractor_dev.jsonl --output-dir artifacts/extractor --epochs 3 --batch-size 8
```

## Train normalizer (Stage B, seq2seq)
Prepare normalizer training pairs:
- `data/normalizer_train.jsonl`, `data/normalizer_dev.jsonl`
- Each row: `{span, refdate, target}` where `target` is a single-line DSL_TIME_V1.

**Generate normalizer data from the prompt generator** (accurate by construction):
```bash
python scripts/generate_prompts.py --n 5000 --seed 42 --with_spans --profile pipeline --out_normalizer data/normalizer_train.jsonl --refdate 2025-02-05
```
Use `--out_normalizer` to write `{span, refdate, target}` JSONL; only DSL-compatible intents are emitted (pipeline profile). Split into train/dev with your preferred script.

Train and save to `artifacts/normalizer`:
```bash
python -m roc_time_parser.normalizer.train --train data/normalizer_train.jsonl --dev data/normalizer_dev.jsonl --output-dir artifacts/normalizer --epochs 3 --batch-size 8 --base-model google/mt5-small
```

## Next steps after training the extractor (Stage A)

Per the [implementation plan](roc_time_parsing_pipeline_end_to_end_implementation_plan_python.md), the pipeline is **Stage A (Extractor) → Stage B (Normalizer) → Resolver**.

1. **Evaluate Stage A (extractor only)**  
   Run span-level precision/recall/F1 on a labeled test set using your trained model:
   ```bash
   python scripts/evaluate_pipeline.py --input data/extractor_dev.jsonl --refdate 2026-02-04 --stage a --threshold 0.5 --extractor-dir artifacts/extractor --out artifacts/eval_stageA.jsonl
   ```

2. **Run the full pipeline (extractor + normalizer + resolver)**  
   If you trained a seq2seq normalizer, pass `--normalizer-dir` to use it:
   ```bash
   roc-time-parser parse --text "植保事業群各事業部114年第一季年終獎金" --refdate 2026-02-04 --normalizer-dir artifacts/normalizer
   ```
   If you omit `--normalizer-dir`, Stage B uses **Ollama** (LLM). Set `.env` with `MODEL_NAME`, `OLLAMA_URL` (see Environment above).

3. **End-to-end evaluation**  
   Evaluate extractor + normalizer + resolver together (gold spans are still used for Stage B in this script; extractor is used for e2e):
   ```bash
   python scripts/evaluate_pipeline.py --input data/spans_labeled.jsonl --refdate 2026-02-04 --stage all --threshold 0.5 --extractor-dir artifacts/extractor --normalizer-dir artifacts/normalizer --out artifacts/eval_e2e.jsonl
   ```

4. **Ollama fallback (optional)**  
   If you want to run Stage B via Ollama, omit `--normalizer-dir` and make sure `.env` is configured.

## Quickstart (CLI)
```bash
roc-time-parser parse --text "植保事業群各事業部114年第一季年終獎金" --refdate 2026-02-04
```

## Evaluate (smoke + metrics)
```bash
python scripts/evaluate_pipeline.py --input data/spans_labeled.jsonl --refdate 2026-02-04 --stage a --threshold 0.5 --out artifacts/eval_runA.jsonl
```

## Compare hyperparameter runs
After training different configs and evaluating each with a distinct `--out`:
```bash
python scripts/compare_eval_runs.py artifacts/eval_runA.jsonl artifacts/eval_runB.jsonl artifacts/eval_runC.jsonl
```
Optional short names:
```bash
python scripts/compare_eval_runs.py --names "lr5e-5" "lr3e-5+aug" "lr2e-5" artifacts/eval_runA.jsonl artifacts/eval_runB.jsonl artifacts/eval_runC.jsonl
```
Prints a table: Run, Precision, Recall, F1, TP, FP, FN, N, and the run with best F1.

## Data
Expected JSONL formats:
- `raw_prompts.jsonl`: `{id,text}` (no labels)
- `spans_labeled.jsonl`: `{id,text,spans:[{start,end,label:"TIME"}], ...}` (with time span labels)

You can use either:
- Manually labeled data in `data/spans_labeled.jsonl`
- Synthetically generated data (see "Generate synthetic training data" above)

## Notes
- Offsets in datasets and pipeline output are always **relative to the original raw prompt**.
- Any preprocessing used before extraction must be **offset-preserving** (no length-changing operations).
