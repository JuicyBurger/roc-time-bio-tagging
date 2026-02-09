# ROC Time Parser API

Minimal HTTP service that parses zh_TW prompts and returns structured time ranges (YYYY-MM-DD).

## Prerequisites

- Installed package and trained models (see project root README):
  ```bash
  pip install -e .
  ```
- `.env` at repo root with `EXTRACTOR_MODEL_DIR` (and optionally `NORMALIZER_MODEL_DIR` or Ollama settings for Stage B).

## Run

From the **project root** (so `api` and `roc_time_parser` resolve):

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

- **Health:** `GET http://localhost:8000/health`
- **Parse:** `POST http://localhost:8000/parse` with JSON body:
  - `prompt` (required): raw prompt string
  - `reference_date` (optional): `YYYY-MM-DD`; default is today

## Response shape

```json
{
  "prompt": "植保事業群114年第一季年終獎金",
  "reference_date": "2026-02-06",
  "time_expressions": [
    {
      "source_text": "114年第一季年終",
      "ranges": [
        {
          "start_time": "2025-01-01",
          "end_time": "2025-03-31",
          "granularity": "quarter"
        }
      ],
      "confidence": 0.9,
      "flags": []
    }
  ],
  "warnings": []
}
```

- **Ranges** use **end-inclusive** dates (e.g. Q1 → `end_time: "2025-03-31"`) for DB compatibility.
- **Granularity** values match the resolver / `scripts/generate_prompts.py`: `month`, `quarter`, `half`, `year`, `month_range`, `quarter_range`, `ytd`, `qtd`, `mtd`, `htd`, `rolling_months`, `rolling_quarters`, `rolling_years`.

## Example

```bash
curl -X POST http://localhost:8000/parse \
  -H "Content-Type: application/json" \
  -d '{"prompt": "植保事業群114年第一季年終獎金", "reference_date": "2026-02-06"}'
```
