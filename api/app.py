"""
Minimal API service for ROC time parsing.

POST /parse: accepts a prompt (and optional reference_date), returns structured
time_expressions with DB-friendly YYYY-MM-DD ranges (end-inclusive).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schema import (
    ParseRequest,
    ParseResponse,
    TimeExpression,
    TimeRange,
    parse_refdate,
)


def _end_inclusive(end_exclusive: date) -> date:
    """Convert resolver end-exclusive date to end-inclusive for DB."""
    return end_exclusive - timedelta(days=1)


def _pipeline_result_to_response(prompt: str, refdate: date, raw: dict[str, Any]) -> ParseResponse:
    """Map pipeline output to API response schema (end-inclusive ranges)."""
    time_expressions: list[TimeExpression] = []
    for span in raw.get("spans") or []:
        ranges_out: list[TimeRange] = []
        for r in span.get("ranges") or []:
            start = r.get("start")
            end = r.get("end")
            grain = r.get("grain", "month")
            if start and end:
                start_d = date.fromisoformat(start)
                end_d = date.fromisoformat(end)
                ranges_out.append(
                    TimeRange(
                        start_time=start_d.isoformat(),
                        end_time=_end_inclusive(end_d).isoformat(),
                        granularity=grain,
                    )
                )
        time_expressions.append(
            TimeExpression(
                source_text=span.get("text", ""),
                ranges=ranges_out,
                confidence=float(span.get("dsl_confidence", 0.0)),
                flags=list(span.get("warnings") or []),
            )
        )
    return ParseResponse(
        prompt=prompt,
        reference_date=refdate.isoformat(),
        time_expressions=time_expressions,
        warnings=list(raw.get("warnings") or []),
    )


app = FastAPI(
    title="ROC Time Parser API",
    description="Parse zh_TW prompts into DB-compatible time ranges (YYYY-MM-DD).",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded at startup, used in /parse
_models: Any = None
_policy: Any = None


@app.on_event("startup")
def startup() -> None:
    from roc_time_parser.config import load_dotenv_into_env, load_settings
    from roc_time_parser.extractor.model import load_extractor
    from roc_time_parser.normalizer.model import load_normalizer
    from roc_time_parser.pipeline import Models
    from roc_time_parser.policy import Policy

    load_dotenv_into_env()
    ex_model, ex_tok = load_extractor()
    n_model, n_tok = None, None
    settings = None
    try:
        n_model, n_tok = load_normalizer()
    except (FileNotFoundError, RuntimeError):
        try:
            settings = load_settings()
        except RuntimeError:
            pass  # /parse will fail if both normalizer and Ollama are missing
    global _models, _policy
    _models = Models(
        extractor_model=ex_model,
        extractor_tokenizer=ex_tok,
        normalizer_settings=settings,
        normalizer_model=n_model,
        normalizer_tokenizer=n_tok,
    )
    _policy = Policy()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/parse", response_model=ParseResponse)
def parse(req: ParseRequest) -> ParseResponse:
    """Parse a prompt and return structured time expressions with date ranges."""
    from roc_time_parser.pipeline import parse_prompt

    if _models is None or _policy is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        refdate = parse_refdate(req.reference_date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid reference_date: {e}") from e
    raw = parse_prompt(
        req.prompt,
        refdate=refdate,
        models=_models,
        policy=_policy,
        extractor_threshold=0.5,
    )
    return _pipeline_result_to_response(req.prompt, refdate, raw)
