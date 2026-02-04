# ROC Time Parsing Pipeline (Python) — End-to-End Implementation Plan

## Goal
Build a robust ROC/AD time parser for noisy user prompts (OCR-like noise). Implement:
- **Stage A (Extractor):** BIO/NER token classifier to extract one-or-more time spans from free-form text.
- **Stage B (Normalizer):** Seq2seq model to convert extracted spans into a constrained **Time DSL/JSON**, then a deterministic resolver to output `[start, end)` date ranges.

---

## Repo Layout (suggested)

```
roc_time_parser/
  pyproject.toml
  README.md
  src/roc_time_parser/
    __init__.py
    preprocess.py
    schema.py
    resolver.py
    extractor/
      __init__.py
      dataset.py
      train.py
      infer.py
      model.py
    normalizer/
      __init__.py
      dataset.py
      train.py
      infer.py
      model.py
    pipeline.py
    cli.py
  data/
    raw_prompts.jsonl
    spans_labeled.jsonl
    extractor_train.jsonl
    extractor_dev.jsonl
    normalizer_train.jsonl
    normalizer_dev.jsonl
  scripts/
    split_data.py
    make_extractor_labels.py
    bootstrap_normalizer_labels.py
    evaluate_pipeline.py
```

---

## Dependencies

### Python packages
- `transformers`, `datasets`, `accelerate`
- `torch`
- `evaluate`, `seqeval`
- `pydantic` (or `jsonschema`)
- `rapidfuzz` (optional: fuzzy OCR cleanup)
- `requests` (optional: call Ollama HTTP API from Python)

Install:
```bash
pip install -U transformers datasets accelerate evaluate seqeval pydantic rapidfuzz torch requests
```

---

## Data Contracts

### 1) Raw prompt records: `data/raw_prompts.jsonl`
```json
{"id":"...","text":"植保事業群各事業部114年第一季年終獎金"}
```

### 2) Span-labeled records (manual): `data/spans_labeled.jsonl`
Character offsets are stored against the **original raw prompt** (`text`) and are **end-exclusive**.
```json
{"id":"...","text":"...114年第一季...","spans":[{"start":8,"end":14,"label":"TIME"}]}
```
Notes:
- Offsets MUST remain valid on the raw `text` even if we run preprocessing for modeling.
- If any preprocessing step changes string length (e.g., trimming/collapsing whitespace), we must either:
  - avoid it for Stage A extraction, OR
  - maintain a mapping to convert offsets back to the raw prompt (see Preprocess section).

### 3) Normalizer training pairs: `data/normalizer_train.jsonl`
Input is extracted span (not full prompt) + reference date string:
```json
{"span":"114年第一季","refdate":"2026-02-04","target":"T1: YEAR=ROC(114,assumed=true); PERIOD=Q1"}
```

---

## OCR/Noise Normalization (shared)

### File: `src/roc_time_parser/preprocess.py`
Implement:
- `normalize_unicode(text:str)->str`
  - convert full-width digits/letters/punctuations to half-width
  - normalize dashes: `～—–－~` → `-`
- `normalize_ocr(text:str)->str`
  - common OCR confusions (configurable table), examples:
    - `O`→`0` (when surrounded by digits)
    - `I`/`l`→`1` (digit context)
    - `Q１`→`Q1` (via full-width conversion)
  - optional fuzzy fixes using `rapidfuzz` for limited token set: {Q1..Q4,H1,H2,上半年,下半年,前三季}
- `preprocess(text:str, mode:Literal["offset_preserving","compact"]="offset_preserving")->str`
  - `offset_preserving` MUST NOT change string length:
    - do NOT trim
    - do NOT collapse whitespace
    - do NOT apply fuzzy fixes that insert/delete characters
    - only allow 1-to-1 character replacements
  - `compact` MAY normalize whitespace and trim (for normalizer inputs where offsets are not needed)
- (Optional but recommended) `preprocess_with_mapping(text:str)->tuple[str, list[int]]`
  - returns `(text_norm, norm2raw_index)` where `norm2raw_index[i]` gives the raw index for `text_norm[i]`
  - only required if we ever use `compact` preprocessing before span extraction; otherwise Stage A can stay `offset_preserving`

Expected: idempotent and safe (don’t change non-time words aggressively).

---

## Time DSL (constrained output)

### File: `src/roc_time_parser/schema.py`
Define **DSL_TIME_V1** and a strict parser/validator.

**DSL format** (single-line, per span): the normalizer MUST output exactly **one** spec line per extracted span.
- Canonical: `T1: YEAR=<...>; PERIOD=<...>; [EXTRA=<...>]; [FLAGS=<...>]`
- If the model cannot produce a valid spec, it should output `T1: FLAGS=NEEDS_CLARIFICATION` (still one line).

Allowed YEAR:
- `AD(YYYY)`
- `ROC(NNN,assumed=true|false)`
- `REL(this_year|last_year|two_years_ago|next_year|same_period_last_year)`
- `RANGE(YYYY-YYYY)`

Allowed PERIOD:
- `YEAR`
- `Q1|Q2|Q3|Q4`
- `H1|H2`
- `Q1_Q3`
- `MONTH(m)`                         # single month
- `MONTH_RANGE(m1,m2)`               # inclusive months; resolver returns [m1 start, m2+1 start)
- `YTD|QTD|HTD|MTD`                  # to-date variants (end controlled by policy.rolling_end)
- `REL(this_quarter|last_quarter|next_quarter)`
- `REL(this_half|last_half|next_half)`
- `REL(this_month|last_month|next_month)`
- `ROLLING(months=n)`                # month-boundary-based rolling window (see resolver)
- `ROLLING(quarters=n)`              # derived from quarter boundaries
- `ROLLING(years=n)`                 # derived from year boundaries
- `ANCHOR`                           # requires another span as anchor (e.g., "去年同期")

### Expanded pattern coverage (MVP)
Examples the pipeline SHOULD support (normalize to DSL + resolve deterministically when possible):
- `今年` => `T1: YEAR=REL(this_year); PERIOD=YEAR`
- `去年` => `T1: YEAR=REL(last_year); PERIOD=YEAR`
- `明年` => `T1: YEAR=REL(next_year); PERIOD=YEAR`
- `本季` / `上季` / `下季` => `T1: YEAR=REL(this_year); PERIOD=REL(this_quarter|last_quarter|next_quarter)`
- `本月` / `上月` / `下月` => `T1: YEAR=REL(this_year); PERIOD=REL(this_month|last_month|next_month)`
- `今年至今` => `T1: YEAR=REL(this_year); PERIOD=YTD`
- `本季至今` => `T1: YEAR=REL(this_year); PERIOD=QTD`
- `本月至今` => `T1: YEAR=REL(this_year); PERIOD=MTD`
- `近3個月` => `T1: YEAR=REL(this_year); PERIOD=ROLLING(months=3)` (month-boundary-based)
- `近2季` => `T1: YEAR=REL(this_year); PERIOD=ROLLING(quarters=2)`
- `近3年` => `T1: YEAR=REL(this_year); PERIOD=ROLLING(years=3)`
- `去年同期` (no explicit period) => `T1: YEAR=REL(same_period_last_year); PERIOD=ANCHOR; FLAGS=NEEDS_ANCHOR`
- `去年上半年` / `去年Q1` (explicit period) => fully resolvable without anchor, e.g. `T1: YEAR=REL(last_year); PERIOD=H1`
Ambiguity policy:
- If span contains only `至今/截至目前` without an explicit anchor period/year, output `T1: FLAGS=NEEDS_CLARIFICATION` (do not guess).

Validator:
- quarters in 1..4
- ROC year 1..999
- AD year 1000..9999
- month range 1..12
- rolling months/quarters/years: n in 1..120 (configurable hard limit)

Expose:
- `parse_dsl(dsl:str)->ValidatedSpec`
- `to_json(spec)->dict`

---

## Deterministic Resolver (DB ranges)

### File: `src/roc_time_parser/resolver.py`
Implement:
- `roc_to_ad_year(roc:int)->int`  (ad=roc+1911)
- `resolve(spec:ValidatedSpec, refdate:date, policy:Policy)->List[ResolvedRange]`

Policy options:
- `assume_bare_roc_year: bool` (used in normalizer outputs when uncertain)
- `rolling_end: "refdate"|"tomorrow"` (end-exclusive)
- `recent_half_year_mode: "rolling"|"last_full_months"`

Recommended safe defaults:
- `assume_bare_roc_year=False` (don’t guess ROC unless explicitly marked)
- `rolling_end="tomorrow"` (includes `refdate` day in `[start,end)` windows)
- `recent_half_year_mode="rolling"`

ResolvedRange output:
```python
@dataclass
class ResolvedRange:
  start: date
  end: date   # end-exclusive
  grain: str
  warnings: list[str]
  source_text: str
```

Must support:
- AD / ROC years
- relative years using `refdate.year`
- Q1..Q4 => month boundaries
- H1/H2
- Q1_Q3 => [Jan1, Oct1)
- MONTH(m) => [y-m-01, first day of (m+1)]
- MONTH_RANGE(m1,m2) => [y-mStart-01, first day after mEnd]
- RANGE(YYYY-YYYY) => **multiple per-year ranges**: for each year y in [start..end], emit [y-01-01, (y+1)-01-01)
- PERIOD=REL(...) => derive the target period from `refdate` (may cross year boundaries); YEAR field is treated as a hint only
- ROLLING(months=n) (month-boundary-based):
  - rolling mode: start at first day of month that is (n-1) months before `refdate`’s month; end per `rolling_end`
  - last_full_months mode: exclude current partial month; end at first day of current month; start at first day of month n months before that
- ROLLING(quarters=n) => quarter-boundary-based windows (analogous to months)
- ROLLING(years=n) => year-boundary-based windows (analogous to months)
- To-date: `YTD/QTD/HTD/MTD` => start at period boundary; end per `rolling_end`
- `PERIOD=ANCHOR`:
  - resolver requires an anchor range (provided by pipeline); if missing, attach warning + mark as unresolved

---

## Stage A — Extractor (BIO token classifier)

### Model choice
- Start: `bert-base-chinese` token classification.
- Optional upgrade: `xlm-roberta-base` if mixed scripts/English tokens are frequent.

### Files

#### `src/roc_time_parser/extractor/dataset.py`
Implement:
- `load_spans_jsonl(path)->Dataset`
- `char_bio_labels(text, spans)->List[str]` (B-TIME/I-TIME/O)
  - labels are computed on **raw text** using raw offsets from `spans_labeled.jsonl`
- `tokenize_and_align_labels(examples, tokenizer)->dict`
  - apply `preprocess(text, mode="offset_preserving")` to the model input text (keep same length as raw)
  - use `return_offsets_mapping=True`
  - assign label per token from char labels
  - set special tokens label to `-100`

#### `src/roc_time_parser/extractor/train.py`
Implement CLI:
- inputs: train/dev jsonl, model_name, output_dir
- training: HF `Trainer` with `seqeval` metrics
- save: model + tokenizer + label map

#### `src/roc_time_parser/extractor/infer.py`
Implement:
- `extract_time_spans(text:str, model, tokenizer, threshold:float)->List[Span]`
  - keep `raw_text=text`
  - preprocess with `mode="offset_preserving"` (or maintain mapping if not)
  - predict token labels + probs
  - decode BIO into spans using offsets
  - validate spans exist in raw text; merge adjacent spans separated by small punctuation/space
  - output offsets MUST refer back to **raw_text**

Span object:
```python
@dataclass
class Span:
  start:int; end:int; text:str; score:float
```

Expected behavior:
- returns **multiple spans** when present
- robust to OCR noise via preprocess

---

## Stage B — Normalizer (Seq2seq)

### Model choice
- Start: `google/mt5-small` OR `google/byt5-small` (prefer ByT5 if OCR noise is heavy)
  - Optional alternative (inference-first): use a remote GPU-served LLM via **Ollama** with strict output validation (see Notes below).

### Files

#### `src/roc_time_parser/normalizer/dataset.py`
Implement:
- `make_input(span, refdate)->str` (e.g., `REFDATE=YYYY-MM-DD\nSPAN=...`)
- load jsonl pairs: `{span, refdate, target}`

#### `src/roc_time_parser/normalizer/train.py`
Implement:
- seq2seq training with `Trainer` (or `Seq2SeqTrainer`)
- constrain output to **single-line** DSL (no extra text) and exactly **one** `T1: ...` spec per input span
- save model/tokenizer

#### `src/roc_time_parser/normalizer/infer.py`
Implement:
- `normalize_span(span_text:str, refdate:date)->Tuple[dsl:str, confidence:float]`
- decoding settings: `num_beams=2-4`, `temperature=0.0`, `max_new_tokens` small
- validate with `parse_dsl`:
  - if invalid => retry once with stricter prompt prefix `OUTPUT ONLY DSL_TIME_V1` (same decoding)
  - if still invalid => return `T1: FLAGS=NEEDS_CLARIFICATION`
Behavior rules (important):
- Enforce **1 spec per span** (single-line `T1: ...` only).
- `"去年同期"` is **anchor-dependent**:
  - If the span does NOT explicitly include a period (e.g., no Q/H/month range), output `T1: YEAR=REL(same_period_last_year); PERIOD=ANCHOR; FLAGS=NEEDS_ANCHOR`
  - If the span DOES include a period (e.g., `"去年Q1"`, `"去年上半年"`), output a fully resolvable spec (no anchor required).

---

## Orchestration

### File: `src/roc_time_parser/pipeline.py`
Implement:
- `parse_prompt(text:str, refdate:date, models:Models, policy:Policy)->dict`

Steps:
1) keep `raw_text=text`
2) spans = `extract_time_spans(raw_text)` (extractor is responsible for any offset-preserving preprocess)
3) for each span:
   - dsl = `normalize_span(span.text, refdate)`
   - spec = `parse_dsl(dsl)`
   - ranges = `resolve(spec, refdate, policy)` (may be unresolved if `PERIOD=ANCHOR`)
4) anchor pass (recommended):
   - if any span has `PERIOD=ANCHOR`, attempt to pick an anchor from other spans in the same prompt that resolved successfully
   - if no suitable anchor, keep it unresolved and include `NEEDS_ANCHOR`
   - recommended anchor selection (deterministic):
     - prefer the nearest successfully-resolved span **to the left** (lower `start`) in the raw prompt
     - if tie, prefer higher extractor span score; if still tie, prefer higher normalizer confidence
   - recommended anchor resolution:
     - for `YEAR=REL(same_period_last_year); PERIOD=ANCHOR`, shift the anchor range by -1 year (calendar-aware) to produce the resolved range
5) return JSON:
```json
{
  "reference_date":"...",
  "spans":[{"start":0,"end":0,"text":"...","dsl":"...","ranges":[...],"warnings":[...]}],
  "warnings":["MULTIPLE_TIME_SPANS"|"NO_TIME_FOUND"|...]
}
```

### File: `src/roc_time_parser/cli.py`
Add CLI entrypoints:
- `extract --text ...`
- `normalize --span ... --refdate ...`
- `parse --text ... --refdate ...`

---

## Bootstrapping Strategy (reduce manual labeling)

### Script: `scripts/make_extractor_labels.py`
- Generate weak labels from simple keyword heuristics:
  - year markers: 年, 民國, 西元, 今年, 去年, 明年, 前年
  - period markers: 季, 第.*季, Q[1-4], 上半年/下半年, H[1-2]
  - range markers: 至, 到, -, ～, ~, 1-3月, 2025-2027
  - rolling/to-date: 近, 最近, 過去, 至今, 截至目前, 本季至今, 今年至今
  - anchor phrases: 去年同期 (mark as candidate even if incomplete)
- Output candidate spans for humans to correct (semi-automatic annotation)

### Script: `scripts/bootstrap_normalizer_labels.py`
- Use current rule-based normalizer to produce DSL targets for high-confidence spans
- Mark uncertain ones with `FLAGS=NEEDS_CLARIFICATION`

---

## Evaluation

### Stage A metrics
- span-level precision/recall/F1 using `seqeval`

### Stage B metrics
- DSL validity rate (>= 99.5% after retry)
- exact-match accuracy on canonical DSL

### End-to-end
- range accuracy on `[start,end)` for a fixed `refdate`

### Script: `scripts/evaluate_pipeline.py`
- runs pipeline against a test jsonl
- prints summary metrics + dumps failure cases

---

## Expected Results (Definition of Done)

1) `parse_prompt()` returns correct normalized ranges for sample inputs:
- ROC: `114年第一季` => 2025 Q1
- AD: `2025年Q1` => 2025 Q1
- relative: `去年上半年` => (refyear-1) H1
- range: `2025年1-3月` => [2025-01-01, 2025-04-01)
- multi-span: `114年Q1與去年同期` => 2 spans; second span resolves using the first as anchor (=> 2024 Q1)

2) Robustness:
- supports full-width digits/letters and dash variants
- tolerates minor OCR noise around Q/H markers

3) Safety:
- invalid DSL or low confidence => `NEEDS_CLARIFICATION` (no guessing)

---

## Notes
- Keep Stage A and Stage B models independent; version them separately.
- Keep resolver deterministic and fully unit-tested.
- Log production failures for active learning.
- If using Ollama for Stage B (optional):
  - run the model on your remote GPU server and call it from Python via an Ollama client/HTTP
  - keep `temperature=0` and validate every output with `parse_dsl`; on failure, fall back to `NEEDS_CLARIFICATION`

