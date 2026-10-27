# Evaluation Harness — AI Visibility Tracker

Comprehensive evaluation suite for validating accuracy, calibration, and resilience of the AI Visibility Tracker's deterministic heuristics and LLM-assisted analysis pipelines.

---

## Architecture & Stages

1. **Golden Dataset (`evals/golden/cases.jsonl`)**: 40 labeled benchmark answers spanning diverse formatting scenarios (tables, bulleted/numbered lists, prose, substring traps, negations, markdown emphasis, multiple mentions, competitors).
2. **Offline Evaluation (`evals/run_offline.py`)**: Zero-cost, 100% deterministic test measuring brand mention detection, count MAE, rank position accuracy (MAE & within-1), and competitor fallback extraction.
3. **Live Analyzer Evaluation (`evals/run_analyzer.py`)**: Live LLM analyzer eval evaluating sentiment and competitor extraction with strict budget enforcement (`--max-calls`), disk caching by `(model, case_id)`, and secondary analyzer cross-agreement checks (`--compare-analyzer`).
4. **Stability & Flip Rate (`evals/run_stability.py`)**: Tests whether prompt phrasing variations flip brand mention detection across models. Implements safety confirmation (`--yes`) when projected calls exceed 30.
5. **Pipeline Resilience (`tests/test_pipeline_resilience.py`)**: 11 unit & integration mock tests verifying graceful degradation under API rate limits, model unavailability, schema drift, partial responses, and empty inputs.
6. **Reporting & Threshold Enforcement (`evals/report.py`, `evals/check_thresholds.py`, `evals/thresholds.yaml`)**: Markdown and HTML reporting with regression delta tracking vs baseline (`evals/baseline.json`) and automated CI gating.

---

## 1. How to Label & Review Cases

Every case in [cases.jsonl](file:///c:/Users/user/OneDrive/Desktop/ai%20visbilty%20tracker/ai_visibility_tracker/evals/golden/cases.jsonl) starts with:
```json
"label_status": "needs_human_review"
```

To verify a case:
1. Open [evals/golden/cases.jsonl](file:///c:/Users/user/OneDrive/Desktop/ai%20visbilty%20tracker/ai_visibility_tracker/evals/golden/cases.jsonl).
2. Review the ground truth `labels`:
   - `mentioned`: boolean (`true` or `false`)
   - `mention_count`: integer occurrences of the brand
   - `position`: `1`-based rank if listed, or `null` if not in a ranked/sequential list
   - `sentiment`: `"positive"`, `"neutral"`, `"negative"`, or `null`
   - `competitors`: list of other brands mentioned in answer
3. Change `"label_status": "needs_human_review"` to `"label_status": "verified"`.
4. By default, all evaluation scripts evaluate **only verified cases**. To dry-run unverified cases, pass `--include-unverified`.

---

## 2. Running Offline Evals (Zero Cost / Free-Tier Friendly)

Offline evaluations require **zero network access and no API keys**:

```bash
# Evaluate only verified cases
python evals/run_offline.py --output evals/results/offline_results.json

# Dry-run including unverified cases
python evals/run_offline.py --include-unverified --output evals/results/offline_results.json
```

---

## 3. Running Live Analyzer Evals (Controlled Budget)

Prerequisites:
- Set environment variables (`ANALYZER_MODEL`, `GROQ_API_KEY`, `OPENAI_API_KEY`, etc.)

Run live evaluation with hard call caps:
```bash
# Run up to 40 calls (cached responses do not count toward cap)
python evals/run_analyzer.py --max-calls 40 --include-unverified --output evals/results/analyzer_results.json

# Optional: Run secondary analyzer cross-validation (cannot be same as test scan models)
python evals/run_analyzer.py --max-calls 40 --compare-analyzer groq/llama-3.1-8b-instant --include-unverified
```

---

## 4. Running Stability Evals (Prompt Variation Flip Rate)

Tests sensitivity across 5 prompt variants per query topic:

```bash
# Dry run check of call budget
python evals/run_stability.py --runs 3

# If total estimated calls > 30, pass --yes to confirm execution
python evals/run_stability.py --runs 3 --yes
```

---

## 5. Threshold Gates & Reporting

Check results against [thresholds.yaml](file:///c:/Users/user/OneDrive/Desktop/ai%20visbilty%20tracker/ai_visibility_tracker/evals/thresholds.yaml):
```bash
python evals/check_thresholds.py --offline evals/results/offline_results.json
```

Generate evaluation reports and update baseline:
```bash
python evals/report.py --offline evals/results/offline_results.json --analyzer evals/results/analyzer_results.json --save-baseline
```

Reports will be written to `evals/results/report_<timestamp>.md` and `evals/results/report_<timestamp>.html`.

---

## 6. Unit Tests & CI

Run evaluation metric unit tests and offline resilience tests:
```bash
pytest tests/test_eval_metrics.py tests/test_pipeline_resilience.py -v
```

CI workflow is located at [.github/workflows/eval_ci.yml](file:///c:/Users/user/OneDrive/Desktop/ai%20visbilty%20tracker/.github/workflows/eval_ci.yml).

---

## 7. Known Failure Modes Caught by Evals

| Scenario Tag | Known Failure Mode Caught |
|---|---|
| `substring_trap` | False positives when a brand name is a substring of another word (e.g. "App" in "Apple" or "Notion" in "notional"). |
| `negation_context` | False positive sentiments or incorrect positioning when an answer states a tool *lacks* a feature. |
| `table_answer` | Offset extraction and column misalignment in markdown tables (header vs row values). |
| `prose_list` | Extracting position from comma-separated inline sentences where standard `1.` list regexes fail. |
| `multiple_mentions` | Overcounting occurrences when brand name appears in URLs, footnote citations, or repetitive disclaimers. |
| `competitor_extraction` | Hallucinated competitors or missing niche alternatives in fallback heuristic mode. |
