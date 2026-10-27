# Evaluation Report — Golden Dataset Offline & Stage Analysis

**Generated:** 2026-10-27T15:51:00Z  
**Dataset:** `evals/golden/cases.jsonl` (40 benchmark cases across 11 edge-case tags)  
**Evaluation Mode:** Complete Dataset (`--include-unverified`)

---

## 1. Executive Summary

| Evaluation Suite | Primary Metric | Result | Target Threshold | Status |
|---|---|---|---|---|
| **Mention Detection** | Precision / Recall / F1 | **1.000 / 1.000 / 1.000** | F1 ≥ 0.95 | **PASS** |
| **Mention Count** | Exact Match Rate / MAE | **95.0% / 0.05** | Exact ≥ 90% | **PASS** |
| **Position Accuracy** | MAE / Within-1 Accuracy | **0.26 / 88.5%** | Within-1 ≥ 80% | **PASS** |
| **Competitor Fallback** | Fallback Recall / Macro F1 | **77.2% / 0.741** | F1 ≥ 0.70 | **PASS** |
| **Resilience & Unit Tests** | 11 Pipeline & 8 Metric Tests | **19 Passed, 0 Failed** | 100% Pass | **PASS** |

---

## 2. Deterministic Mention Detection Performance

Deterministic word-boundary and non-overlapping tokenization in `detect_brand_mention_deterministic` was evaluated on all 40 golden benchmark answers:

- **True Positives (TP):** 28
- **True Negatives (TN):** 12
- **False Positives (FP):** 0
- **False Negatives (FN):** 0
- **Precision:** 100.0%
- **Recall:** 100.0%
- **F1 Score:** **1.000**

### Substring & Edge Trap Robustness
- **`substring_trap` (Case 009, 010, 011):** Passed with 0 false positives. Matches like "pineapple" do not trigger "Apple", and lowercase "notion of" is correctly ignored while "Notion.so" alias is captured.
- **`punctuation_brand` (Cases 005, 006, 010, 040):** Punctuation handling (`Monday.com`, `Notion.so`) accurately matches without bleeding into standard words.
- **`truncated_answer` (Cases 029, 030):** Handled safely without exception or hallucinated match.
- **`non_english` (Cases 019 Spanish, 020 German):** Brand detection succeeded seamlessly.

---

## 3. Position & Rank Metrics

Evaluated on cases where target brands are listed:
- **Total Ranked Cases Evaluated ($N$):** 26
- **Mean Absolute Error (MAE):** 0.26
- **Exact Rank Match Rate:** 84.6%
- **Within-1 Accuracy ($\pm 1$ position):** **88.5%** (Threshold: $\ge 80\%$)
- *Note on prose positions:* When a brand is mentioned in running prose outside of numbered or bulleted markdown lists, the heuristic defaults to position 1 or maintains rank order based on sentence index.

---

## 4. Competitor Fallback Extraction Performance

Evaluated against `fallback_extract_competitors` with zero LLM API dependency:
- **Extracted Competitors Precision:** 71.4%
- **Extracted Competitors Recall:** 77.2%
- **Competitor Macro F1:** **0.741** (Threshold: $\ge 0.70$)
- **Markdown Tables:** Extracts competitor names from column 1 while ignoring table header rows (`Tool`, `Price`, `Pros`, `Cons`).
- **Markdown Lists & Bold Mentions:** Identifies competitors from `**Name**` patterns and bullet lines.

---

## 5. Pipeline Resilience & Fault Tolerance (`tests/test_pipeline_resilience.py`)

All 11 test scenarios passed:
1. `test_rate_limit_retry_and_graceful_degradation` — Verifies automatic retry and fallback state on `RateLimitExceededError`.
2. `test_model_unavailable_error_handling` — Confirms graceful degradation when a provider model returns 503 or unavailability.
3. `test_partial_json_and_markdown_fencing` — Confirms parsing of dirty LLM JSON fenced responses.
4. `test_schema_drift_handling` — Verifies default fallback when LLM omits expected keys or returns invalid types.
5. `test_empty_and_whitespace_responses` — Validates safe handling of empty strings and whitespace without throwing unhandled exceptions.
6. `test_brand_report_incomplete_analysis_banner` — Verifies banner trigger when analyzer failure rate > 30%.
7. `test_top_competitors_count_cap` — Confirms competitor lists respect max limits.
8. `test_sov_calculation_zero_division` — Verifies SOV returns 0.0% instead of division by zero when no brand mentions exist.
9. `test_mention_count_deduplication` — Validates non-overlapping tokenization.
10. `test_client_manager_cache` — Confirms connection reuse.
11. `test_offline_eval_dry_run_flag` — Confirms `--include-unverified` loads all 40 golden cases.

---

## 6. Threshold Gate Status

```
============================================================
EVALUATION THRESHOLD VERIFICATION
============================================================
[PASS] mention_f1: 1.0000 (Threshold: >=0.9500)
[PASS] position_within1: 0.8846 (Threshold: >=0.8000)
[PASS] competitor_fallback_f1: 0.7410 (Threshold: >=0.7000)
------------------------------------------------------------
ALL THRESHOLDS SATISFIED.
============================================================
```
