import pytest
from unittest.mock import patch, MagicMock

from analysis import (
    detect_brand_mention_deterministic,
    parse_json_from_llm,
    analyze_sentiment_and_competitors,
    analyze_brand_mention,
    extract_competitors,
    fallback_extract_competitors,
    calculate_visibility_score,
    calculate_model_scores,
    compute_brand_report_metrics
)


# ==========================================
# Deterministic Brand Mention & Boundary Tests
# ==========================================

def test_word_boundary_detection():
    """Verify apple does not match pineapple, and case-insensitivity works."""
    text_pineapple = "We love pineapple on pizza and BigApple is nice."
    res1 = detect_brand_mention_deterministic(text_pineapple, "apple")
    assert res1["brand_mentioned"] is False
    assert res1["mention_count"] == 0

    text_apple = "The new Apple MacBook Pro is great. apple makes solid hardware."
    res2 = detect_brand_mention_deterministic(text_apple, "Apple")
    assert res2["brand_mentioned"] is True
    assert res2["mention_count"] == 2


def test_brand_aliases_and_punctuation():
    """Verify aliases like 'Monday' / 'Monday.com' match properly."""
    text = "We use Monday.com for task tracking and Monday for scheduling."
    res = detect_brand_mention_deterministic(text, "Monday", aliases=["Monday.com"])
    assert res["brand_mentioned"] is True
    assert res["mention_count"] == 2

    # Brand with dots/symbols e.g. Notion.so
    text2 = "Check out Notion.so for documentation."
    res2 = detect_brand_mention_deterministic(text2, "Notion.so")
    assert res2["brand_mentioned"] is True
    assert res2["mention_count"] == 1


def test_mention_position_in_numbered_and_bulleted_lists():
    """Verify position extracts list rank."""
    numbered_text = """Top running shoes:
1. Brooks Ghost 15
2. Nike Air Zoom Pegasus
3. Asics Gel-Kayano
"""
    res = detect_brand_mention_deterministic(numbered_text, "Nike")
    assert res["brand_mentioned"] is True
    assert res["position"] == 2

    bulleted_text = """Top tools:
* Slack
* Monday.com
* Asana
"""
    res2 = detect_brand_mention_deterministic(bulleted_text, "Monday.com")
    assert res2["brand_mentioned"] is True
    assert res2["position"] == 2


# ==========================================
# Robust JSON Parsing & LLM Fallback Tests
# ==========================================

def test_parse_json_from_llm():
    # Clean JSON
    assert parse_json_from_llm('{"sentiment": "positive", "competitors": ["A"]}') == {
        "sentiment": "positive", "competitors": ["A"]
    }

    # Markdown fenced JSON with commentary
    fenced = """Here is your output:
```json
{
    "sentiment": "neutral",
    "competitors": ["Slack", "Teams"]
}
```
Hope that helps!"""
    parsed = parse_json_from_llm(fenced)
    assert parsed is not None
    assert parsed["sentiment"] == "neutral"
    assert parsed["competitors"] == ["Slack", "Teams"]

    # Malformed text without JSON returns None
    assert parse_json_from_llm("This response has no brackets at all.") is None


def test_malformed_json_fallback_and_unknown_sentiment():
    """When analyzer model outputs malformed JSON, mark sentiment as 'unknown' and fallback competitors."""
    with patch("analysis.query_ai", return_value="I couldn't generate json sorry"):
        analysis = analyze_sentiment_and_competitors(
            ai_response="Asana and Trello are good alternatives to Monday.com",
            brand_name="Monday.com",
            analyzer_model="groq:llama-3.1-8b-instant",
            known_competitors=["Asana", "Trello", "ClickUp"]
        )
        assert analysis["sentiment"] == "unknown"
        assert "Asana" in analysis["competitors"]
        assert "Trello" in analysis["competitors"]


def test_combined_sentiment_and_competitor_single_call():
    """Verify single call returns both sentiment and deduplicated competitors excluding target brand."""
    mock_llm_json = """{
        "sentiment": "positive",
        "competitors": ["Nike", "Adidas", "Puma", "adidas"]
    }"""
    with patch("analysis.query_ai", return_value=mock_llm_json) as mock_query:
        analysis = analyze_sentiment_and_competitors(
            ai_response="Nike is best, but Adidas and Puma are decent.",
            brand_name="Nike",
            analyzer_model="groq:llama-3.1-8b-instant"
        )
        assert mock_query.call_count == 1
        assert analysis["sentiment"] == "positive"
        # Nike excluded, adidas deduplicated
        assert analysis["competitors"] == ["Adidas", "Puma"]


# ==========================================
# Score Calculation Tests
# ==========================================

def test_calculate_visibility_score_zero_prompts():
    assert calculate_visibility_score([]) is None


def test_calculate_visibility_score_excludes_failed_calls():
    """Failed calls must be excluded from the denominator."""
    results = [
        {"status": "ok", "brand_mentioned": True, "model": "m1", "provider": "p1"},
        {"status": "ok", "brand_mentioned": False, "model": "m1", "provider": "p1"},
        {"status": "failed", "brand_mentioned": False, "error": "429 Rate limited", "model": "m1", "provider": "p1"}
    ]
    # Total successful = 2, mentions = 1 => 50.0% (NOT 33.3%)
    score = calculate_visibility_score(results)
    assert score == 50.0

    # If all calls failed, returns None
    failed_only = [
        {"status": "failed", "brand_mentioned": False, "error": "Timeout", "model": "m1", "provider": "p1"}
    ]
    assert calculate_visibility_score(failed_only) is None


def test_calculate_model_scores_ranges():
    results = [
        # Run 1: 1 mentioned out of 1 -> 100%
        {"status": "ok", "brand_mentioned": True, "position": 1, "sentiment": "positive", "run_index": 1, "model": "m1", "provider": "p1"},
        # Run 2: 0 mentioned out of 1 -> 0%
        {"status": "ok", "brand_mentioned": False, "position": None, "sentiment": "neutral", "run_index": 2, "model": "m1", "provider": "p1"},
        # Run 3: failed call
        {"status": "failed", "brand_mentioned": False, "position": None, "sentiment": "neutral", "run_index": 3, "model": "m1", "provider": "p1"}
    ]
    summary = calculate_model_scores(results)
    m1_stats = summary["p1:m1"]
    assert m1_stats["score"] == 50.0  # 1 out of 2 successful
    assert m1_stats["score_min"] == 0.0
    assert m1_stats["score_max"] == 100.0
    assert m1_stats["successful_calls"] == 2
    assert m1_stats["failed_calls"] == 1
    assert m1_stats["avg_position"] == 1.0
    assert m1_stats["sentiments"]["positive"] == 1


# ==========================================
# Brand Report Metrics (Stage 2) Tests
# ==========================================

def test_compute_brand_report_metrics_exclusions_and_sov():
    """Verify failed language model calls are excluded from metrics and SOV calculates accurately."""
    mock_results = [
        # Call 1: Target brand HubSpot mentioned (#1), competitors Salesforce, Pipedrive
        {
            "status": "ok",
            "prompt": "Best CRM",
            "provider": "gemini",
            "model": "gemini-1.5-flash",
            "brand_mentioned": True,
            "mention_count": 1,
            "position": 1,
            "sentiment": "positive",
            "competitors": ["Salesforce", "Pipedrive"],
            "scan_id": 1,
            "scan_created_at": "2026-10-01 10:00:00"
        },
        # Call 2: Target brand mentioned (#2), competitor Salesforce
        {
            "status": "ok",
            "prompt": "Top sales tools",
            "provider": "groq",
            "model": "llama-3.1-8b-instant",
            "brand_mentioned": True,
            "mention_count": 2,
            "position": 2,
            "sentiment": "positive",
            "competitors": ["Salesforce"],
            "scan_id": 1,
            "scan_created_at": "2026-10-01 10:00:00"
        },
        # Call 3: Target brand NOT mentioned, competitor Salesforce, Zoho
        {
            "status": "ok",
            "prompt": "Customer management",
            "provider": "gemini",
            "model": "gemini-1.5-flash",
            "brand_mentioned": False,
            "mention_count": 0,
            "position": None,
            "sentiment": "neutral",
            "competitors": ["Salesforce", "Zoho"],
            "scan_id": 2,
            "scan_created_at": "2026-10-10 10:00:00"
        },
        # Call 4: FAILED call - MUST be excluded from all metric calculations
        {
            "status": "failed",
            "prompt": "Best CRM",
            "provider": "openrouter",
            "model": "mistral-7b",
            "error": "429 Rate limited",
            "brand_mentioned": False,
            "mention_count": 0,
            "position": None,
            "sentiment": "neutral",
            "competitors": [],
            "scan_id": 1,
            "scan_created_at": "2026-10-01 10:00:00"
        }
    ]

    metrics = compute_brand_report_metrics(mock_results, target_brand="HubSpot", top_competitors_count=5)

    assert metrics["total_answers"] == 4
    assert metrics["successful_answers"] == 3
    assert metrics["failed_calls"] == 1
    assert metrics["unique_prompts_count"] == 3

    # Share of voice is measured in answers, not word occurrences:
    # HubSpot is named in 2 answers (call 1, call 2) even though call 2 repeats it twice.
    # Salesforce is named in 3 answers, Pipedrive in 1, Zoho in 1.
    # Every successful answer names HubSpot or a competitor, so n = 3.
    # Share of voice = 2 / 3 = 66.7%
    assert metrics["target_brand_mentions"] == 2
    assert metrics["answers_naming_any_tracked_brand"] == 3
    assert metrics["sov_n"] == 3
    assert metrics["share_of_voice_pct"] == 66.7

    # Sentiment distribution
    assert metrics["sentiment_counts"]["positive"] == 2
    assert metrics["sentiment_counts"]["neutral"] == 0

    # Comparison table check
    table = metrics["comparison_table"]
    # Top brand is HubSpot (target)
    target_row = [r for r in table if r["is_target"]][0]
    # Coverage: HubSpot mentioned in 2 of 3 successful answers = 66.67%
    assert round(target_row["coverage_pct"], 1) == 66.7
    assert target_row["answers_with_mention"] == 2
    assert target_row["total_answers"] == 3
    # Average position: (1 + 2) / 2 = 1.5
    assert target_row["avg_position"] == 1.5

    # Comparison shares use the same answer unit as the target brand.
    assert target_row["mentions"] == 2
    assert target_row["share_of_mentions_pct"] == 66.7

    # Top competitor: Salesforce (named in 3 of 3 answers = 100%)
    sf_row = [r for r in table if r["brand"] == "Salesforce"][0]
    assert sf_row["coverage_pct"] == 100.0
    assert sf_row["answers_with_mention"] == 3
    assert sf_row["mentions"] == 3
    assert sf_row["share_of_mentions_pct"] == 100.0

    # Coverage over time: 2 distinct scans (scan 1 and scan 2)
    assert metrics["scans_count"] == 2
    assert len(metrics["coverage_over_time"]) > 0

