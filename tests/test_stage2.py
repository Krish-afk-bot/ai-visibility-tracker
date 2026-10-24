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
    calculate_model_scores
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
