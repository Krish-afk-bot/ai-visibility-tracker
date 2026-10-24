import pytest
from unittest.mock import patch

from llm_client import (
    get_available_models,
    get_analyzer_model,
    validate_analyzer_config
)
from analysis import (
    find_model_disagreements,
    generate_comparison_summary,
    calculate_model_scores,
    analyze_sentiment_and_competitors,
    find_visibility_gaps,
    generate_prompt_ideas,
    generate_gap_recommendation
)


# ==========================================
# 1. Validation & Analyzer Checks
# ==========================================

def test_gemini_included_only_when_both_key_and_models_set(monkeypatch):
    """Gemini must only be included in get_available_models() when both key and models are set."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    
    # Neither set
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODELS", raising=False)
    assert get_available_models() == []

    # Only key set
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MODELS", raising=False)
    assert get_available_models() == []

    # Both set
    monkeypatch.setenv("GEMINI_MODELS", "gemini-1.5-flash, gemini-2.0-flash")
    available = get_available_models()
    assert "gemini:gemini-1.5-flash" in available
    assert "gemini:gemini-2.0-flash" in available


def test_analyzer_model_required_no_fallback(monkeypatch):
    """ANALYZER_MODEL is required and does not fall back to a tested model."""
    monkeypatch.delenv("ANALYZER_MODEL", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.setenv("GROQ_MODELS", "llama-3.1-8b-instant")
    
    # Unset analyzer returns None
    assert get_analyzer_model() is None

    # Validation must reject unset analyzer
    is_valid, err = validate_analyzer_config(None)
    assert not is_valid
    assert "ANALYZER_MODEL is required" in err


def test_analyzer_model_unsupported_provider():
    """Unsupported provider in ANALYZER_MODEL is rejected."""
    is_valid, err = validate_analyzer_config("cohere:command-r")
    assert not is_valid
    assert "unsupported provider 'cohere'" in err


def test_analyzer_model_missing_key(monkeypatch):
    """Provider configured for analyzer must have an API key."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    is_valid, err = validate_analyzer_config("groq:llama-3.1-8b-instant")
    assert not is_valid
    assert "Missing API key for analyzer provider 'groq'" in err


def test_analyzer_model_cannot_be_in_selected_models(monkeypatch):
    """Model must never grade its own answer."""
    monkeypatch.setenv("GEMINI_API_KEY", "valid-key")
    selected = ["groq:llama-3.1-8b-instant", "gemini:gemini-1.5-flash"]
    is_valid, err = validate_analyzer_config("gemini:gemini-1.5-flash", selected_models=selected)
    assert not is_valid
    assert "cannot be in the list of selected test models" in err


def test_analyzer_failure_reports_failed_status_and_error():
    """When analyzer fails, it returns analyzer_status='failed' and records the error."""
    with patch("analysis.query_ai", side_effect=Exception("Rate limit 429 on analyzer")):
        res = analyze_sentiment_and_competitors(
            ai_response="Some text mentioning Nike",
            brand_name="Nike",
            analyzer_model="groq:llama-3.1-8b-instant"
        )
        assert res["analyzer_status"] == "failed"
        assert res["sentiment"] == "unknown"
        assert "Rate limit 429 on analyzer" in res["analyzer_error"]


# ==========================================
# 2. Multi-Model Disagreement & Summary Tests
# ==========================================

def test_find_model_disagreements():
    """Identify prompts where some models mention the brand and others do not."""
    results = [
        # Prompt 1: Disagreement (m1 mentions, m2 does not)
        {"prompt": "Best CRM?", "provider": "p1", "model": "m1", "brand_mentioned": True, "status": "ok"},
        {"prompt": "Best CRM?", "provider": "p2", "model": "m2", "brand_mentioned": False, "status": "ok"},
        # Prompt 2: Agreement (both mention)
        {"prompt": "Top tools?", "provider": "p1", "model": "m1", "brand_mentioned": True, "status": "ok"},
        {"prompt": "Top tools?", "provider": "p2", "model": "m2", "brand_mentioned": True, "status": "ok"},
        # Prompt 3: Agreement (neither mentions)
        {"prompt": "Other things?", "provider": "p1", "model": "m1", "brand_mentioned": False, "status": "ok"},
        {"prompt": "Other things?", "provider": "p2", "model": "m2", "brand_mentioned": False, "status": "ok"}
    ]

    disagreements = find_model_disagreements(results)
    assert len(disagreements) == 1
    d = disagreements[0]
    assert d["prompt"] == "Best CRM?"
    assert d["mentioned_by"] == ["p1:m1"]
    assert d["not_mentioned_by"] == ["p2:m2"]


def test_generate_comparison_summary_and_ranking():
    """Verify summary string and failed model exclusion from 0%."""
    results = [
        {"prompt": "P1", "provider": "p1", "model": "m1", "brand_mentioned": True, "status": "ok", "run_index": 1, "sentiment": "positive"},
        {"prompt": "P2", "provider": "p1", "model": "m1", "brand_mentioned": False, "status": "ok", "run_index": 1, "sentiment": "neutral"},
        {"prompt": "P1", "provider": "p2", "model": "m2", "brand_mentioned": True, "status": "ok", "run_index": 1, "sentiment": "positive"},
        {"prompt": "P2", "provider": "p2", "model": "m2", "brand_mentioned": True, "status": "ok", "run_index": 1, "sentiment": "positive"},
        # m3 failed completely
        {"prompt": "P1", "provider": "p3", "model": "m3", "brand_mentioned": False, "status": "failed", "run_index": 1, "error": "timeout"},
        {"prompt": "P2", "provider": "p3", "model": "m3", "brand_mentioned": False, "status": "failed", "run_index": 1, "error": "timeout"}
    ]

    scores = calculate_model_scores(results)
    assert scores["p1:m1"]["score"] == 50.0
    assert scores["p2:m2"]["score"] == 100.0
    # Failed model score is None (NOT 0.0)
    assert scores["p3:m3"]["score"] is None
    assert scores["p3:m3"]["failed_calls"] == 2

    summary = generate_comparison_summary(scores)
    assert "p2:m2" in summary
    assert "2/2 prompts (100%)" in summary
    assert "p1:m1" in summary
    assert "1/2 prompts (50%)" in summary
    assert "p3:m3 failed calls excluded" in summary


# ==========================================
# 3. Visibility Gaps & Prompt Ideas (Stage 3) Tests
# ==========================================

def test_find_visibility_gaps_rule_based():
    """Verify rule-based gap detection identifies prompts where competitors appear in >=50% and brand in <50%."""
    results = [
        # Prompt 1: High Gap (Competitors in 2 of 2 = 100%, Brand in 0 of 2 = 0% -> Gap = 100%)
        {
            "prompt": "Top CRM for startups",
            "provider": "gemini",
            "model": "gemini-1.5-flash",
            "brand_mentioned": False,
            "status": "ok",
            "competitors": ["Salesforce", "Pipedrive"]
        },
        {
            "prompt": "Top CRM for startups",
            "provider": "groq",
            "model": "llama-3.1-8b-instant",
            "brand_mentioned": False,
            "status": "ok",
            "competitors": ["Salesforce", "Zoho"]
        },
        # Prompt 2: Moderate Gap (Competitors in 2 of 2 = 100%, Brand in 1 of 2 = 50% -> NOT a gap since brand >= 50%)
        {
            "prompt": "Best inbound marketing tool",
            "provider": "gemini",
            "model": "gemini-1.5-flash",
            "brand_mentioned": True,
            "status": "ok",
            "competitors": ["ActiveCampaign"]
        },
        {
            "prompt": "Best inbound marketing tool",
            "provider": "groq",
            "model": "llama-3.1-8b-instant",
            "brand_mentioned": False,
            "status": "ok",
            "competitors": ["ActiveCampaign"]
        },
        # Prompt 3: Minor Gap (Competitors in 1 of 2 = 50%, Brand in 0 of 2 = 0% -> Gap = 50%)
        {
            "prompt": "Email marketing automation",
            "provider": "gemini",
            "model": "gemini-1.5-flash",
            "brand_mentioned": False,
            "status": "ok",
            "competitors": ["Mailchimp"]
        },
        {
            "prompt": "Email marketing automation",
            "provider": "groq",
            "model": "llama-3.1-8b-instant",
            "brand_mentioned": False,
            "status": "ok",
            "competitors": []
        }
    ]

    gaps = find_visibility_gaps(results, target_brand="HubSpot")
    assert len(gaps) == 2

    # Top gap should be "Top CRM for startups" with gap_size = 100.0
    assert gaps[0]["prompt"] == "Top CRM for startups"
    assert gaps[0]["gap_size"] == 100.0
    assert gaps[0]["competitor_coverage_pct"] == 100.0
    assert gaps[0]["brand_coverage_pct"] == 0.0
    # Salesforce should be top competitor with 2 mentions
    assert gaps[0]["competitors"][0]["name"] == "Salesforce"
    assert gaps[0]["competitors"][0]["count"] == 2
    assert "gemini:gemini-1.5-flash" in gaps[0]["models_omitting_brand"]

    # Second gap should be "Email marketing automation" with gap_size = 50.0
    assert gaps[1]["prompt"] == "Email marketing automation"
    assert gaps[1]["gap_size"] == 50.0


def test_generate_prompt_ideas_and_tag_assignment():
    """Verify generate_prompt_ideas parses valid funnel tags and handles LLM output."""
    mock_json_resp = """{
        "prompts": [
            {"text": "What is the best CRM for small teams?", "tag": "top-of-funnel"},
            {"text": "HubSpot pricing comparison", "tag": "branded"},
            {"text": "Top sales software alternatives", "tag": "non-branded"},
            {"text": "Which CRM has automated deal scoring?", "tag": "bottom-of-funnel"}
        ]
    }"""

    with patch("analysis.query_ai", return_value=mock_json_resp):
        ideas = generate_prompt_ideas(
            industry="CRM Software",
            brand_name="HubSpot",
            analyzer_model="groq:llama-3.1-8b-instant",
            count=4
        )
        assert len(ideas) == 4
        assert ideas[0]["tag"] == "top-of-funnel"
        assert ideas[1]["tag"] == "branded"
        assert ideas[2]["tag"] == "non-branded"
        assert ideas[3]["tag"] == "bottom-of-funnel"
        assert ideas[0]["text"] == "What is the best CRM for small teams?"


def test_generate_gap_recommendation_data_driven():
    """Verify recommendations are strictly factual based on data without ranking guarantee claims."""
    gap_data = {
        "prompt": "Top CRM for startups",
        "gap_size": 100.0,
        "answers_count": 3,
        "brand_answers_count": 0,
        "competitors": [{"name": "Asana", "count": 3}],
        "models_omitting_brand": ["gemini:gemini-1.5-flash", "groq:llama-3", "openrouter:mistral"]
    }

    rec = generate_gap_recommendation(gap_data, target_brand="Monday.com")
    assert "Asana was named in 3 of 3 language model answers" in rec
    assert "Monday.com was omitted across all 3 answers" in rec
    assert "Check whether your public documentation or website has an informative resource" in rec
    # Ensure no false guarantees or manipulation claims
    assert "guarantee" not in rec.lower()
    assert "rank higher" not in rec.lower()


