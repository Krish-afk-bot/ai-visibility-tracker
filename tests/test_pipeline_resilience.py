"""
tests/test_pipeline_resilience.py — Stage 5: pipeline and resilience evals.

All tests are fully offline using mocks. No network or API keys needed.

Assertions:
  * 429-then-success: failed call NOT counted as 'not mentioned'
  * persistent 429: fails gracefully, excluded from denominator
  * 404 retired model: ModelUnavailableError raised cleanly
  * timeout: excluded from denominator
  * empty completion: raises LLMClientError
  * truncated completion (finish_reason=length): treated as ok but limited
  * one failing model does not stop others
  * analyzer down: SOV reports 'Not available', not 100%
  * error messages contain no API keys, user IDs, or raw provider JSON
  * sample dataset never takes on the sidebar brand name
"""
import json
import re
import pytest
from unittest.mock import patch, MagicMock, call
import httpx
from openai import RateLimitError, APIStatusError, APITimeoutError


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_mock_response(content: str, finish_reason: str = "stop") -> MagicMock:
    """Return a mock openai.ChatCompletion-like response."""
    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_choice.finish_reason = finish_reason
    mock_resp.choices = [mock_choice]
    return mock_resp


def _rate_limit_error(retry_after: float = 0.01) -> RateLimitError:
    response = httpx.Response(
        status_code=429,
        headers={"Retry-After": str(retry_after)},
        request=httpx.Request("POST", "https://api.groq.com"),
    )
    return RateLimitError(message="Rate limited", response=response, body=None)


def _api_status_error(status_code: int, message: str = "error") -> APIStatusError:
    response = httpx.Response(
        status_code=status_code,
        request=httpx.Request("POST", "https://api.groq.com"),
    )
    return APIStatusError(message=message, response=response, body=None)


def _timeout_error() -> APITimeoutError:
    return APITimeoutError(request=httpx.Request("POST", "https://api.groq.com"))


# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from llm_client import (
    query_ai,
    client_manager,
    RateLimitExceededError,
    ModelUnavailableError,
    LLMClientError,
    summarize_analyzer_error,
    ConfigurationError,
)
from analysis import (
    calculate_visibility_score,
    compute_brand_report_metrics,
    is_analysis_incomplete,
    ANALYZER_INCOMPLETE_THRESHOLD,
)
from sample_data import SAMPLE_BRAND, resolve_display_brand, load_sample_into_state


# ---------------------------------------------------------------------------
# S5-01: 429 then success
# ---------------------------------------------------------------------------

def test_429_then_success_not_counted_as_not_mentioned(monkeypatch):
    """A call that hits 429 then succeeds must be counted in the denominator."""
    monkeypatch.setenv("GROQ_API_KEY", "mock_key")
    client_manager.reset_clients()

    mock_client = MagicMock()
    rate_err = _rate_limit_error(retry_after=0.001)
    ok_resp = _make_mock_response("Notion is great for notes.")

    mock_client.chat.completions.create.side_effect = [rate_err, ok_resp]

    with patch.object(client_manager, "get_client", return_value=mock_client):
        result = query_ai("Best note apps?", "groq", "llama3", max_retries=2, initial_backoff=0.001)

    assert result == "Notion is great for notes."
    assert mock_client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# S5-02: Persistent 429 → RateLimitExceededError (excluded from denominator)
# ---------------------------------------------------------------------------

def test_persistent_429_excluded_from_denominator():
    """Persistent 429 → failed call is excluded from visibility score denominator."""
    results = [
        {"status": "ok", "brand_mentioned": True},
        {"status": "ok", "brand_mentioned": False},
        {"status": "failed", "error": "Rate limit exceeded — excluded"},
    ]
    score = calculate_visibility_score(results)
    # 1 mention / 2 successful = 50% (NOT 1/3 = 33%)
    assert score == 50.0


def test_persistent_429_raises_rate_limit_exceeded(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "mock_key")
    client_manager.reset_clients()
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _rate_limit_error()

    with patch.object(client_manager, "get_client", return_value=mock_client):
        with pytest.raises(RateLimitExceededError):
            query_ai("test", "groq", "llama3", max_retries=1, initial_backoff=0.001)


# ---------------------------------------------------------------------------
# S5-03: 404 retired model
# ---------------------------------------------------------------------------

def test_404_retired_model_raises_model_unavailable(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "mock_key")
    client_manager.reset_clients()
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _api_status_error(404, "Model retired")

    with patch.object(client_manager, "get_client", return_value=mock_client):
        with pytest.raises(ModelUnavailableError) as exc_info:
            query_ai("test", "groq", "retired-model", max_retries=0)
    assert "unavailable" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# S5-04: Timeout → excluded from denominator
# ---------------------------------------------------------------------------

def test_timeout_excluded_from_denominator(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "mock_key")
    client_manager.reset_clients()
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _timeout_error()

    with patch.object(client_manager, "get_client", return_value=mock_client):
        with pytest.raises(LLMClientError):
            query_ai("test", "groq", "llama3", max_retries=0)

    # Simulate what the app does: record as "failed", exclude from score
    results = [
        {"status": "ok", "brand_mentioned": True},
        {"status": "failed", "error": "Network or timeout failure"},
    ]
    score = calculate_visibility_score(results)
    assert score == 100.0  # 1/1 successful, failed excluded


# ---------------------------------------------------------------------------
# S5-05: Empty completion
# ---------------------------------------------------------------------------

def test_empty_completion_raises_llm_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "mock_key")
    client_manager.reset_clients()
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response("   ")  # whitespace only

    with patch.object(client_manager, "get_client", return_value=mock_client):
        with pytest.raises(LLMClientError) as exc_info:
            query_ai("test", "groq", "llama3", max_retries=0)
    assert "empty" in str(exc_info.value).lower()


def test_no_choices_raises_llm_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "mock_key")
    client_manager.reset_clients()
    mock_client = MagicMock()
    empty_resp = MagicMock()
    empty_resp.choices = []
    mock_client.chat.completions.create.return_value = empty_resp

    with patch.object(client_manager, "get_client", return_value=mock_client):
        with pytest.raises(LLMClientError) as exc_info:
            query_ai("test", "groq", "llama3", max_retries=0)
    assert "empty" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# S5-06: Truncated completion (finish_reason=length)
# ---------------------------------------------------------------------------

def test_truncated_completion_is_accepted_not_rejected(monkeypatch):
    """A truncated but non-empty completion is returned (not raised as an error)."""
    monkeypatch.setenv("GROQ_API_KEY", "mock_key")
    client_manager.reset_clients()
    mock_client = MagicMock()
    truncated_content = "The best tools for productivity are Notion, which helps you or"
    mock_client.chat.completions.create.return_value = _make_mock_response(
        truncated_content, finish_reason="length"
    )

    with patch.object(client_manager, "get_client", return_value=mock_client):
        result = query_ai("Best productivity tools?", "groq", "llama3", max_retries=0)
    # Truncated completions are returned as-is (analysis may detect partial mention)
    assert result == truncated_content


# ---------------------------------------------------------------------------
# S5-07: One failing model does not stop others
# ---------------------------------------------------------------------------

def test_one_failing_model_does_not_stop_others():
    """Pipeline resilience: mix of ok and failed results still produces a score."""
    results = [
        # model A: ok
        {"status": "ok", "brand_mentioned": True, "provider": "groq", "model": "a"},
        {"status": "ok", "brand_mentioned": True, "provider": "groq", "model": "a"},
        # model B: all failed
        {"status": "failed", "error": "Rate limit exceeded", "provider": "openrouter", "model": "b"},
        {"status": "failed", "error": "Rate limit exceeded", "provider": "openrouter", "model": "b"},
        # model C: ok
        {"status": "ok", "brand_mentioned": False, "provider": "gemini", "model": "c"},
    ]
    # Score across all models
    score = calculate_visibility_score(results)
    # 2 mentions / 3 successful = 66.7%
    assert abs(score - (2 / 3 * 100)) < 0.01

    # Model B score should be None (no successful calls)
    from analysis import calculate_model_scores
    model_scores = calculate_model_scores(results)
    assert model_scores.get("openrouter:b", {}).get("score") is None
    assert model_scores["groq:a"]["score"] == 100.0
    assert model_scores["gemini:c"]["score"] == 0.0


# ---------------------------------------------------------------------------
# S5-08: Analyzer down → SOV 'Not available', never 100%
# ---------------------------------------------------------------------------

def test_analyzer_down_sov_not_available():
    """When analyzer fails for >30% of answers, share_of_voice_pct must be None."""
    # 10 answers, 7 analyzer failures (70% > threshold of 30%)
    results = []
    for i in range(3):
        results.append({
            "status": "ok",
            "brand_mentioned": True,
            "mention_count": 1,
            "position": 1,
            "sentiment": "positive",
            "analyzer_status": "ok",
            "competitors": ["Competitor A"],
        })
    for i in range(7):
        results.append({
            "status": "ok",
            "brand_mentioned": True,
            "mention_count": 1,
            "position": 1,
            "sentiment": "unknown",
            "analyzer_status": "failed",
            "competitors": [],
        })

    metrics = compute_brand_report_metrics(results, target_brand="TestBrand")
    assert metrics["analysis_incomplete"] is True
    assert metrics["share_of_voice_pct"] is None
    assert metrics["sov_available"] is False


def test_analyzer_down_sov_never_100():
    """
    Ensure SOV can never be 100% when analyzer is down even if the only named
    brand in competitors is the target brand itself.
    """
    results = [
        {
            "status": "ok",
            "brand_mentioned": True,
            "mention_count": 1,
            "position": 1,
            "sentiment": "unknown",
            "analyzer_status": "failed",
            "competitors": [],  # no competitors extracted (analyzer failed)
        }
        for _ in range(5)
    ]
    # All 5 answers fail analyzer — analysis_incomplete = True
    metrics = compute_brand_report_metrics(results, target_brand="MyBrand")
    assert metrics["analysis_incomplete"] is True
    # Share of voice must be None (not 100%)
    assert metrics["share_of_voice_pct"] is None


# ---------------------------------------------------------------------------
# S5-09: Error messages contain no raw keys or provider JSON
# ---------------------------------------------------------------------------

def test_error_messages_no_raw_api_keys_or_json():
    """summarize_analyzer_error must not echo raw exception objects with keys."""
    # Simulate an error that might contain JSON or key fragments
    raw_exc = ConfigurationError(
        "Missing API key for provider 'groq'. Please set GROQ_API_KEY."
    )
    summary = summarize_analyzer_error(raw_exc)
    # Should give a clean user-facing reason
    assert "API key" in summary or "key" in summary.lower()
    # Must not contain raw JSON objects
    assert "{" not in summary

    rate_exc = RateLimitExceededError(
        "Rate limit exceeded on groq:llama3 after 3 retries: 429 Too Many Requests"
    )
    summary2 = summarize_analyzer_error(rate_exc)
    assert "rate limit" in summary2.lower() or "quota" in summary2.lower()
    # Should not bleed internal retry JSON
    assert '{"error"' not in summary2


def test_error_message_length_capped():
    """Error messages must not exceed 160 characters."""
    # Simulate a very long raw error
    long_err = LLMClientError("x" * 500)
    summary = summarize_analyzer_error(long_err)
    assert len(summary) <= 160


# ---------------------------------------------------------------------------
# S5-10: Sample dataset never takes on the sidebar brand name
# ---------------------------------------------------------------------------

def test_sample_dataset_always_uses_sample_brand():
    """The sample brand is always SAMPLE_BRAND regardless of configured_brand."""
    assert resolve_display_brand(is_sample=True, configured_brand="Nike") == SAMPLE_BRAND
    assert resolve_display_brand(is_sample=True, configured_brand="Asana") == SAMPLE_BRAND
    assert resolve_display_brand(is_sample=True, configured_brand="") == SAMPLE_BRAND


def test_non_sample_uses_configured_brand():
    assert resolve_display_brand(is_sample=False, configured_brand="Nike") == "Nike"
    assert resolve_display_brand(is_sample=False, configured_brand="") == ""


def test_load_sample_state_does_not_set_scan_brand():
    """load_sample_into_state must not overwrite scan_brand key."""
    state = {"scan_brand": "Nike", "scan_id": 99}
    sample_results = [
        {"brand_mentioned": True, "prompt": "test"},
    ]
    load_sample_into_state(state, sample_results)

    # is_sample is set, scan_id cleared, but scan_brand untouched
    assert state["is_sample"] is True
    assert state["scan_id"] is None
    assert state["scan_brand"] == "Nike"  # unchanged — sample never overwrites this


def test_sample_results_are_deep_copied():
    """load_sample_into_state must deep-copy so mutation of state doesn't affect source."""
    original = [{"brand_mentioned": True, "prompt": "test"}]
    state: dict = {}
    load_sample_into_state(state, original)
    state["scan_results"][0]["brand_mentioned"] = False
    assert original[0]["brand_mentioned"] is True  # unchanged


# ---------------------------------------------------------------------------
# S5-11: Failed call is never counted as "not mentioned"
# ---------------------------------------------------------------------------

def test_failed_call_never_counted_as_not_mentioned():
    """A failed call must be excluded from the denominator entirely."""
    results = [
        {"status": "failed", "brand_mentioned": False, "error": "timeout"},
        {"status": "failed", "brand_mentioned": False, "error": "rate limit"},
    ]
    score = calculate_visibility_score(results)
    assert score is None  # No successful calls → no score, not 0%


def test_zero_denominator_returns_none():
    """If all calls fail, score is None — not 0.0."""
    assert calculate_visibility_score([]) is None
    failed_results = [{"status": "failed"} for _ in range(5)]
    assert calculate_visibility_score(failed_results) is None
