from unittest.mock import patch

from llm_client import (
    ConfigurationError,
    LLMClientError,
    ModelUnavailableError,
    RateLimitExceededError,
    analyzer_health_gate_message,
    probe_analyzer_model,
    summarize_analyzer_error,
)


def test_probe_success_reports_latency_and_excerpt():
    with patch("llm_client.query_ai", return_value="OK") as mocked:
        health = probe_analyzer_model("groq:llama-3.1-8b-instant")

    assert health["ok"] is True
    assert health["reason"] is None
    assert health["latency_ms"] is not None
    assert health["response_excerpt"] == "OK"
    assert health["model"] == "groq:llama-3.1-8b-instant"
    mocked.assert_called_once()


def test_probe_is_one_tiny_call_without_retries():
    with patch("llm_client.query_ai", return_value="OK") as mocked:
        probe_analyzer_model("groq:some-model")

    args, kwargs = mocked.call_args
    assert kwargs.get("max_retries") == 0
    assert len(args[0]) < 60  # one short prompt, not a real grading call


def test_probe_maps_failures_to_short_reasons():
    with patch("llm_client.query_ai", side_effect=RateLimitExceededError("HTTP 429 after retries")):
        health = probe_analyzer_model("groq:m")
    assert health["ok"] is False
    assert "rate limited" in health["reason"].lower()
    assert health["response_excerpt"] is None

    with patch("llm_client.query_ai", side_effect=ModelUnavailableError("HTTP 404 model gone")):
        health = probe_analyzer_model("groq:m")
    assert "unavailable" in health["reason"].lower()

    with patch("llm_client.query_ai", side_effect=ConfigurationError("missing key")):
        health = probe_analyzer_model("groq:m")
    assert "api key" in health["reason"].lower()


def test_probe_rejects_missing_or_malformed_analyzer_configuration():
    missing = probe_analyzer_model(None)
    assert missing["ok"] is False
    assert "not set" in missing["reason"].lower()

    malformed = probe_analyzer_model("not-a-provider-model")
    assert malformed["ok"] is False
    assert "provider:model" in malformed["reason"]

    unsupported = probe_analyzer_model("cohere:command-r")
    assert unsupported["ok"] is False
    assert "unsupported" in unsupported["reason"].lower()


def test_probe_truncates_long_errors():
    reason = summarize_analyzer_error(LLMClientError("x" * 500))
    assert len(reason) <= 160
    assert reason.endswith("...")


def test_gate_requires_confirmation_after_a_failure():
    health = {"ok": False, "reason": "Analyzer is rate limited (quota exhausted)."}

    message = analyzer_health_gate_message(health, confirmed=False)
    assert message is not None
    assert "rate limited" in message.lower()
    assert "confirm" in message.lower()

    # Explicit confirmation lets the scan proceed.
    assert analyzer_health_gate_message(health, confirmed=True) is None
    # A healthy analyzer never blocks.
    assert analyzer_health_gate_message({"ok": True}, confirmed=False) is None
    # No probe result yet also does not block (the probe itself is the gate).
    assert analyzer_health_gate_message(None, confirmed=False) is None


def test_failed_probe_blocks_the_scan_until_confirmed(tmp_path, monkeypatch):
    """End to end: no model call happens while the analyzer is unconfirmed."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    import config
    import database
    import llm_client

    calls = {"probe": 0, "query": 0}

    monkeypatch.setattr(
        config, "get_setting",
        lambda key, default="": "" if key in ("APP_PASSWORD", "ANALYZER_MODEL") else default,
    )
    monkeypatch.setattr(config, "get_int_setting", lambda key, default=0: default)
    monkeypatch.setattr(llm_client, "get_analyzer_model", lambda: "groq:stub-analyzer")
    monkeypatch.setattr(llm_client, "get_available_models", lambda: ["groq:a", "groq:b"])
    monkeypatch.setattr(
        llm_client, "validate_analyzer_config", lambda *args, **kwargs: (True, None)
    )

    def fake_probe(model):
        calls["probe"] += 1
        return {
            "ok": False,
            "reason": "Analyzer is rate limited (quota exhausted).",
            "model": model,
            "latency_ms": 5,
            "response_excerpt": None,
        }

    def fail_query(*args, **kwargs):
        calls["query"] += 1
        raise llm_client.LLMClientError("simulated 503")

    monkeypatch.setattr(llm_client, "probe_analyzer_model", fake_probe)
    monkeypatch.setattr(llm_client, "query_ai", fail_query)

    db_path = str(tmp_path / "gate.db")
    database.init_db(db_path)
    monkeypatch.setattr(database, "get_db_path", lambda: db_path)
    st.cache_data.clear()
    st.cache_resource.clear()

    at = AppTest.from_file("app.py", default_timeout=90)
    at.session_state["scan_brand"] = "Acme"
    at.session_state["is_sample"] = False
    at.session_state["scan_results"] = []
    at.run()

    [area for area in at.text_area if area.label.startswith("Prompts")][0].set_value(
        "Best CRM?\nTop tools?"
    )
    at.run()

    [button for button in at.button if button.label == "Run scan"][0].click()
    at.run()

    # Blocked: one probe, zero model calls, no scan saved, reason shown.
    assert calls == {"probe": 1, "query": 0}
    assert database.get_all_scans() == []
    assert any(
        "Confirm in the sidebar" in error.value for error in at.error
    )
    assert any(
        checkbox.key == "analyzer_override_confirmed" for checkbox in at.checkbox
    )

    # Confirming lets the scan run; the accounting still reconciles exactly.
    at.checkbox(key="analyzer_override_confirmed").check()
    at.run()
    [button for button in at.button if button.label == "Run scan"][0].click()
    at.run()

    scans = database.get_all_scans()
    assert len(scans) == 1
    assert scans[0]["expected_calls"] == 4  # 2 prompts x 2 models x 1 run
    assert calls["query"] == 4

    rows = database.get_scan_results(scans[0]["id"])
    assert len(rows) == 4
    assert all(row["status"] == "failed" for row in rows)
