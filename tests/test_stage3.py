from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

from app import execute_query_unit
from llm_client import LLMClientError


def test_one_failing_provider_does_not_break_scan():
    """Verify that when one provider/model raises an error, other models complete normally."""
    query_units = [
        ("Prompt 1", "groq:llama-3.1-8b-instant", 1),
        ("Prompt 1", "openrouter:flaky-model:free", 1)
    ]

    def mock_query(prompt, provider, model, **kwargs):
        if provider == "openrouter":
            raise LLMClientError("503 Service Unavailable on OpenRouter free tier")
        return "HubSpot is a leading CRM for startups."

    def mock_sentiment_comps(*args, **kwargs):
        return {"sentiment": "positive", "competitors": ["Salesforce"]}

    with patch("app.query_ai", side_effect=mock_query), \
         patch("app.analyze_sentiment_and_competitors", side_effect=mock_sentiment_comps):

        results = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    execute_query_unit,
                    prompt=p,
                    pm_str=pm,
                    run_idx=r,
                    brand_name="HubSpot",
                    aliases=[],
                    known_competitors=[],
                    analyzer_model="groq:llama-3.1-8b-instant",
                    use_cache=False
                )
                for (p, pm, r) in query_units
            ]
            for f in as_completed(futures):
                results.append(f.result())

        assert len(results) == 2

        # Check groq succeeded
        groq_res = next(r for r in results if r["provider"] == "groq")
        assert groq_res["status"] == "ok"
        assert groq_res["brand_mentioned"] is True
        assert groq_res["error"] is None

        # Check openrouter recorded as failed without stopping execution
        or_res = next(r for r in results if r["provider"] == "openrouter")
        assert or_res["status"] == "failed"
        assert "503 Service Unavailable" in or_res["error"]
        assert or_res["brand_mentioned"] is False


def test_cache_hit_skips_query_ai():
    """Verify that when a cached response exists and use_cache is True, query_ai is never called."""
    with patch("app.get_cached_response", return_value="Cached answer about Nike."), \
         patch("app.query_ai") as mock_query, \
         patch("app.analyze_sentiment_and_competitors", return_value={"sentiment": "positive", "competitors": []}):

        res = execute_query_unit(
            prompt="Best shoes?",
            pm_str="groq:llama-3.1-8b-instant",
            run_idx=1,
            brand_name="Nike",
            aliases=[],
            known_competitors=[],
            analyzer_model="groq:llama-3.1-8b-instant",
            use_cache=True
        )

        assert mock_query.call_count == 0
        assert res["status"] == "ok"
        assert res["ai_response"] == "Cached answer about Nike."
        assert res["brand_mentioned"] is True
        assert res["cached"] is True
