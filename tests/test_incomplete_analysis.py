import database
from streamlit.testing.v1 import AppTest

from analysis import (
    compute_brand_report_metrics,
    incomplete_analysis_message,
    is_analysis_incomplete,
)
from database import create_scan, init_db, save_result


def _row(
    prompt,
    analyzer_status="ok",
    brand_mentioned=True,
    competitors=None,
    sentiment="positive",
):
    return {
        "status": "ok",
        "prompt": prompt,
        "provider": "groq",
        "model": "llama",
        "brand_mentioned": brand_mentioned,
        "mention_count": 1 if brand_mentioned else 0,
        "position": 1 if brand_mentioned else None,
        "sentiment": sentiment,
        "competitors": competitors or [],
        "analyzer_status": analyzer_status,
        "analyzer_error": "429 rate limited" if analyzer_status == "failed" else None,
        "scan_id": 1,
        "scan_created_at": "2026-10-01",
    }


def test_threshold_is_strictly_more_than_thirty_percent():
    assert is_analysis_incomplete(3, 10) is False   # exactly 30% is not "more than"
    assert is_analysis_incomplete(4, 10) is True
    assert is_analysis_incomplete(0, 0) is False


def test_banner_text_names_the_counts():
    assert incomplete_analysis_message(4, 10) == (
        "Competitor and sentiment data is incomplete "
        "(4 of 10 answers). Fix the analyzer and rerun."
    )


def test_incomplete_analysis_withholds_competitor_and_sentiment_metrics():
    results = [
        _row(
            f"Prompt {i}",
            analyzer_status="failed" if i < 4 else "ok",
            competitors=[] if i < 4 else ["Salesforce"],
        )
        for i in range(10)
    ]

    metrics = compute_brand_report_metrics(results, target_brand="HubSpot")

    assert metrics["analysis_incomplete"] is True
    assert metrics["analyzer_failures"] == 4
    assert metrics["analysis_incomplete_message"] == incomplete_analysis_message(4, 10)
    assert metrics["share_of_voice_pct"] is None
    assert metrics["sov_available"] is False
    assert metrics["sentiment_available"] is False
    assert all(row["share_of_mentions_pct"] is None for row in metrics["comparison_table"])


def test_boundary_case_keeps_metrics_available():
    results = [
        _row(
            f"Prompt {i}",
            analyzer_status="failed" if i < 3 else "ok",
            brand_mentioned=i < 7,
            competitors=["Salesforce"],
        )
        for i in range(10)
    ]

    metrics = compute_brand_report_metrics(results, target_brand="HubSpot")

    assert metrics["analysis_incomplete"] is False
    assert metrics["share_of_voice_pct"] == 70.0  # 7 of 10 answers
    assert metrics["sentiment_available"] is True
    # Brand mentioned in 7 answers, but the analyzer only graded 4 of them.
    assert metrics["sentiment_evaluated"] == 4


def test_report_renders_not_available_states(tmp_path, monkeypatch):
    db_path = str(tmp_path / "incomplete_report.db")
    init_db(db_path)

    scan_id = create_scan(
        "HubSpot", industry="CRM", expected_calls=10, db_path=db_path
    )
    for i in range(10):
        analyzer_failed = i < 4
        save_result(
            scan_id=scan_id,
            prompt=f"Best CRM question {i}",
            ai_response="HubSpot and Salesforce are options.",
            brand_mentioned=True,
            mention_count=1,
            sentiment="positive",
            competitors=[] if analyzer_failed else ["Salesforce"],
            provider="groq",
            model="llama",
            run_index=1,
            position=1,
            analyzer_status="failed" if analyzer_failed else "ok",
            analyzer_error="429 rate limited" if analyzer_failed else None,
            db_path=db_path,
        )

    monkeypatch.setattr(database, "get_db_path", lambda: db_path)

    at = AppTest.from_file("app.py", default_timeout=90)
    at.session_state["scan_brand"] = "HubSpot"
    at.session_state["is_sample"] = False
    at.run()
    assert not at.exception

    rendered = "\n".join(m.value for m in at.markdown)
    rendered += "\n" + "\n".join(info.value for info in at.info)

    assert "Not available" in rendered
    assert "Not evaluated" in rendered
    assert "Can't be calculated without competitor data" in rendered

    warnings = "\n".join(w.value for w in at.warning)
    assert (
        "Competitor and sentiment data is incomplete (4 of 10 answers). "
        "Fix the analyzer and rerun."
    ) in warnings
