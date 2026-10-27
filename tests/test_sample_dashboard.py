import database
from streamlit.testing.v1 import AppTest

from sample_data import (
    SAMPLE_BRAND,
    SAMPLE_DATA_LABEL,
    load_sample_into_state,
    resolve_display_brand,
)


def test_sample_brand_is_independent_of_the_sidebar_brand():
    assert resolve_display_brand(True, "Nike") == SAMPLE_BRAND
    assert resolve_display_brand(True, "") == SAMPLE_BRAND
    assert resolve_display_brand(False, "  Nike  ") == "Nike"


def test_sample_label_states_no_api_calls_were_made():
    assert SAMPLE_DATA_LABEL == "Sample data, no API calls were made."


def test_loading_sample_preserves_brand_and_clears_scan_id():
    source = [{"prompt": "P", "brand_mentioned": True}]
    state = {
        "scan_brand": "Nike",
        "scan_id": 7,
        "is_sample": False,
        "scan_results": [],
    }

    load_sample_into_state(state, source)

    assert state["scan_brand"] == "Nike"
    assert state["scan_id"] is None
    assert state["is_sample"] is True
    assert state["scan_results"] == source
    assert state["scan_results"] is not source  # deep copy, not a shared list


def test_app_load_sample_keeps_sidebar_brand_and_writes_no_scan(tmp_path, monkeypatch):
    db_path = str(tmp_path / "sample_dashboard.db")
    database.init_db(db_path)
    monkeypatch.setattr(database, "get_db_path", lambda: db_path)

    at = AppTest.from_file("app.py", default_timeout=90)
    at.session_state["scan_brand"] = "Nike"
    at.session_state["is_sample"] = False
    at.session_state["scan_results"] = [{
        "provider": "groq",
        "model": "llama",
        "prompt": "Best running shoes?",
        "run_index": 1,
        "ai_response": "Nike is a popular choice.",
        "brand_mentioned": True,
        "mention_count": 1,
        "sentiment": "positive",
        "position": 1,
        "competitors": [],
        "status": "ok",
        "error": None,
        "analyzer_status": "ok",
        "analyzer_error": None,
    }]
    at.run()
    assert not at.exception

    load_buttons = [b for b in at.button if b.label == "Load sample dashboard"]
    assert load_buttons, "the sample dashboard must be loadable"
    load_buttons[0].click()
    at.run()
    assert not at.exception

    # The demo dataset keeps its own brand; the sidebar brand is untouched.
    assert at.session_state["scan_brand"] == "Nike"
    assert at.session_state["is_sample"] is True
    brand_input = [
        ti for ti in at.text_input
        if ti.label == "Brand name" and ti.key is None
    ][0]
    assert brand_input.value == "Nike"

    # Explicit label, and the demo brand is what the dashboard reports.
    assert any(info.value == SAMPLE_DATA_LABEL for info in at.info)
    markdown = "\n".join(m.value for m in at.markdown)
    assert f"{SAMPLE_BRAND} (you)" in markdown
    # Share of voice reports its answer denominator (n) on the demo view.
    assert "n=8" in markdown

    # Loading a view must never create a scan or results.
    assert database.get_all_scans() == []
