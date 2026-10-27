import database
from streamlit.testing.v1 import AppTest

from analysis import compute_call_accounting, count_result_statuses
from database import (
    create_scan,
    get_scan,
    get_scan_status_counts,
    init_db,
    save_result,
)


def test_reconciles_when_completed_plus_failed_equals_expected():
    accounting = compute_call_accounting(9, 8, 1)

    assert accounting["expected"] == 9
    assert accounting["completed"] == 8
    assert accounting["failed"] == 1
    assert accounting["missing"] == 0
    assert accounting["reconciles"] is True
    assert accounting["warning"] is None


def test_missing_calls_trigger_a_warning():
    accounting = compute_call_accounting(9, 8, 0)

    assert accounting["missing"] == 1
    assert accounting["reconciles"] is False
    assert "expected 9 calls" in accounting["warning"]
    assert "8 completed" in accounting["warning"]
    assert "1 missing" in accounting["warning"]


def test_count_result_statuses():
    results = [{"status": "ok"}, {"status": "failed"}, {"status": "ok"}]

    assert count_result_statuses(results) == {
        "completed": 2,
        "failed": 1,
        "total": 3,
    }


def test_scan_stores_expected_calls_and_counts_statuses(tmp_path):
    db_path = str(tmp_path / "accounting.db")
    init_db(db_path)

    scan_id = create_scan("Acme", expected_calls=6, db_path=db_path)
    for i in range(4):
        save_result(
            scan_id=scan_id,
            prompt=f"Prompt {i}",
            ai_response="Acme" if i < 3 else None,
            brand_mentioned=i < 3,
            mention_count=0,
            sentiment="neutral",
            competitors=[],
            provider="groq",
            model="llama",
            run_index=1,
            status="failed" if i == 3 else "ok",
            db_path=db_path,
        )

    assert get_scan(scan_id, db_path=db_path)["expected_calls"] == 6

    counts = get_scan_status_counts(db_path=db_path)[scan_id]
    assert counts["completed"] == 3
    assert counts["failed"] == 1

    accounting = compute_call_accounting(
        6, counts["completed"], counts["failed"]
    )
    assert accounting["missing"] == 2
    assert accounting["warning"] is not None


def test_history_shows_accounting_columns_and_warns(tmp_path, monkeypatch):
    db_path = str(tmp_path / "history.db")
    init_db(db_path)

    scan_id = create_scan("Acme", expected_calls=6, db_path=db_path)
    for i in range(4):
        save_result(
            scan_id=scan_id,
            prompt=f"Prompt {i}",
            ai_response="Acme",
            brand_mentioned=True,
            mention_count=1,
            sentiment="positive",
            competitors=[],
            provider="groq",
            model="llama",
            run_index=1,
            status="failed" if i == 3 else "ok",
            db_path=db_path,
        )

    monkeypatch.setattr(database, "get_db_path", lambda: db_path)

    at = AppTest.from_file("app.py", default_timeout=90)
    at.session_state["scan_brand"] = "Acme"
    at.session_state["is_sample"] = False
    at.run()
    assert not at.exception

    markdown = "\n".join(m.value for m in at.markdown)
    for heading in ("Expected", "Completed", "Failed", "Missing"):
        assert f"{heading}</th>" in markdown

    warnings = "\n".join(w.value for w in at.warning)
    assert "Scan #1" in warnings
    assert "does not reconcile" in warnings
    assert "expected 6 calls" in warnings
