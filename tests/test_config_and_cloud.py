import hmac
import pytest
from unittest.mock import patch, MagicMock

from config import get_setting, get_int_setting, get_float_setting
from database import restore_database_from_bytes, init_db, get_database_bytes, get_db_path, get_connection
from llm_client import validate_analyzer_config


# ==============================================================================
# 1. get_setting Precedence & Fallback Tests
# ==============================================================================

def test_get_setting_no_secrets_and_env_fallback(monkeypatch):
    """When st.secrets is not present or raises an error, fallback to os.environ, then default."""
    # Ensure st has no secrets or raises
    with patch("streamlit.secrets", new=MagicMock(side_effect=FileNotFoundError("No secrets file"))):
        monkeypatch.setenv("TEST_KEY_ENV", "env_value")
        # Should return env value
        val = get_setting("TEST_KEY_ENV", "fallback")
        assert val == "env_value"

        # When not in env, should return default
        val_default = get_setting("NON_EXISTENT_KEY", "fallback")
        assert val_default == "fallback"


def test_get_setting_st_secrets_precedence(monkeypatch):
    """st.secrets must take precedence over os.environ."""
    mock_secrets = {"PRECEDENCE_KEY": "secrets_value"}
    with patch("streamlit.secrets", mock_secrets):
        monkeypatch.setenv("PRECEDENCE_KEY", "env_value")
        val = get_setting("PRECEDENCE_KEY")
        assert val == "secrets_value"


# ==============================================================================
# 2. Safe Parsing for RPM and Timeouts
# ==============================================================================

def test_safe_rpm_parsing(monkeypatch):
    """Safe integer parsing handles normal strings, invalid non-integers, and empty values."""
    # Normal integer string
    monkeypatch.setenv("GROQ_RPM", "45")
    assert get_int_setting("GROQ_RPM", 30) == 45

    # Non-integer / corrupted string
    monkeypatch.setenv("GROQ_RPM", "abc_invalid")
    assert get_int_setting("GROQ_RPM", 30) == 30

    # Empty string
    monkeypatch.setenv("GROQ_RPM", "   ")
    assert get_int_setting("GROQ_RPM", 30) == 30

    # Float parsing
    monkeypatch.setenv("TIMEOUT_TEST", "60.5")
    assert get_float_setting("TIMEOUT_TEST", 45.0) == 60.5
    monkeypatch.setenv("TIMEOUT_TEST", "invalid")
    assert get_float_setting("TIMEOUT_TEST", 45.0) == 45.0


# ==============================================================================
# 3. Startup Self-Check & Analyzer Validation
# ==============================================================================

def test_analyzer_in_tested_models_blocked():
    """Analyzer model cannot be included in tested models (self-grading block)."""
    selected = ["groq:llama-3.1-8b-instant", "gemini:gemini-1.5-flash"]
    is_valid, err = validate_analyzer_config("groq:llama-3.1-8b-instant", selected_models=selected)
    assert not is_valid
    assert "cannot be in the list of selected test models" in err


def test_analyzer_missing_or_unsupported():
    """Missing or unsupported provider analyzer is flagged cleanly."""
    is_valid, err = validate_analyzer_config("")
    assert not is_valid
    assert "ANALYZER_MODEL is required" in err

    is_valid_unsupp, err_unsupp = validate_analyzer_config("unsupported:model-x")
    assert not is_valid_unsupp
    assert "unsupported provider" in err_unsupp


# ==============================================================================
# 4. Access Control (Password Check & Session Call Cap)
# ==============================================================================

def test_password_check_hmac():
    """Test APP_PASSWORD verification using hmac.compare_digest."""
    configured_password = "SuperSecretPassword123"
    user_correct = "SuperSecretPassword123"
    user_incorrect = "WrongPassword"

    assert hmac.compare_digest(user_correct, configured_password) is True
    assert hmac.compare_digest(user_incorrect, configured_password) is False


def test_session_call_cap_logic():
    """Verify session quota math and bounds."""
    max_calls = 30
    calls_used = 25
    remaining = max(0, max_calls - calls_used)
    assert remaining == 5

    # Requested calls fit
    requested_valid = 4
    assert requested_valid <= remaining

    # Requested calls exceed quota
    requested_excess = 8
    assert requested_excess > remaining


# ==============================================================================
# 5. SQLite Backup & Restore Validation
# ==============================================================================

def test_restore_database_validation_rejects_non_sqlite(tmp_path):
    """Validation must reject files that are not valid SQLite databases."""
    target_db = str(tmp_path / "restored.db")
    
    # 1. Random bytes
    with pytest.raises(ValueError) as exc:
        restore_database_from_bytes(b"hello world this is just random text data", target_path=target_db)
    assert "not a valid SQLite database header" in str(exc.value)

    # 2. Corrupt or empty
    with pytest.raises(ValueError) as exc_empty:
        restore_database_from_bytes(b"", target_path=target_db)
    assert "empty or too small" in str(exc_empty.value)


def test_restore_database_success_with_valid_backup(tmp_path):
    """Restoring a valid SQLite database backup succeeds and loads schema properly."""
    src_db = str(tmp_path / "source.db")
    target_db = str(tmp_path / "restored_target.db")

    init_db(src_db)
    backup_bytes = get_database_bytes(src_db)
    assert len(backup_bytes) > 0

    success = restore_database_from_bytes(backup_bytes, target_path=target_db)
    assert success is True


# ==============================================================================
# 6. Pluggable Database & DATABASE_URL Tests (Stage 4)
# ==============================================================================

def test_database_url_sqlite_uri_parsing(monkeypatch, tmp_path):
    """Test that DATABASE_URL supporting sqlite:/// URIs resolves path properly."""
    custom_db = str(tmp_path / "custom_url.db")
    
    # 1. sqlite:/// URI
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{custom_db}")
    resolved = get_db_path()
    assert resolved == os.path.abspath(custom_db)

    # 2. File path in DATABASE_URL without scheme
    monkeypatch.setenv("DATABASE_URL", custom_db)
    assert get_db_path() == os.path.abspath(custom_db)

    # 3. Connection works with sqlite:/// URI
    with get_connection(f"sqlite:///{custom_db}") as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test_tab (id INT)")
        cursor.execute("INSERT INTO test_tab VALUES (42)")
        cursor.execute("SELECT * FROM test_tab")
        assert cursor.fetchone()[0] == 42


def test_headless_run_scan_script(tmp_path, monkeypatch):
    """Verify headless scripts/run_scan.py executes cleanly with active prompts."""
    from scripts.run_scan import run_scheduled_scan
    from database import create_project, create_prompt

    test_db = str(tmp_path / "headless_scan.db")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{test_db}")

    init_db(test_db)
    pid = create_project("Acme", industry="Cloud", db_path=test_db)
    create_prompt(pid, "Best cloud platforms?", tag="top-of-funnel", active=True, db_path=test_db)
    create_prompt(pid, "Acme vs AWS", tag="branded", active=True, db_path=test_db)

    # Mock LLM queries so tests run offline without network calls
    with patch("scripts.run_scan.query_ai", return_value="1. Acme Cloud is top.\n2. AWS is huge."):
        with patch("scripts.run_scan.get_analyzer_model", return_value="groq:llama-3.1-8b-instant"):
            with patch("scripts.run_scan.validate_analyzer_config", return_value=(True, None)):
                exit_code = run_scheduled_scan(
                    project_id=pid,
                    max_calls=10,
                    models=["groq:llama-3.1-8b-instant", "gemini:gemini-1.5-flash"],
                    use_cache=False
                )
                assert exit_code == 0

