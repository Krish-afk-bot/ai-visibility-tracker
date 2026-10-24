import sqlite3
import os
import pytest
from unittest.mock import patch, MagicMock
from openai import RateLimitError, APIStatusError, APITimeoutError
import httpx

from database import (
    init_db,
    create_scan,
    update_scan_score,
    save_result,
    get_scan_results,
    get_cached_response,
    set_cached_response,
    get_all_scans,
    get_connection,
    create_project,
    get_project,
    get_projects,
    update_project,
    delete_project,
    create_prompt,
    get_prompts,
    update_prompt,
    toggle_prompt_active,
    delete_prompt,
    bulk_add_prompts,
    VALID_PROMPT_TAGS
)
from llm_client import (
    query_ai,
    get_available_models,
    parse_provider_model,
    client_manager,
    ConfigurationError,
    ModelUnavailableError,
    RateLimitExceededError,
    LLMClientError
)


# ==========================================
# Database & Migration Tests
# ==========================================

def test_database_migration_on_old_schema(tmp_path):
    """Test idempotent migration on a database with the exact old schema."""
    test_db = str(tmp_path / "test_old_schema.db")
    
    # 1. Create the legacy database structure manually
    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand_name TEXT NOT NULL,
            industry TEXT,
            visibility_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE scan_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id INTEGER NOT NULL,
            prompt TEXT NOT NULL,
            ai_response TEXT,
            brand_mentioned INTEGER,
            mention_count INTEGER,
            sentiment TEXT,
            competitors TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (scan_id) REFERENCES scans(id)
        )
    """)
    
    # Insert legacy record
    cursor.execute("INSERT INTO scans (brand_name, industry, visibility_score) VALUES ('Acme', 'Tech', 100.0)")
    scan_id = cursor.lastrowid
    cursor.execute("""
        INSERT INTO scan_results (scan_id, prompt, ai_response, brand_mentioned, mention_count, sentiment, competitors)
        VALUES (?, 'Best software?', 'Acme is top.', 1, 1, 'positive', '["Beta"]')
    """, (scan_id,))
    conn.commit()
    conn.close()

    # 2. Run init_db migration on the existing legacy DB
    init_db(test_db)

    # 3. Verify columns were added without losing existing rows
    with get_connection(test_db) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(scan_results)")
        cols = {row["name"]: row for row in cursor.fetchall()}
        
        # Check newly migrated columns exist
        assert "provider" in cols
        assert "model" in cols
        assert "run_index" in cols
        assert "status" in cols
        assert "error" in cols
        assert "position" in cols

        # Check existing row was preserved and defaulted properly
        cursor.execute("SELECT * FROM scan_results WHERE id = 1")
        old_row = dict(cursor.fetchone())
        assert old_row["prompt"] == "Best software?"
        assert old_row["provider"] == "gemini"
        assert old_row["status"] == "ok"
        assert old_row["model"] == "gemini-2.0-flash"

    # 4. Running init_db again is completely idempotent
    init_db(test_db)


def test_database_crud_and_cache(tmp_path):
    """Test full CRUD operations on scan, scan_results, and cache."""
    test_db = str(tmp_path / "test_tracker.db")
    init_db(test_db)

    # Create scan
    scan_id = create_scan("Nike", "Sportswear", db_path=test_db)
    assert scan_id == 1

    # Save results
    save_result(
        scan_id=scan_id,
        prompt="Running shoes",
        ai_response="Nike Air Zoom is great.",
        brand_mentioned=True,
        mention_count=1,
        sentiment="positive",
        competitors=["Adidas", "Puma"],
        provider="groq",
        model="llama-3.1-8b-instant",
        run_index=1,
        status="ok",
        position=1,
        db_path=test_db
    )

    # Failed result test
    save_result(
        scan_id=scan_id,
        prompt="Running shoes 2",
        ai_response=None,
        brand_mentioned=False,
        mention_count=0,
        sentiment="neutral",
        competitors=[],
        provider="groq",
        model="llama-3.1-8b-instant",
        run_index=1,
        status="failed",
        error="Rate limit exceeded",
        position=None,
        db_path=test_db
    )

    update_scan_score(scan_id, 50.0, db_path=test_db)

    results = get_scan_results(scan_id, db_path=test_db)
    assert len(results) == 2
    assert results[0]["brand_mentioned"] == 1
    assert results[0]["competitors"] == ["Adidas", "Puma"]
    assert results[0]["position"] == 1
    assert results[1]["status"] == "failed"
    assert results[1]["error"] == "Rate limit exceeded"

    scans = get_all_scans(db_path=test_db)
    assert len(scans) == 1
    assert scans[0]["brand_name"] == "Nike"
    assert scans[0]["visibility_score"] == 50.0

    # Cache tests
    assert get_cached_response("groq", "llama-3.1-8b-instant", "best shoes", 1, db_path=test_db) is None
    set_cached_response("groq", "llama-3.1-8b-instant", "best shoes", "Cached answer", 1, db_path=test_db)
    cached = get_cached_response("groq", "llama-3.1-8b-instant", "best shoes", 1, db_path=test_db)
    assert cached == "Cached answer"


# ==========================================
# LLM Client Tests
# ==========================================

def test_parse_provider_model():
    p, m = parse_provider_model("groq:llama-3.1-8b-instant")
    assert p == "groq"
    assert m == "llama-3.1-8b-instant"

    p2, m2 = parse_provider_model("openrouter:meta-llama/llama-3-8b:free")
    assert p2 == "openrouter"
    assert m2 == "meta-llama/llama-3-8b:free"

    with pytest.raises(ValueError):
        parse_provider_model("invalid_string_no_colon")


def test_get_available_models(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_or_key")
    monkeypatch.setenv("OPENROUTER_MODELS", "m1:free, m2:free")
    monkeypatch.setenv("GROQ_API_KEY", "test_groq_key")
    monkeypatch.setenv("GROQ_MODELS", "llama3, mixtral")

    available = get_available_models()
    assert "openrouter:m1:free" in available
    assert "openrouter:m2:free" in available
    assert "groq:llama3" in available
    assert "groq:mixtral" in available

    # Test unavailable when key missing
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    available2 = get_available_models()
    assert not any(m.startswith("openrouter:") for m in available2)
    assert "groq:llama3" in available2


def test_missing_key_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client_manager.reset_clients()

    with pytest.raises(ConfigurationError) as exc_info:
        query_ai("hello", "groq", "llama3")
    assert "Missing API key for provider 'groq'" in str(exc_info.value)


def test_query_ai_success(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "mock_key")
    client_manager.reset_clients()

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Here is an unbiased answer"
    mock_resp.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_resp

    with patch.object(client_manager, "get_client", return_value=mock_client):
        res = query_ai("What is the best shoe?", "groq", "llama-3.1-8b-instant")
        assert res == "Here is an unbiased answer"
        # Confirm user prompt sent as written without wrapper
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["content"] == "What is the best shoe?"


def test_query_ai_retry_on_429_then_succeed(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "mock_key")
    client_manager.reset_clients()

    mock_client = MagicMock()
    
    # Create mock 429 response
    response_429 = httpx.Response(
        status_code=429,
        headers={"Retry-After": "0.01"},
        request=httpx.Request("POST", "https://api.groq.com")
    )
    rate_err = RateLimitError(message="Rate limited", response=response_429, body=None)

    mock_choice = MagicMock()
    mock_choice.message.content = "Recovered response"
    mock_success = MagicMock()
    mock_success.choices = [mock_choice]

    mock_client.chat.completions.create.side_effect = [rate_err, mock_success]

    with patch.object(client_manager, "get_client", return_value=mock_client):
        res = query_ai("Hello", "groq", "llama3", max_retries=2, initial_backoff=0.01)
        assert res == "Recovered response"
        assert mock_client.chat.completions.create.call_count == 2


def test_query_ai_rate_limit_exceeded(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "mock_key")
    client_manager.reset_clients()

    mock_client = MagicMock()
    response_429 = httpx.Response(
        status_code=429,
        request=httpx.Request("POST", "https://api.groq.com")
    )
    rate_err = RateLimitError(message="Rate limited", response=response_429, body=None)
    mock_client.chat.completions.create.side_effect = rate_err

    with patch.object(client_manager, "get_client", return_value=mock_client):
        with pytest.raises(RateLimitExceededError):
            query_ai("Hello", "groq", "llama3", max_retries=1, initial_backoff=0.01)


def test_query_ai_model_retired_404(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock_key")
    client_manager.reset_clients()

    mock_client = MagicMock()
    response_404 = httpx.Response(
        status_code=404,
        request=httpx.Request("POST", "https://openrouter.ai")
    )
    status_err = APIStatusError(message="Model retired", response=response_404, body=None)
    mock_client.chat.completions.create.side_effect = status_err

    with patch.object(client_manager, "get_client", return_value=mock_client):
        with pytest.raises(ModelUnavailableError) as exc_info:
            query_ai("Hello", "openrouter", "retired-model:free", max_retries=1)
        assert "unavailable, not found, or retired" in str(exc_info.value)


# ==========================================
# Projects & Prompts (Stage 1) Tests
# ==========================================

def test_project_crud_and_aliases(tmp_path):
    """Test creating, reading, updating and deleting a project."""
    test_db = str(tmp_path / "test_projects.db")
    init_db(test_db)

    # 1. Create project
    proj_id = create_project(
        brand_name="Acme Corp",
        aliases=["Acme", "AcmeCorp"],
        industry="SaaS",
        db_path=test_db
    )
    assert proj_id == 1

    # 2. Get project
    proj = get_project(proj_id, db_path=test_db)
    assert proj is not None
    assert proj["brand_name"] == "Acme Corp"
    assert proj["aliases"] == ["Acme", "AcmeCorp"]
    assert proj["industry"] == "SaaS"

    # 3. Update project
    update_project(
        proj_id,
        brand_name="Acme International",
        aliases=["AcmeIntl"],
        industry="Enterprise Software",
        db_path=test_db
    )
    proj_updated = get_project(proj_id, db_path=test_db)
    assert proj_updated["brand_name"] == "Acme International"
    assert proj_updated["aliases"] == ["AcmeIntl"]
    assert proj_updated["industry"] == "Enterprise Software"

    # 4. List projects
    projects = get_projects(db_path=test_db)
    assert len(projects) == 1
    assert projects[0]["id"] == proj_id

    # 5. Delete project
    delete_project(proj_id, db_path=test_db)
    assert get_project(proj_id, db_path=test_db) is None
    assert len(get_projects(db_path=test_db)) == 0


def test_prompts_crud_tags_and_bulk_add(tmp_path):
    """Test prompt library CRUD, tags validation, active toggling, and bulk paste."""
    test_db = str(tmp_path / "test_prompts.db")
    init_db(test_db)

    proj_id = create_project("BrandX", db_path=test_db)

    # 1. Valid tags
    for tag in ["branded", "non-branded", "top-of-funnel", "bottom-of-funnel"]:
        p_id = create_prompt(proj_id, f"Prompt for {tag}", tag=tag, db_path=test_db)
        assert p_id > 0

    # 2. Invalid tag raises ValueError
    with pytest.raises(ValueError):
        create_prompt(proj_id, "Bad tag prompt", tag="invalid-tag", db_path=test_db)

    # 3. Get prompts with active/tag filter
    all_prompts = get_prompts(proj_id, db_path=test_db)
    assert len(all_prompts) == 4

    bof_prompts = get_prompts(proj_id, tag="bottom-of-funnel", db_path=test_db)
    assert len(bof_prompts) == 1
    assert bof_prompts[0]["text"] == "Prompt for bottom-of-funnel"

    # 4. Toggle active state
    first_id = all_prompts[0]["id"]
    toggle_prompt_active(first_id, active=False, db_path=test_db)
    active_prompts = get_prompts(proj_id, active_only=True, db_path=test_db)
    assert len(active_prompts) == 3

    # 5. Update prompt text and tag
    update_prompt(first_id, "Updated prompt text", tag="branded", active=True, db_path=test_db)
    p_updated = [p for p in get_prompts(proj_id, db_path=test_db) if p["id"] == first_id][0]
    assert p_updated["text"] == "Updated prompt text"
    assert p_updated["active"] == 1

    # 6. Bulk add prompts (stripping bullet numbers / hyphens)
    raw_bulk = """
    1. What is the best CRM for small teams?
    2. Top CRM platforms in 2026
    * HubSpot vs Salesforce comparison
    - Best automated sales tool
    """
    added_count = bulk_add_prompts(proj_id, raw_bulk, tag="top-of-funnel", db_path=test_db)
    assert added_count == 4

    tof_prompts = get_prompts(proj_id, tag="top-of-funnel", db_path=test_db)
    # 1 initial + 4 bulk = 5
    assert len(tof_prompts) == 5
    # Verify bullets and numbering were cleaned
    texts = [p["text"] for p in tof_prompts]
    assert "What is the best CRM for small teams?" in texts
    assert "HubSpot vs Salesforce comparison" in texts

    # 7. Delete prompt
    delete_prompt(first_id, db_path=test_db)
    assert len(get_prompts(proj_id, db_path=test_db)) == 4 + 4 - 1


def test_scans_linked_to_project_migration(tmp_path):
    """Test creating scans linked to projects and querying them."""
    test_db = str(tmp_path / "test_scan_projects.db")
    init_db(test_db)

    proj_id = create_project("TestBrand", industry="Fintech", db_path=test_db)
    
    # Create scan with project_id
    scan1 = create_scan("TestBrand", industry="Fintech", project_id=proj_id, db_path=test_db)
    # Create unlinked legacy-style scan
    scan2 = create_scan("OldBrand", industry="Retail", project_id=None, db_path=test_db)

    proj_scans = get_all_scans(project_id=proj_id, db_path=test_db)
    assert len(proj_scans) == 1
    assert proj_scans[0]["id"] == scan1
    assert proj_scans[0]["project_id"] == proj_id

    all_scans = get_all_scans(db_path=test_db)
    assert len(all_scans) == 2

