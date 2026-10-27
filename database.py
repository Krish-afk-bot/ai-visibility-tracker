import sqlite3
import os
import json
import re
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Generator

from config import get_setting

DEFAULT_DB_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, "visibility_tracker.db")


def get_db_path() -> str:
    """Return configured database path or default local path.
    
    Supports DATABASE_URL (e.g. 'sqlite:///path/to/db.sqlite3' or hosted libSQL/Postgres)
    as well as DB_PATH for local SQLite file.
    """
    db_url = get_setting("DATABASE_URL")
    if db_url and str(db_url).strip():
        url_str = str(db_url).strip()
        if url_str.startswith("sqlite:///"):
            return os.path.abspath(url_str[len("sqlite:///"):])
        elif url_str.startswith("sqlite://"):
            return os.path.abspath(url_str[len("sqlite://"):])
        # If it's a file path supplied in DATABASE_URL
        if "://" not in url_str:
            return os.path.abspath(url_str)
        return url_str

    configured = get_setting("DB_PATH")
    if configured and str(configured).strip():
        return os.path.abspath(str(configured).strip())
    return DEFAULT_DB_PATH


@contextmanager
def get_connection(db_path: Optional[str] = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for SQLite database connection.
    
    Pluggable via DATABASE_URL or DB_PATH.
    Anchored to module folder or DB_PATH setting by default.
    Ensures connection is committed on success and always closed on exit.
    """
    path = db_path if db_path is not None else get_db_path()
    
    # Handle sqlite:/// URI prefix if passed directly
    if path.startswith("sqlite:///"):
        path = os.path.abspath(path[len("sqlite:///"):])
    elif path.startswith("sqlite://"):
        path = os.path.abspath(path[len("sqlite://"):])

    # Ensure parent directory exists for local files
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _get_table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    """Helper to inspect column names of a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    rows = cursor.fetchall()
    return [row["name"] for row in rows]


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize database tables and run safe, idempotent migrations.
    
    Existing tables/columns are preserved without deleting data.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # 1. projects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brand_name TEXT NOT NULL,
                aliases TEXT,
                industry TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. prompts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                tag TEXT NOT NULL,
                active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
        """)

        # 3. scans table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brand_name TEXT NOT NULL,
                industry TEXT,
                visibility_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Safe migration for scans (project_id, expected_calls)
        existing_scan_cols = _get_table_columns(conn, "scans")
        if "project_id" not in existing_scan_cols:
            cursor.execute("ALTER TABLE scans ADD COLUMN project_id INTEGER REFERENCES projects(id)")
        if "expected_calls" not in existing_scan_cols:
            cursor.execute("ALTER TABLE scans ADD COLUMN expected_calls INTEGER")

        # 4. scan_results table (base structure)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scan_results (
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
        
        # Safe migration for scan_results new columns
        existing_cols = _get_table_columns(conn, "scan_results")
        
        migrations = [
            ("provider", "TEXT DEFAULT 'gemini'"),
            ("model", "TEXT DEFAULT 'gemini-2.0-flash'"),
            ("run_index", "INTEGER DEFAULT 1"),
            ("status", "TEXT DEFAULT 'ok'"),
            ("error", "TEXT"),
            ("position", "INTEGER"),
            ("analyzer_status", "TEXT DEFAULT 'ok'"),
            ("analyzer_error", "TEXT"),
        ]
        
        for col_name, col_def in migrations:
            if col_name not in existing_cols:
                cursor.execute(f"ALTER TABLE scan_results ADD COLUMN {col_name} {col_def}")
                
        # 5. response_cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS response_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt TEXT NOT NULL,
                run_index INTEGER NOT NULL DEFAULT 1,
                ai_response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(provider, model, prompt, run_index)
            )
        """)
        
        # Indexes for fast lookup
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_prompts_project_id
            ON prompts(project_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_scans_project_id
            ON scans(project_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_scan_results_scan_id
            ON scan_results(scan_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_lookup
            ON response_cache(provider, model, prompt, run_index)
        """)


def create_project(
    brand_name: str,
    aliases: Optional[List[str]] = None,
    industry: Optional[str] = None,
    db_path: Optional[str] = None
) -> int:
    """Create a new project record."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        aliases_json = json.dumps(aliases) if aliases else "[]"
        cursor.execute(
            "INSERT INTO projects (brand_name, aliases, industry) VALUES (?, ?, ?)",
            (brand_name.strip(), aliases_json, industry.strip() if industry else None)
        )
        return cursor.lastrowid


def get_project(project_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get project by ID."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        if isinstance(d.get("aliases"), str):
            try:
                d["aliases"] = json.loads(d["aliases"])
            except Exception:
                d["aliases"] = [a.strip() for a in d["aliases"].split(",") if a.strip()]
        return d


def get_projects(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all projects ordered by creation date descending."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM projects ORDER BY created_at DESC, id DESC")
        rows = cursor.fetchall()
        projects = []
        for row in rows:
            d = dict(row)
            if isinstance(d.get("aliases"), str):
                try:
                    d["aliases"] = json.loads(d["aliases"])
                except Exception:
                    d["aliases"] = [a.strip() for a in d["aliases"].split(",") if a.strip()]
            projects.append(d)
        return projects


def update_project(
    project_id: int,
    brand_name: str,
    aliases: Optional[List[str]] = None,
    industry: Optional[str] = None,
    db_path: Optional[str] = None
) -> None:
    """Update project details."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        aliases_json = json.dumps(aliases) if aliases else "[]"
        cursor.execute(
            "UPDATE projects SET brand_name = ?, aliases = ?, industry = ? WHERE id = ?",
            (brand_name.strip(), aliases_json, industry.strip() if industry else None, project_id)
        )


def delete_project(project_id: int, db_path: Optional[str] = None) -> None:
    """Delete a project and its associated prompts (scans are preserved with project_id decoupled)."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM prompts WHERE project_id = ?", (project_id,))
        cursor.execute("UPDATE scans SET project_id = NULL WHERE project_id = ?", (project_id,))
        cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))


VALID_PROMPT_TAGS = {"branded", "non-branded", "top-of-funnel", "bottom-of-funnel"}


def create_prompt(
    project_id: int,
    text: str,
    tag: str = "non-branded",
    active: bool = True,
    db_path: Optional[str] = None
) -> int:
    """Add a prompt to a project."""
    if tag not in VALID_PROMPT_TAGS:
        raise ValueError(f"Invalid tag '{tag}'. Must be one of: {sorted(list(VALID_PROMPT_TAGS))}")
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO prompts (project_id, text, tag, active) VALUES (?, ?, ?, ?)",
            (project_id, text.strip(), tag, 1 if active else 0)
        )
        return cursor.lastrowid


def get_prompts(
    project_id: int,
    active_only: bool = False,
    tag: Optional[str] = None,
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get prompts for a project with optional filters."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM prompts WHERE project_id = ?"
        params: List[Any] = [project_id]
        if active_only:
            query += " AND active = 1"
        if tag:
            query += " AND tag = ?"
            params.append(tag)
        query += " ORDER BY id ASC"
        cursor.execute(query, tuple(params))
        return [dict(r) for r in cursor.fetchall()]


def update_prompt(
    prompt_id: int,
    text: str,
    tag: str,
    active: bool = True,
    db_path: Optional[str] = None
) -> None:
    """Update prompt text, tag, or active status."""
    if tag not in VALID_PROMPT_TAGS:
        raise ValueError(f"Invalid tag '{tag}'. Must be one of: {sorted(list(VALID_PROMPT_TAGS))}")
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE prompts SET text = ?, tag = ?, active = ? WHERE id = ?",
            (text.strip(), tag, 1 if active else 0, prompt_id)
        )


def toggle_prompt_active(prompt_id: int, active: bool, db_path: Optional[str] = None) -> None:
    """Quickly toggle active state of a prompt."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE prompts SET active = ? WHERE id = ?",
            (1 if active else 0, prompt_id)
        )


def delete_prompt(prompt_id: int, db_path: Optional[str] = None) -> None:
    """Delete a prompt by ID."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))


def bulk_add_prompts(
    project_id: int,
    prompts_text: str,
    tag: str = "non-branded",
    db_path: Optional[str] = None
) -> int:
    """Bulk add prompts from a newline-separated string. Returns count added."""
    if tag not in VALID_PROMPT_TAGS:
        raise ValueError(f"Invalid tag '{tag}'. Must be one of: {sorted(list(VALID_PROMPT_TAGS))}")
    lines = [line.strip() for line in prompts_text.splitlines() if line.strip()]
    count = 0
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        for line in lines:
            # Strip bullet prefixes if pasted directly from lists
            cleaned = re.sub(r"^\s*(?:#?\d+[\.\)]|[\*\-\u2022])\s+", "", line).strip()
            if cleaned:
                cursor.execute(
                    "INSERT INTO prompts (project_id, text, tag, active) VALUES (?, ?, ?, 1)",
                    (project_id, cleaned, tag)
                )
                count += 1
    return count


def create_scan(
    brand_name: str,
    industry: Optional[str] = None,
    project_id: Optional[int] = None,
    expected_calls: Optional[int] = None,
    db_path: Optional[str] = None
) -> int:
    """Create a new scan record linked optionally to a project.

    `expected_calls` records prompts x models x runs up front so the scan can
    later prove whether every planned call produced a row.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO scans (brand_name, industry, project_id, expected_calls) VALUES (?, ?, ?, ?)",
            (brand_name, industry, project_id, expected_calls)
        )
        scan_id = cursor.lastrowid
        return scan_id


def update_scan_score(scan_id: int, visibility_score: float, db_path: Optional[str] = None) -> None:
    """Update visibility score of an existing scan."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE scans SET visibility_score = ? WHERE id = ?",
            (visibility_score, scan_id)
        )


def save_result(
    scan_id: int,
    prompt: str,
    ai_response: Optional[str],
    brand_mentioned: bool,
    mention_count: int,
    sentiment: str,
    competitors: List[str],
    provider: str = "gemini",
    model: str = "gemini-2.0-flash",
    run_index: int = 1,
    status: str = "ok",
    error: Optional[str] = None,
    position: Optional[int] = None,
    analyzer_status: str = "ok",
    analyzer_error: Optional[str] = None,
    db_path: Optional[str] = None
) -> int:
    """Save a scan result row.
    
    Supports extended columns while preserving backward compatibility.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        competitors_json = json.dumps(competitors) if competitors is not None else "[]"
        cursor.execute(
            """INSERT INTO scan_results 
               (scan_id, prompt, ai_response, brand_mentioned, mention_count, sentiment,
                competitors, provider, model, run_index, status, error, position,
                analyzer_status, analyzer_error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                scan_id,
                prompt,
                ai_response,
                1 if brand_mentioned else 0,
                mention_count,
                sentiment,
                competitors_json,
                provider,
                model,
                run_index,
                status,
                error,
                position,
                analyzer_status or "ok",
                analyzer_error
            )
        )
        return cursor.lastrowid


def get_scan_results(scan_id: int, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all results for a specific scan."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM scan_results WHERE scan_id = ? ORDER BY id",
            (scan_id,)
        )
        results = cursor.fetchall()
        parsed_results = []
        for row in results:
            d = dict(row)
            if isinstance(d.get("competitors"), str):
                try:
                    d["competitors"] = json.loads(d["competitors"])
                except Exception:
                    d["competitors"] = []
            parsed_results.append(d)
        return parsed_results


def get_cached_response(
    provider: str,
    model: str,
    prompt: str,
    run_index: int = 1,
    db_path: Optional[str] = None
) -> Optional[str]:
    """Retrieve cached AI response if it exists."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT ai_response FROM response_cache
               WHERE provider = ? AND model = ? AND prompt = ? AND run_index = ?""",
            (provider, model, prompt, run_index)
        )
        row = cursor.fetchone()
        if row:
            return row["ai_response"]
        return None


def set_cached_response(
    provider: str,
    model: str,
    prompt: str,
    ai_response: str,
    run_index: int = 1,
    db_path: Optional[str] = None
) -> None:
    """Store or update AI response in cache."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO response_cache (provider, model, prompt, run_index, ai_response)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(provider, model, prompt, run_index)
               DO UPDATE SET ai_response = excluded.ai_response, created_at = CURRENT_TIMESTAMP""",
            (provider, model, prompt, run_index, ai_response)
        )


def get_all_scans(project_id: Optional[int] = None, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all past scans ordered by creation date descending, optionally filtered by project_id."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if project_id is not None:
            cursor.execute("SELECT * FROM scans WHERE project_id = ? ORDER BY created_at DESC, id DESC", (project_id,))
        else:
            cursor.execute("SELECT * FROM scans ORDER BY created_at DESC, id DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_scan(scan_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return one scan record (including expected_calls), or None."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM scans WHERE id = ?", (scan_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_scan_status_counts(db_path: Optional[str] = None) -> Dict[int, Dict[str, int]]:
    """Return per-scan completed/failed row counts plus an inferred expected total.

    expected_inferred = distinct prompts x distinct models x max run index. It is
    only a fallback for scans recorded before expected_calls was stored.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                scan_id,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
                SUM(CASE WHEN status != 'failed' THEN 1 ELSE 0 END) AS completed,
                COUNT(DISTINCT prompt)
                    * COUNT(DISTINCT provider || ':' || model)
                    * MAX(COALESCE(run_index, 1)) AS expected_inferred
            FROM scan_results
            GROUP BY scan_id
        """)
        rows = cursor.fetchall()
        return {
            row["scan_id"]: {
                "completed": int(row["completed"] or 0),
                "failed": int(row["failed"] or 0),
                "expected_inferred": int(row["expected_inferred"] or 0),
            }
            for row in rows
        }


def get_scans_and_results_in_range(
    brand_name: Optional[str] = None,
    project_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieve joined scan and scan_results rows within date range and project/brand filters."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        query = """
            SELECT 
                s.id as scan_id,
                s.brand_name as scan_brand,
                s.industry as scan_industry,
                s.project_id,
                s.expected_calls,
                s.created_at as scan_created_at,
                sr.id as result_id,
                sr.prompt,
                sr.ai_response,
                sr.brand_mentioned,
                sr.mention_count,
                sr.sentiment,
                sr.competitors,
                sr.provider,
                sr.model,
                sr.run_index,
                sr.status,
                sr.error,
                sr.position,
                sr.analyzer_status,
                sr.analyzer_error,
                p.tag as prompt_tag
            FROM scans s
            JOIN scan_results sr ON s.id = sr.scan_id
            LEFT JOIN prompts p ON s.project_id = p.project_id AND sr.prompt = p.text
            WHERE 1=1
        """
        params: List[Any] = []
        if project_id is not None:
            query += " AND s.project_id = ?"
            params.append(project_id)
        elif brand_name is not None:
            query += " AND LOWER(s.brand_name) = LOWER(?)"
            params.append(brand_name.strip())

        if start_date:
            query += " AND s.created_at >= ?"
            params.append(start_date)
        if end_date:
            query += " AND s.created_at <= ?"
            params.append(end_date)

        query += " ORDER BY s.created_at ASC, sr.id ASC"
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        parsed = []
        for r in rows:
            d = dict(r)
            if isinstance(d.get("competitors"), str):
                try:
                    d["competitors"] = json.loads(d["competitors"])
                except Exception:
                    d["competitors"] = []
            parsed.append(d)
        return parsed


def get_database_bytes(db_path: Optional[str] = None) -> bytes:
    """Read the raw SQLite database file as bytes for backup."""
    path = db_path if db_path is not None else get_db_path()
    if not os.path.exists(path):
        init_db(path)
    with open(path, "rb") as f:
        return f.read()


def restore_database_from_bytes(data: bytes, target_path: Optional[str] = None) -> bool:
    """Validate that data is a valid SQLite database with expected tables,
    and restore it safely to target_path.
    
    Raises ValueError with descriptive message if validation fails.
    """
    if not data:
        raise ValueError("Invalid backup file: file is empty or too small.")

    # SQLite files begin with "SQLite format 3\000"
    if not data.startswith(b"SQLite format 3\x00"):
        raise ValueError("Invalid backup file: not a valid SQLite database header.")

    if len(data) < 100:
        raise ValueError("Invalid backup file: file is empty or too small.")

    
    # Write to a temporary file first to test integrity and schema
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        tmp_name = tmp.name
        tmp.write(data)
        tmp.flush()

    try:
        test_conn = sqlite3.connect(tmp_name)
        cursor = test_conn.cursor()
        # Run integrity check
        cursor.execute("PRAGMA integrity_check")
        row = cursor.fetchone()
        if not row or row[0] != "ok":
            raise ValueError(f"SQLite integrity check failed: {row[0] if row else 'unknown error'}")

        # Check for required tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cursor.fetchall()}
        required = {"scans", "scan_results"}
        if not required.issubset(tables):
            missing = required - tables
            raise ValueError(f"Backup database missing required tables: {missing}")

        test_conn.close()

        # If validation succeeded, replace active database
        path = target_path if target_path is not None else get_db_path()
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
            
        with open(path, "wb") as f:
            f.write(data)
            f.flush()

        # Run init_db on newly restored DB to ensure any missing new columns exist
        init_db(path)
        return True

    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except Exception:
                pass
