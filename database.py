import sqlite3
import os
import json
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Generator

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "visibility_tracker.db")


@contextmanager
def get_connection(db_path: Optional[str] = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for SQLite database connection.
    
    Anchored to module folder by default.
    Ensures connection is committed on success and always closed on exit.
    """
    path = db_path if db_path is not None else DB_PATH
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
        
        # 1. scans table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brand_name TEXT NOT NULL,
                industry TEXT,
                visibility_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 2. scan_results table (base structure)
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
        ]
        
        for col_name, col_def in migrations:
            if col_name not in existing_cols:
                cursor.execute(f"ALTER TABLE scan_results ADD COLUMN {col_name} {col_def}")
                
        # 3. response_cache table
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
        
        # Index on scan_results for faster lookup by scan_id
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_scan_results_scan_id
            ON scan_results(scan_id)
        """)
        
        # Index on cache lookup
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_lookup
            ON response_cache(provider, model, prompt, run_index)
        """)


def create_scan(brand_name: str, industry: Optional[str] = None, db_path: Optional[str] = None) -> int:
    """Create a new scan record."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO scans (brand_name, industry) VALUES (?, ?)",
            (brand_name, industry)
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
                competitors, provider, model, run_index, status, error, position)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                position
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


def get_all_scans(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all past scans ordered by creation date descending."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM scans ORDER BY created_at DESC, id DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
