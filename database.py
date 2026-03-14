import sqlite3
from datetime import datetime
from typing import Optional
import json


def get_connection():
    conn = sqlite3.connect("visibility_tracker.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand_name TEXT NOT NULL,
            industry TEXT,
            visibility_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
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
    
    conn.commit()
    conn.close()


def create_scan(brand_name: str, industry: Optional[str] = None) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO scans (brand_name, industry) VALUES (?, ?)",
        (brand_name, industry)
    )
    scan_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return scan_id


def update_scan_score(scan_id: int, visibility_score: float):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE scans SET visibility_score = ? WHERE id = ?",
        (visibility_score, scan_id)
    )
    conn.commit()
    conn.close()


def save_result(
    scan_id: int,
    prompt: str,
    ai_response: str,
    brand_mentioned: bool,
    mention_count: int,
    sentiment: str,
    competitors: list
):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO scan_results 
           (scan_id, prompt, ai_response, brand_mentioned, mention_count, sentiment, competitors)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (scan_id, prompt, ai_response, int(brand_mentioned), mention_count, sentiment, json.dumps(competitors))
    )
    conn.commit()
    conn.close()


def get_scan_results(scan_id: int) -> list:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM scan_results WHERE scan_id = ? ORDER BY id",
        (scan_id,)
    )
    results = cursor.fetchall()
    conn.close()
    return [dict(row) for row in results]
