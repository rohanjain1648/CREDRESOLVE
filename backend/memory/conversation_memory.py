"""
Part 7: Conversation Memory — tracks in-session message history and state transitions.
Works alongside LangGraph's built-in checkpointing.
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path


SESSION_DB_PATH = Path("./credsolve_sessions.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(SESSION_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_session_db():
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            customer_id TEXT,
            loan_id TEXT,
            channel TEXT DEFAULT 'text',
            started_at TEXT,
            ended_at TEXT,
            final_state TEXT,
            total_turns INTEGER DEFAULT 0,
            token_usage INTEGER DEFAULT 0,
            latency_ms REAL DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS state_transitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            from_state TEXT,
            to_state TEXT,
            trigger TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def create_session(
    session_id: str,
    customer_id: str,
    loan_id: str,
    channel: str = "text",
):
    conn = _get_conn()
    conn.execute(
        """INSERT OR IGNORE INTO sessions
           (session_id, customer_id, loan_id, channel, started_at)
           VALUES (?, ?, ?, ?, ?)""",
        (session_id, customer_id, loan_id, channel, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def log_transition(
    session_id: str,
    from_state: str,
    to_state: str,
    trigger: str = "",
):
    conn = _get_conn()
    conn.execute(
        """INSERT INTO state_transitions
           (session_id, from_state, to_state, trigger, timestamp)
           VALUES (?, ?, ?, ?, ?)""",
        (session_id, from_state, to_state, trigger, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def close_session(
    session_id: str,
    final_state: str,
    token_usage: int = 0,
    latency_ms: float = 0,
):
    conn = _get_conn()
    conn.execute(
        """UPDATE sessions
           SET ended_at = ?, final_state = ?, token_usage = ?, latency_ms = ?
           WHERE session_id = ?""",
        (datetime.now().isoformat(), final_state, token_usage, latency_ms, session_id),
    )
    conn.commit()
    conn.close()


def get_session_transitions(session_id: str) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM state_transitions WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


init_session_db()
