"""
Part 7: Memory & Context Persistence — User Memory
Persists borrower preferences, commitments, and interaction history across sessions.
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path


DB_PATH = Path("./credsolve_memory.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            customer_id TEXT PRIMARY KEY,
            loan_id TEXT,
            name TEXT,
            preferred_language TEXT DEFAULT 'hi',
            preferred_contact_time TEXT DEFAULT 'morning',
            hardship_declared INTEGER DEFAULT 0,
            total_interactions INTEGER DEFAULT 0,
            successful_ptps INTEGER DEFAULT 0,
            broken_ptps INTEGER DEFAULT 0,
            risk_score REAL DEFAULT 0.5,
            notes TEXT DEFAULT '',
            created_at TEXT,
            updated_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS interaction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            customer_id TEXT,
            loan_id TEXT,
            intent TEXT,
            outcome TEXT,
            agent_response_summary TEXT,
            ptp_date TEXT,
            ptp_amount REAL,
            escalated INTEGER DEFAULT 0,
            satisfaction_score REAL,
            duration_seconds INTEGER,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def upsert_user_profile(customer_id: str, loan_id: str, name: str, **kwargs):
    conn = _get_conn()
    now = datetime.now().isoformat()
    existing = conn.execute(
        "SELECT * FROM user_profiles WHERE customer_id = ?", (customer_id,)
    ).fetchone()

    if existing:
        updates = {k: v for k, v in kwargs.items()}
        updates["updated_at"] = now
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        conn.execute(
            f"UPDATE user_profiles SET {set_clause} WHERE customer_id = ?",
            (*updates.values(), customer_id),
        )
    else:
        conn.execute(
            """INSERT INTO user_profiles
               (customer_id, loan_id, name, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (customer_id, loan_id, name, now, now),
        )
    conn.commit()
    conn.close()


def get_user_profile(customer_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM user_profiles WHERE customer_id = ?", (customer_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def log_interaction(
    session_id: str,
    customer_id: str,
    loan_id: str,
    intent: str,
    outcome: str,
    agent_response_summary: str,
    ptp_date: str = None,
    ptp_amount: float = None,
    escalated: bool = False,
    satisfaction_score: float = None,
    duration_seconds: int = 0,
):
    conn = _get_conn()
    conn.execute(
        """INSERT INTO interaction_history
           (session_id, customer_id, loan_id, intent, outcome,
            agent_response_summary, ptp_date, ptp_amount, escalated,
            satisfaction_score, duration_seconds, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id, customer_id, loan_id, intent, outcome,
            agent_response_summary, ptp_date, ptp_amount,
            int(escalated), satisfaction_score, duration_seconds,
            datetime.now().isoformat(),
        ),
    )
    # Increment total_interactions
    conn.execute(
        "UPDATE user_profiles SET total_interactions = total_interactions + 1, updated_at = ? WHERE customer_id = ?",
        (datetime.now().isoformat(), customer_id),
    )
    conn.commit()
    conn.close()


def get_interaction_history(customer_id: str, limit: int = 5) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        """SELECT * FROM interaction_history
           WHERE customer_id = ?
           ORDER BY created_at DESC LIMIT ?""",
        (customer_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def build_memory_summary(customer_id: str) -> str:
    profile = get_user_profile(customer_id)
    history = get_interaction_history(customer_id, limit=3)

    if not profile and not history:
        return "पहली बातचीत — कोई पूर्व इतिहास नहीं।"

    parts = []
    if profile:
        parts.append(
            f"कुल बातचीत: {profile.get('total_interactions', 0)} | "
            f"सफल PTP: {profile.get('successful_ptps', 0)} | "
            f"टूटे PTP: {profile.get('broken_ptps', 0)} | "
            f"पसंदीदा भाषा: {profile.get('preferred_language', 'hi')}"
        )
    if history:
        for h in history:
            parts.append(
                f"[{h['created_at'][:10]}] Intent={h['intent']}, Outcome={h['outcome']}"
            )
    return "\n".join(parts)


# Initialize DB on import
init_db()
