"""Database layer: SQLite by default, PostgreSQL when DATABASE_URL is set.

Why two backends: on Render's free tier the local disk is wiped on every deploy
and restart, so SQLite would delete every user account. Point DATABASE_URL at a
hosted Postgres (Neon, Supabase, Render Postgres...) for persistent accounts, or
mount a Render Disk and set DB_PATH to a file on it.

All SQL in the app uses '?' placeholders; they are translated for Postgres.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
IS_PG = DATABASE_URL.startswith(("postgres://", "postgresql://"))
SQLITE_PATH = Path(os.getenv("DB_PATH") or Path(__file__).resolve().parent.parent / "history.db")

if IS_PG:
    import psycopg
    from psycopg.rows import dict_row

    INTEGRITY_ERRORS: tuple = (psycopg.IntegrityError,)
else:
    INTEGRITY_ERRORS = (sqlite3.IntegrityError,)


class Conn:
    """Thin wrapper so both backends expose execute()/fetchone()/fetchall()/rowcount."""

    def __init__(self, raw):
        self.raw = raw

    def execute(self, sql: str, params: tuple = ()):
        if IS_PG:
            cur = self.raw.cursor()
            cur.execute(sql.replace("?", "%s"), params)
            return cur
        return self.raw.execute(sql, params)


@contextmanager
def connect() -> Iterator[Conn]:
    """Open a connection; commit on success, roll back on error, always close."""
    if IS_PG:
        # prepare_threshold=None: no server-side prepared statements, so this also works through
        # connection poolers such as Neon's "pooled" connection string (PgBouncer).
        raw = psycopg.connect(DATABASE_URL, row_factory=dict_row, connect_timeout=10, prepare_threshold=None)
    else:
        raw = sqlite3.connect(SQLITE_PATH, timeout=10)
        raw.row_factory = sqlite3.Row
    try:
        yield Conn(raw)
        raw.commit()
    except BaseException:
        raw.rollback()
        raise
    finally:
        raw.close()


def init_db() -> None:
    auto_id = "BIGSERIAL PRIMARY KEY" if IS_PG else "INTEGER PRIMARY KEY"
    statements = [
        "CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT, "
        "password_hash TEXT NOT NULL, created TEXT)",
        "CREATE TABLE IF NOT EXISTS sessions(token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, "
        "created TEXT, expires TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS conversations(id TEXT PRIMARY KEY, title TEXT, created TEXT, updated TEXT, "
        "client_id TEXT, user_id TEXT)",
        f"CREATE TABLE IF NOT EXISTS messages(id {auto_id}, conversation_id TEXT, role TEXT, content TEXT, "
        "metadata TEXT, created TEXT)",
    ]
    with connect() as c:
        for sql in statements:
            c.execute(sql)
        # Migrate databases created by earlier versions of the app.
        if IS_PG:
            c.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS client_id TEXT")
            c.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS user_id TEXT")
        else:
            cols = [r["name"] for r in c.execute("PRAGMA table_info(conversations)").fetchall()]
            for col in ("client_id", "user_id"):
                if col not in cols:
                    c.execute(f"ALTER TABLE conversations ADD COLUMN {col} TEXT")
        for sql in (
                "CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, updated)",
                "CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, id)",
                "CREATE INDEX IF NOT EXISTS idx_sess_user ON sessions(user_id)",
        ):
            c.execute(sql)