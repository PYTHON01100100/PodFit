"""SQLite storage: saved rigs + a TTL cache for HuggingFace responses."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid

DB_PATH = os.environ.get("PODFIT_DB", "/data/podfit.db")
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
    return _conn


def init() -> None:
    with _lock:
        c = conn()
        c.executescript("""
        CREATE TABLE IF NOT EXISTS rigs (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            payload TEXT NOT NULL,
            builtin INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            ts REAL NOT NULL
        );
        """)
        c.commit()


# --- cache -----------------------------------------------------------------
def cache_get(key: str, ttl: float):
    with _lock:
        row = conn().execute("SELECT value, ts FROM cache WHERE key=?", (key,)).fetchone()
    if not row or time.time() - row["ts"] > ttl:
        return None
    try:
        return json.loads(row["value"])
    except json.JSONDecodeError:
        return None


def cache_put(key: str, value) -> None:
    with _lock:
        c = conn()
        c.execute("INSERT OR REPLACE INTO cache(key,value,ts) VALUES(?,?,?)",
                  (key, json.dumps(value), time.time()))
        c.commit()


def cache_stats() -> dict:
    with _lock:
        row = conn().execute("SELECT COUNT(*) n, MAX(ts) newest FROM cache").fetchone()
    return {"entries": row["n"], "newest": row["newest"]}


def cache_clear() -> int:
    with _lock:
        c = conn()
        n = c.execute("SELECT COUNT(*) n FROM cache").fetchone()["n"]
        c.execute("DELETE FROM cache")
        c.commit()
    return n


# --- rigs ------------------------------------------------------------------
def list_rigs() -> list:
    with _lock:
        rows = conn().execute(
            "SELECT * FROM rigs ORDER BY builtin DESC, name COLLATE NOCASE").fetchall()
    return [{"id": r["id"], "name": r["name"], "builtin": bool(r["builtin"]),
             "updated_at": r["updated_at"], **json.loads(r["payload"])} for r in rows]


def get_rig(rig_id: str):
    with _lock:
        r = conn().execute("SELECT * FROM rigs WHERE id=?", (rig_id,)).fetchone()
    if not r:
        return None
    return {"id": r["id"], "name": r["name"], "builtin": bool(r["builtin"]),
            **json.loads(r["payload"])}


def save_rig(name: str, payload: dict, rig_id: str | None = None,
             builtin: bool = False) -> str:
    rid = rig_id or uuid.uuid4().hex[:12]
    now = time.time()
    body = json.dumps({k: v for k, v in payload.items()
                       if k not in ("id", "name", "builtin", "updated_at")})
    with _lock:
        c = conn()
        exists = c.execute("SELECT created_at FROM rigs WHERE id=?", (rid,)).fetchone()
        created = exists["created_at"] if exists else now
        c.execute("INSERT OR REPLACE INTO rigs(id,name,payload,builtin,created_at,updated_at)"
                  " VALUES(?,?,?,?,?,?)", (rid, name, body, int(builtin), created, now))
        c.commit()
    return rid


def delete_rig(rig_id: str) -> bool:
    with _lock:
        c = conn()
        n = c.execute("DELETE FROM rigs WHERE id=?", (rig_id,)).rowcount
        c.commit()
    return n > 0


def seed_rig_if_absent(rig_id: str, name: str, payload: dict) -> bool:
    """Insert a detected-hardware rig once; never clobber user edits."""
    with _lock:
        exists = conn().execute("SELECT 1 FROM rigs WHERE id=?", (rig_id,)).fetchone()
    if exists:
        return False
    save_rig(name, payload, rig_id=rig_id, builtin=True)
    return True
