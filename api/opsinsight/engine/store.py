"""The app's own small database (SQLite): every upload and every manager decision.

SQLite is one file (data/app/opsinsight.db) that comes with Python, so there is nothing to install.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from ..models.config import ROOT

DB_FILE = ROOT / "data/app/opsinsight.db"

TABLES = """
CREATE TABLE IF NOT EXISTS uploads (
    id TEXT PRIMARY KEY, company TEXT, created_at TEXT, source TEXT,
    folder TEXT, main_file TEXT, status TEXT, domain TEXT, report TEXT);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, analysis_key TEXT, rec_id TEXT, decision TEXT,
    decided_by TEXT, comment TEXT, edited TEXT, decided_at TEXT);
CREATE TABLE IF NOT EXISTS mappings (
    company TEXT, domain TEXT, mapping TEXT, saved_at TEXT, PRIMARY KEY (company, domain));
"""


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextmanager
def connect():
    """Open the database, save the changes, and always close it again (Windows locks open files)."""
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row            # rows behave like dictionaries
    try:
        con.executescript(TABLES)             # creates the tables the first time only
        yield con
        con.commit()
    finally:
        con.close()


# --------------------------------------------------------------------------- uploads
def add_upload(upload_id, company, source, folder, main_file, status, domain, report):
    with connect() as con:
        con.execute("INSERT INTO uploads VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (upload_id, company, now(), source, str(folder), main_file, status, domain,
                     json.dumps(report, default=str)))


def update_upload(upload_id, **fields):
    if "report" in fields:
        fields["report"] = json.dumps(fields["report"], default=str)
    sets = ", ".join(f"{k} = ?" for k in fields)
    with connect() as con:
        con.execute(f"UPDATE uploads SET {sets} WHERE id = ?", (*fields.values(), upload_id))


def get_upload(upload_id) -> dict | None:
    with connect() as con:
        row = con.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
    if row is None:
        return None
    out = dict(row)
    out["report"] = json.loads(out["report"]) if out["report"] else None
    return out


def list_uploads(limit=20) -> list:
    """Newest uploads first, without server folders or the full report."""
    with connect() as con:
        rows = con.execute("SELECT id, company, created_at, source, main_file, status, domain FROM uploads "
                           "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- remembered column mappings
def save_mapping(company, domain, mapping):
    """The mapping a company confirmed, so its next upload of the same export is mapped for it."""
    if not company or company == "unknown":
        return
    with connect() as con:
        con.execute("INSERT OR REPLACE INTO mappings VALUES (?, ?, ?, ?)", (company, domain, json.dumps(mapping), now()))


def saved_mapping(company, domain) -> dict:
    with connect() as con:
        row = con.execute("SELECT mapping FROM mappings WHERE company = ? AND domain = ?", (company, domain)).fetchone()
    return json.loads(row["mapping"]) if row else {}


# --------------------------------------------------------------------------- decisions
def add_decision(analysis_key, rec_id, decision, decided_by, comment=None, edited=None) -> dict:
    entry = {"decision": decision, "decided_by": decided_by, "comment": comment, "edited": edited, "decided_at": now()}
    with connect() as con:
        con.execute("INSERT INTO decisions (analysis_key, rec_id, decision, decided_by, comment, edited, decided_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (analysis_key, rec_id, decision, decided_by, comment, json.dumps(edited) if edited else None,
                     entry["decided_at"]))
    return entry


def latest_decisions(analysis_key) -> dict:
    """{rec_id: newest decision}. Older decisions stay in the table as history."""
    with connect() as con:
        rows = con.execute("SELECT * FROM decisions WHERE analysis_key = ? ORDER BY id", (analysis_key,)).fetchall()
    out = {}
    for r in rows:
        out[r["rec_id"]] = {"decision": r["decision"], "decided_by": r["decided_by"], "comment": r["comment"],
                            "edited": json.loads(r["edited"]) if r["edited"] else None, "decided_at": r["decided_at"]}
    return out
