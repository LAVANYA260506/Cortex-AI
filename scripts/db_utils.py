#!/usr/bin/env python3
"""
db_utils.py — SQLite helper for the knowledge graph pipeline.

Called by Gemini CLI (gemini.md) to read/write kg.db.
The embedder daemon does NOT import this — it reads embed_queue.jsonl instead.

When set-summary is called, it:
  1. Writes the summary to kg.db.
  2. Appends an embed job to embed_queue.jsonl via job_queue.enqueue().
The embedder polls that file independently.

Usage (CLI):
    python scripts/db_utils.py init
    python scripts/db_utils.py upsert-node <json>
    python scripts/db_utils.py upsert-link <source_id> <target_id> <reason>
    python scripts/db_utils.py set-summary <node_id> <summary_text>
    python scripts/db_utils.py list-nodes [--orphans] [--no-summary]
    python scripts/db_utils.py list-summaries          ← all nodes with summaries (for linking)
    python scripts/db_utils.py last-seen-max           ← incremental update baseline timestamp
    python scripts/db_utils.py list-links
    python scripts/db_utils.py broken-links
    python scripts/db_utils.py stale-nodes [--days 7]
    python scripts/db_utils.py get-node <node_id_or_filename>
    python scripts/db_utils.py delete-link <source_id> <target_id>
    python scripts/db_utils.py node-id <filename>
"""

from __future__ import annotations

import json
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path("kg.db")
RAW_DIR = Path("raw")
UUID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # URL namespace


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

def node_id_for(filepath: str | Path) -> str:
    """Stable UUID5 from the relative file path (e.g. 'raw/report.md')."""
    rel = Path(filepath)
    if not str(rel).startswith("raw/"):
        rel = RAW_DIR / rel.name
    return str(uuid.uuid5(UUID_NAMESPACE, str(rel)))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id                TEXT PRIMARY KEY,
    filename          TEXT NOT NULL UNIQUE,
    original_filename TEXT,
    chunk             INTEGER DEFAULT 0,
    total_chunks      INTEGER DEFAULT 1,
    processed_date    TEXT,
    last_seen_at      TEXT,
    summary           TEXT,
    summary_updated_at TEXT,
    linked_ids        TEXT DEFAULT '[]'       -- JSON array of target IDs
);

CREATE TABLE IF NOT EXISTS links (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    target_id   TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    reason      TEXT,
    created_at  TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    UNIQUE(source_id, target_id)
);

CREATE INDEX IF NOT EXISTS idx_links_source ON links(source_id);
CREATE INDEX IF NOT EXISTS idx_links_target ON links(target_id);
CREATE INDEX IF NOT EXISTS idx_nodes_summary ON nodes(summary);
"""


def get_conn(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(path: Path = DB_PATH) -> None:
    with get_conn(path) as conn:
        conn.executescript(SCHEMA)
    print(f"Initialised database: {path}")


# ---------------------------------------------------------------------------
# Node operations
# ---------------------------------------------------------------------------

def upsert_node(data: dict, conn: Optional[sqlite3.Connection] = None) -> str:
    """
    Insert or update a node. Returns the node ID.
    data keys: filename, original_filename, chunk, total_chunks, processed_date
    """
    node_id = node_id_for(data["filename"])
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    sql = """
    INSERT INTO nodes (id, filename, original_filename, chunk, total_chunks,
                       processed_date, last_seen_at, linked_ids)
    VALUES (:id, :filename, :original_filename, :chunk, :total_chunks,
            :processed_date, :now, '[]')
    ON CONFLICT(id) DO UPDATE SET
        filename          = excluded.filename,
        original_filename = excluded.original_filename,
        chunk             = excluded.chunk,
        total_chunks      = excluded.total_chunks,
        processed_date    = excluded.processed_date,
        last_seen_at      = excluded.last_seen_at
    """
    params = {
        "id": node_id,
        "filename": str(data["filename"]),
        "original_filename": data.get("original_filename", ""),
        "chunk": data.get("chunk", 0),
        "total_chunks": data.get("total_chunks", 1),
        "processed_date": data.get("processed_date", now),
        "now": now,
    }

    close = conn is None
    if close:
        conn = get_conn()
    try:
        conn.execute(sql, params)
        if close:
            conn.commit()
    finally:
        if close:
            conn.close()
    return node_id


def set_summary(node_id: str, summary: str, conn: Optional[sqlite3.Connection] = None) -> None:
    """
    Write summary to kg.db, then append an embed job to embed_queue.jsonl.
    The embedder daemon polls that file — no direct coupling to this function.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sql = "UPDATE nodes SET summary = ?, summary_updated_at = ? WHERE id = ?"

    close = conn is None
    if close:
        conn = get_conn()
    try:
        conn.execute(sql, (summary, now, node_id))
        # Fetch filename so the embedder job carries it (useful for logging)
        row = conn.execute("SELECT filename FROM nodes WHERE id = ?", (node_id,)).fetchone()
        filename = row["filename"] if row else ""
        if close:
            conn.commit()
    finally:
        if close:
            conn.close()

    # Enqueue embed job — fire-and-forget, embedder picks it up independently
    import scripts.job_queue as job_queue  # local import keeps the module optional for pure-DB users
    job_queue.enqueue(node_id, filename, summary)


def get_node(id_or_filename: str, conn: Optional[sqlite3.Connection] = None) -> Optional[dict]:
    close = conn is None
    if close:
        conn = get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM nodes WHERE id = ? OR filename = ?",
            (id_or_filename, id_or_filename)
        ).fetchone()
        return dict(row) if row else None
    finally:
        if close:
            conn.close()


def list_nodes(
    orphans_only: bool = False,
    no_summary: bool = False,
    conn: Optional[sqlite3.Connection] = None,
) -> list[dict]:
    """Return nodes matching filters."""
    close = conn is None
    if close:
        conn = get_conn()
    try:
        clauses = []
        if no_summary:
            clauses.append("summary IS NULL")
        if orphans_only:
            clauses.append("""
                id NOT IN (SELECT source_id FROM links)
                AND id NOT IN (SELECT target_id FROM links)
            """)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = conn.execute(f"SELECT * FROM nodes {where} ORDER BY filename").fetchall()
        return [dict(r) for r in rows]
    finally:
        if close:
            conn.close()


def stale_nodes(days: int = 7, conn: Optional[sqlite3.Connection] = None) -> list[dict]:
    close = conn is None
    if close:
        conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM nodes WHERE last_seen_at < datetime('now', ?)",
            (f"-{days} days",)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        if close:
            conn.close()


def last_seen_max(conn: Optional[sqlite3.Connection] = None) -> str:
    """
    Return the ISO timestamp of the most recently seen node.
    Returns '1970-01-01T00:00:00Z' when the table is empty so that
    md_utils list-files --since that value returns ALL files on first run.
    The CLI calls this at the start of every `update` to find the incremental
    baseline — only files newer than this timestamp are processed.
    """
    close = conn is None
    if close:
        conn = get_conn()
    try:
        row = conn.execute("SELECT MAX(last_seen_at) AS mx FROM nodes").fetchone()
        return row["mx"] if (row and row["mx"]) else "1970-01-01T00:00:00Z"
    finally:
        if close:
            conn.close()


def list_summaries(conn: Optional[sqlite3.Connection] = None) -> list[dict]:
    """
    Return id, filename, summary for every node that has a summary.

    This is the O(1-query) linking strategy: the CLI fetches this once at the
    start of the linking step, then does all thematic comparisons in-memory
    against the summary text — never opening any .md file from disk.

    At 10,000 nodes with 300-word summaries, this is ~3MB of JSON — fast to
    fetch and fast to scan in-context. Contrast with reading 10,000 .md files
    which could be 200MB+ and take minutes.
    """
    close = conn is None
    if close:
        conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT id, filename, summary FROM nodes WHERE summary IS NOT NULL ORDER BY filename"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        if close:
            conn.close()


# ---------------------------------------------------------------------------
# Link operations
# ---------------------------------------------------------------------------

def upsert_link(
    source_id: str,
    target_id: str,
    reason: str = "",
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    sql = """
    INSERT INTO links (source_id, target_id, reason)
    VALUES (?, ?, ?)
    ON CONFLICT(source_id, target_id) DO UPDATE SET reason = excluded.reason
    """
    close = conn is None
    if close:
        conn = get_conn()
    try:
        conn.execute(sql, (source_id, target_id, reason))
        _refresh_linked_ids(source_id, conn)
        if close:
            conn.commit()
    finally:
        if close:
            conn.close()


def delete_link(source_id: str, target_id: str, conn: Optional[sqlite3.Connection] = None) -> None:
    close = conn is None
    if close:
        conn = get_conn()
    try:
        conn.execute("DELETE FROM links WHERE source_id = ? AND target_id = ?", (source_id, target_id))
        _refresh_linked_ids(source_id, conn)
        if close:
            conn.commit()
    finally:
        if close:
            conn.close()


def _refresh_linked_ids(source_id: str, conn: sqlite3.Connection) -> None:
    """Keep nodes.linked_ids JSON in sync with the links table."""
    rows = conn.execute("SELECT target_id FROM links WHERE source_id = ?", (source_id,)).fetchall()
    ids = json.dumps([r["target_id"] for r in rows])
    conn.execute("UPDATE nodes SET linked_ids = ? WHERE id = ?", (ids, source_id))


def list_links(conn: Optional[sqlite3.Connection] = None) -> list[dict]:
    close = conn is None
    if close:
        conn = get_conn()
    try:
        rows = conn.execute("""
            SELECT l.id, l.source_id, l.target_id, l.reason, l.created_at,
                   s.filename AS source_file, t.filename AS target_file
            FROM links l
            JOIN nodes s ON s.id = l.source_id
            JOIN nodes t ON t.id = l.target_id
            ORDER BY l.created_at DESC
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        if close:
            conn.close()


def broken_links(conn: Optional[sqlite3.Connection] = None) -> list[dict]:
    """Links whose source or target file no longer exists on disk."""
    all_links = list_links(conn)
    broken = []
    for lnk in all_links:
        if not (RAW_DIR / Path(lnk["source_file"]).name).exists():
            broken.append({**lnk, "broken_side": "source"})
        elif not (RAW_DIR / Path(lnk["target_file"]).name).exists():
            broken.append({**lnk, "broken_side": "target"})
    return broken


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_json(obj):
    print(json.dumps(obj, indent=2, default=str))


def main(argv: list[str]) -> None:
    if not argv:
        print(__doc__)
        return

    cmd, *args = argv

    if cmd == "init":
        init_db()

    elif cmd == "upsert-node":
        data = json.loads(args[0])
        node_id = upsert_node(data)
        print(f"Upserted node: {node_id}")

    elif cmd == "set-summary":
        node_id, summary = args[0], args[1]
        set_summary(node_id, summary)
        print(f"Summary set for: {node_id}")

    elif cmd == "upsert-link":
        source_id, target_id = args[0], args[1]
        reason = args[2] if len(args) > 2 else ""
        upsert_link(source_id, target_id, reason)
        print(f"Link upserted: {source_id} → {target_id}")

    elif cmd == "delete-link":
        delete_link(args[0], args[1])
        print(f"Link deleted: {args[0]} → {args[1]}")

    elif cmd == "list-nodes":
        orphans = "--orphans" in args
        no_sum = "--no-summary" in args
        _print_json(list_nodes(orphans_only=orphans, no_summary=no_sum))

    elif cmd == "list-links":
        _print_json(list_links())

    elif cmd == "broken-links":
        _print_json(broken_links())

    elif cmd == "stale-nodes":
        days = 7
        if "--days" in args:
            days = int(args[args.index("--days") + 1])
        _print_json(stale_nodes(days))

    elif cmd == "get-node":
        _print_json(get_node(args[0]))

    elif cmd == "node-id":
        print(node_id_for(args[0]))

    elif cmd == "last-seen-max":
        # Prints the ISO timestamp of the most recently seen node.
        # Used by the CLI at the start of `update` to find the incremental baseline.
        # Output is a plain string (not JSON) so it can be shell-interpolated directly:
        #   python scripts/db_utils.py list-files --since "$(python scripts/db_utils.py last-seen-max)"
        print(last_seen_max())

    elif cmd == "list-summaries":
        # Returns [{id, filename, summary}] for all nodes with a summary.
        # The CLI fetches this ONCE per update/lint run and does all thematic
        # comparison in-memory — no per-file disk reads for existing nodes.
        _print_json(list_summaries())

    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
