#!/usr/bin/env python3
"""
job_queue.py — File-based job queue between Gemini CLI and the embedder daemon.

Protocol
--------
The CLI writes jobs; the embedder reads and drains them. They never call each other.

Queue file: embed_queue.jsonl  (one JSON object per line, append-only by CLI)
Done file:  embed_done.jsonl   (embedder appends completed job IDs here for audit)

Each job line:
    {"node_id": "<uuid>", "filename": "<raw/...>", "summary": "<text>", "enqueued_at": "<iso>"}

The embedder drains the queue atomically:
  1. Renames embed_queue.jsonl → embed_queue.jsonl.processing  (atomic on POSIX)
  2. Processes all jobs in the renamed file.
  3. Appends completed job_ids to embed_done.jsonl.
  4. Deletes embed_queue.jsonl.processing.

This means:
  - The CLI can keep appending to embed_queue.jsonl while the embedder is busy.
  - No job is ever lost: if the embedder crashes mid-batch, the .processing file
    stays on disk and is resumed on the next startup.
  - No locking primitives needed — os.rename() is atomic on Linux/macOS.

Usage (CLI via subprocess):
    python scripts/job_queue.py enqueue <node_id> <filename> <summary>
    python scripts/job_queue.py peek          # show pending count
    python scripts/job_queue.py list          # print all pending jobs as JSON

Usage (embedder imports directly):
    from job_queue import dequeue_batch, mark_done, recover_stale
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

QUEUE_FILE = Path("embed_queue.jsonl")
PROCESSING_FILE = Path("embed_queue.jsonl.processing")
DONE_FILE = Path("embed_done.jsonl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    jobs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                jobs.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # skip corrupt lines
    return jobs


# ---------------------------------------------------------------------------
# CLI (Gemini) side — write
# ---------------------------------------------------------------------------

def enqueue(node_id: str, filename: str, summary: str) -> None:
    """
    Append one embed job to the queue file.
    Safe to call from multiple processes — append is atomic for small writes
    on POSIX (lines < PIPE_BUF ≈ 4 KB, which summaries always are).
    """
    job = {
        "node_id": node_id,
        "filename": filename,
        "summary": summary,
        "enqueued_at": _now(),
    }
    with QUEUE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(job) + "\n")


def pending_count() -> int:
    return len(_read_jsonl(QUEUE_FILE))


def list_pending() -> list[dict]:
    return _read_jsonl(QUEUE_FILE)


# ---------------------------------------------------------------------------
# Embedder side — read / drain
# ---------------------------------------------------------------------------

def recover_stale() -> int:
    """
    Call once at embedder startup. If a .processing file exists from a previous
    crashed run, rename it back to the queue file so it gets re-processed.
    Any jobs already in the live queue are appended after the recovered ones.
    Returns the number of recovered jobs.
    """
    if not PROCESSING_FILE.exists():
        return 0

    recovered = _read_jsonl(PROCESSING_FILE)
    if not recovered:
        PROCESSING_FILE.unlink(missing_ok=True)
        return 0

    # Merge: recovered jobs first, then any new ones already in the live queue
    live = _read_jsonl(QUEUE_FILE)
    merged = recovered + live

    # Write merged back to queue, then remove processing file
    QUEUE_FILE.write_text(
        "\n".join(json.dumps(j) for j in merged) + "\n",
        encoding="utf-8",
    )
    PROCESSING_FILE.unlink()
    return len(recovered)


def dequeue_batch() -> list[dict]:
    """
    Atomically move the current queue into a processing file and return its jobs.
    Returns an empty list if the queue is empty.
    The caller MUST call mark_done() or re-enqueue() for each job when finished.
    """
    if not QUEUE_FILE.exists() or QUEUE_FILE.stat().st_size == 0:
        return []

    # Atomic rename — the CLI can keep writing to QUEUE_FILE immediately after
    try:
        os.rename(QUEUE_FILE, PROCESSING_FILE)
    except FileNotFoundError:
        return []  # queue was emptied by a concurrent rename (race guard)

    return _read_jsonl(PROCESSING_FILE)


def mark_done(jobs: list[dict]) -> None:
    """
    Append completed jobs to the audit log and remove the processing file.
    Call this after ALL jobs in a batch have been successfully embedded.
    """
    if jobs:
        with DONE_FILE.open("a", encoding="utf-8") as f:
            for job in jobs:
                done_entry = {**job, "completed_at": _now()}
                f.write(json.dumps(done_entry) + "\n")

    PROCESSING_FILE.unlink(missing_ok=True)


def requeue_failed(jobs: list[dict]) -> None:
    """
    Re-append jobs that failed back to the live queue for retry.
    Call this in an except block when a batch partially fails.
    """
    with QUEUE_FILE.open("a", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(job) + "\n")
    PROCESSING_FILE.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> None:
    if not argv:
        print(__doc__)
        return

    cmd, *args = argv

    if cmd == "enqueue":
        node_id, filename, summary = args[0], args[1], args[2]
        enqueue(node_id, filename, summary)
        print(f"Enqueued: {node_id[:8]}… ({Path(filename).name})")

    elif cmd == "peek":
        count = pending_count()
        print(f"Pending jobs: {count}")

    elif cmd == "list":
        print(json.dumps(list_pending(), indent=2))

    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
