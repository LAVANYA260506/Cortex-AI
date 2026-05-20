#!/usr/bin/env python3
"""
embedder.py — Standalone embedding daemon for the knowledge graph pipeline.

Runs in Terminal 3. Completely decoupled from Gemini CLI.

Trigger mechanism
-----------------
The CLI writes summaries to kg.db AND appends a line to embed_queue.jsonl
via scripts/job_queue.py. This daemon polls embed_queue.jsonl every
POLL_INTERVAL seconds, atomically drains it, encodes with sentence-transformers,
and upserts into a local Qdrant collection.

The CLI never calls this process. This process never calls the CLI.
The only shared artefact is embed_queue.jsonl (written by CLI, read here).

Qdrant
------
Uses qdrant-client in local (on-disk) mode — no Qdrant server needed.
Collection: "kg_summaries", cosine distance, vector size 384 (MiniLM-L6-v2).
Each point: id (UUID), vector, payload={filename, summary, node_id}.

Run:
    python embedder.py run

Search:
    python embedder.py search "your query here" --n 5

Install dependencies:
    pip install sentence-transformers qdrant-client
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from pathlib import Path
from qdrant_client.models import NamedVector

LOG_FORMAT = "%(asctime)s [%(levelname)s] embedder — %(message)s"
POLL_INTERVAL = 10            # seconds between queue polls
QDRANT_DIR = Path("qdrant_db")
COLLECTION_NAME = "kg_summaries"
VECTOR_SIZE = 384             # all-MiniLM-L6-v2 output dim
MODEL_NAME = "all-MiniLM-L6-v2"

# Add scripts/ to path so we can import job_queue
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
import scripts.job_queue as job_queue  # noqa: E402


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


# ---------------------------------------------------------------------------
# Lazy model + Qdrant client loading
# ---------------------------------------------------------------------------

_encoder = None
_qdrant = None


def get_encoder():
    global _encoder
    if _encoder is None:
        logging.info("Loading sentence-transformer model: %s", MODEL_NAME)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            ) from exc
        _encoder = SentenceTransformer(MODEL_NAME)
        logging.info("Model loaded. Vector size: %d", VECTOR_SIZE)
    return _encoder


def get_qdrant():
    global _qdrant
    if _qdrant is None:
        logging.info("Opening Qdrant local store at: %s", QDRANT_DIR)
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client not installed. Run: pip install qdrant-client"
            ) from exc

        QDRANT_DIR.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(QDRANT_DIR))

        # Create collection if it doesn't exist yet
        existing = {c.name for c in client.get_collections().collections}
        if COLLECTION_NAME not in existing:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            logging.info("Created Qdrant collection: %s", COLLECTION_NAME)
        else:
            count = client.get_collection(COLLECTION_NAME).points_count
            logging.info("Qdrant collection '%s' ready. Points: %d", COLLECTION_NAME, count)

        _qdrant = client
    return _qdrant


# ---------------------------------------------------------------------------
# Embedding cycle
# ---------------------------------------------------------------------------

def embed_batch(jobs: list[dict]) -> tuple[int, list[dict]]:
    """
    Encode summaries and upsert into Qdrant.
    Returns (success_count, failed_jobs).
    """
    if not jobs:
        return 0, []

    try:
        from qdrant_client.models import PointStruct
    except ImportError as exc:
        raise RuntimeError("qdrant-client not installed") from exc

    encoder = get_encoder()
    client = get_qdrant()

    summaries = [j["summary"] for j in jobs]
    vectors = encoder.encode(summaries, show_progress_bar=False).tolist()

    points = [
        PointStruct(
            id=job["node_id"],   # UUID string — Qdrant accepts this natively
            vector=vector,
            payload={
                "node_id": job["node_id"],
                "filename": job["filename"],
                "summary": job["summary"],
                "enqueued_at": job.get("enqueued_at", ""),
            },
        )
        for job, vector in zip(jobs, vectors)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)

    for job in jobs:
        logging.info(
            "  ✓ Embedded: %s  (%s)",
            Path(job["filename"]).name,
            job["node_id"][:8],
        )

    return len(jobs), []


def process_queue() -> int:
    """
    Drain the job queue, embed all jobs, return count processed.
    On failure, failed jobs are re-queued for retry next cycle.
    """
    jobs = job_queue.dequeue_batch()
    if not jobs:
        return 0

    logging.info("Dequeued %d job(s) from %s", len(jobs), job_queue.QUEUE_FILE)

    try:
        success_count, failed = embed_batch(jobs)
        job_queue.mark_done(jobs)
        if failed:
            logging.warning("Re-queuing %d failed job(s)", len(failed))
            job_queue.requeue_failed(failed)
        return success_count
    except Exception:
        logging.exception("Batch failed — re-queuing all %d job(s) for retry", len(jobs))
        job_queue.requeue_failed(jobs)
        return 0


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------

def search(query: str, n_results: int = 5) -> list[dict]:
    """
    Return the top-n most semantically similar nodes to `query`.
    Results: [{node_id, filename, summary, score}]
    """
    encoder = get_encoder()
    client = get_qdrant()

    vector = encoder.encode([query])[0].tolist()
        
    results = client.query_points(
        collection_name = COLLECTION_NAME,
        query           = query_vector,
        limit           = top_k,
        with_payload    = True,
    )
    hits = results.points
    return [
        {
            "node_id": h.payload.get("node_id", str(h.id)),
            "filename": h.payload.get("filename", ""),
            "summary": h.payload.get("summary", ""),
            "score": round(h.score, 4),
        }
        for h in hits
    ]


# ---------------------------------------------------------------------------
# Daemon loop
# ---------------------------------------------------------------------------

def run_daemon() -> None:
    setup_logging()
    logging.info(
        "Embedder daemon starting. Queue: %s | Qdrant: %s | Model: %s",
        job_queue.QUEUE_FILE, QDRANT_DIR, MODEL_NAME,
    )

    # Recover any jobs left over from a previous crashed run
    recovered = job_queue.recover_stale()
    if recovered:
        logging.info("Recovered %d job(s) from previous crashed run.", recovered)

    while True:
        try:
            count = process_queue()
            if count == 0:
                logging.debug("Queue empty. Sleeping %ds.", POLL_INTERVAL)
        except Exception:
            logging.exception("Unexpected error in daemon loop — continuing.")
        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Knowledge graph embedding daemon (Qdrant)")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("run", help="Start the embedding daemon (default)")

    search_p = sub.add_parser("search", help="Similarity search the Qdrant collection")
    search_p.add_argument("query", nargs="+", help="Search query text")
    search_p.add_argument("--n", type=int, default=5, help="Number of results")

    sub.add_parser("queue-status", help="Show pending job count in embed_queue.jsonl")

    args = parser.parse_args()

    if args.cmd == "search":
        results = search(" ".join(args.query), args.n)
        print(json.dumps(results, indent=2))

    elif args.cmd == "queue-status":
        count = job_queue.pending_count()
        print(f"Pending jobs in queue: {count}")
        if count:
            print(json.dumps(job_queue.list_pending(), indent=2))

    else:
        run_daemon()
