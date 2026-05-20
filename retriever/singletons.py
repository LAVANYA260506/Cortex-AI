from __future__ import annotations

import logging
import sqlite3

from .config import DB_PATH, MODEL_NAME, QDRANT_DIR, COLLECTION_NAME

log = logging.getLogger(__name__)

_encoder = None
_qdrant  = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading model: %s", MODEL_NAME)
        _encoder = SentenceTransformer(MODEL_NAME)
    return _encoder


def _get_qdrant():
    global _qdrant
    if _qdrant is None:
        from qdrant_client import QdrantClient
        QDRANT_DIR.mkdir(parents=True, exist_ok=True)
        _qdrant = QdrantClient(path=str(QDRANT_DIR))

        existing = {c.name for c in _qdrant.get_collections().collections}
        if COLLECTION_NAME not in existing:
            from qdrant_client.models import Distance, VectorParams
            _qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            log.info("Created collection: %s", COLLECTION_NAME)
    return _qdrant


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn