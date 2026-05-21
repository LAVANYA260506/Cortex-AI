#!/usr/bin/env python3
"""
api.py — Thin FastAPI wrapper around the retriever pipeline.

One endpoint: POST /ask
  - Takes a query string (and optional tuning params) in the request body.
  - Runs the full retriever pipeline (expand → BFS → RRF → rerank → LLM).
  - Returns the LLM-generated answer + source filenames used.

Run
---
    pip install fastapi uvicorn
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Test in Postman
---------------
    POST  http://localhost:8000/ask
    Body (raw JSON):
    {
        "query": "What were the Q2 revenue highlights?"
    }

Optional body params:
    {
        "query":          "What were the Q2 revenue highlights?",
        "top_k":          5,
        "max_nodes":      20,
        "seed_min_score": 0.28,
        "expand":         true
    }

Health check:
    GET http://localhost:8000/health
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import the retriever package (must be in the same directory or on PYTHONPATH)
from retriever import ask, retrieve
from retriever.config import MAX_BFS_NODES, SEED_MIN_SCORE, TOP_K_SEEDS

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"   # use cached model, no HF network calls
os.environ["HF_HUB_OFFLINE"] = "1"

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] api — %(message)s",
)
log = logging.getLogger(__name__)



from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup: load model + open Qdrant before first request ──
    log.info("Pre-loading embedding model and Qdrant client...")
    from retriever.singletons import _get_encoder, _get_qdrant
    _get_encoder()   # loads sentence-transformer into memory
    _get_qdrant()    # opens Qdrant store
    log.info("Warm-up complete. API ready.")
    yield
    # ── Shutdown (nothing to clean up) ──

app = FastAPI(
    title       = "Knowledge Graph Retriever API",
    description = "Ask questions against the local knowledge graph.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str = Field(
        ...,
        min_length  = 3,
        description = "Natural language question to ask the knowledge base.",
        examples    = ["What were the Q2 revenue highlights?"],
    )
    top_k: int = Field(
        default     = TOP_K_SEEDS,
        ge          = 1,
        le          = 20,
        description = "Max Qdrant seed candidates (default: 5).",
    )
    max_nodes: int = Field(
        default     = MAX_BFS_NODES,
        ge          = 1,
        le          = 50,
        description = "BFS node cap across all seeds (default: 20).",
    )
    seed_min_score: float = Field(
        default     = SEED_MIN_SCORE,
        ge          = 0.0,
        le          = 1.0,
        description = "Minimum cosine score for a seed to be accepted (default: 0.28).",
    )
    expand: bool = Field(
        default     = True,
        description = "Whether to expand the query into sub-questions before retrieval.",
    )


class SourceNode(BaseModel):
    filename:       str
    semantic_score: float
    final_score:    float
    hop_depth:      int


class AskResponse(BaseModel):
    query:           str
    answer:          str
    sources:         list[SourceNode]
    query_variants:  list[str]
    nodes_retrieved: int
    took_ms:         float


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
def health():
    """Returns 200 OK if the API is running."""
    return {"status": "ok"}


@app.post(
    "/ask",
    response_model = AskResponse,
    summary        = "Ask a question",
    description    = (
        "Runs the full retrieval pipeline:\n\n"
        "1. Query expansion + max-pool embedding\n"
        "2. Multi-seed Qdrant similarity search\n"
        "3. Semantic priority-queue BFS over the knowledge graph\n"
        "4. Reciprocal Rank Fusion\n"
        "5. Semantic re-rank\n"
        "6. Context assembly\n"
        "7. Claude Sonnet answer generation with source citations"
    ),
)
def ask_question(body: AskRequest):
    log.info("Received query: %r  (expand=%s)", body.query, body.expand)
    t0 = time.perf_counter()

    try:
        result = retrieve(
            query          = body.query,
            top_k          = body.top_k,
            max_nodes       = body.max_nodes,
            seed_min_score  = body.seed_min_score,
            expand          = body.expand,
        )
    except Exception as exc:
        log.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}")

    final_nodes = result.get("final_nodes", [])

    if not final_nodes:
        took = round((time.perf_counter() - t0) * 1000, 1)
        return AskResponse(
            query           = body.query,
            answer          = (
                "The knowledge base does not contain enough information to answer "
                "this question. Try rephrasing, or verify relevant documents are ingested."
            ),
            sources         = [],
            query_variants  = result.get("variants", [body.query]),
            nodes_retrieved = 0,
            took_ms         = took,
        )

    # Generate the LLM answer
    try:
        from retriever.generator import generate_answer
        answer = generate_answer(
            body.query,
            result["context"],
            query_variants = result.get("variants"),
        )
    except Exception as exc:
        log.exception("LLM generation failed")
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}")

    took = round((time.perf_counter() - t0) * 1000, 1)
    log.info("Query answered in %.0f ms using %d node(s).", took, len(final_nodes))

    sources = [
        SourceNode(
            filename       = Path(n.filename).name,
            semantic_score = n.semantic_score,
            final_score    = n.rrf_score,
            hop_depth      = n.hop_depth,
        )
        for n in final_nodes
    ]

    return AskResponse(
        query           = body.query,
        answer          = answer,
        sources         = sources,
        query_variants  = result.get("variants", [body.query]),
        nodes_retrieved = len(result.get("bfs_nodes", [])),
        took_ms         = took,
    )