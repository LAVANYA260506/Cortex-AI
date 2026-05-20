"""
retriever/__init__.py
─────────────────────
Public API for the retriever package.

Import surface
--------------
    from retriever import retrieve, ask          # most common
    from retriever import ScoredNode             # for type annotations
    from retriever.config import TOP_K_SEEDS     # for configuration overrides

The `retrieve` function runs steps 1-6 and returns a rich trace dict.
The `ask` function runs the full pipeline including the LLM (step 7).

Both are importable here so callers never need to know the internal
module structure.  Internal modules should import from each other
directly (e.g. `from .singletons import _get_encoder`) — never through
this __init__ — to avoid circular imports.
"""

from __future__ import annotations

import logging

from .config import (
    MAX_BFS_NODES,
    SEED_MIN_SCORE,
    TOP_K_SEEDS,
    ScoredNode,
)
from .context import build_context
from .fusion import reciprocal_rank_fusion, semantic_rerank
from .generator import generate_answer
from .graph import semantic_bfs
from .query_expander import embed_query
from .search import multi_seed_search

log = logging.getLogger(__name__)

__all__ = ["retrieve", "ask", "ScoredNode"]


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline orchestration
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(
    query:          str,
    top_k:          int   = TOP_K_SEEDS,
    max_nodes:      int   = MAX_BFS_NODES,
    seed_min_score: float = SEED_MIN_SCORE,
    expand:         bool  = True,
) -> dict:
    """
    Run the full retrieval pipeline (steps 1–6) and return a trace dict.

    Steps
    -----
    1. Query expansion + max-pool embedding   (query_expander)
    2. Multi-seed Qdrant search               (search)
    3. Semantic priority-queue BFS            (graph)
    4. Reciprocal Rank Fusion                 (fusion)
    5. Semantic re-rank                       (fusion)
    6. Context assembly                       (context)

    Returns
    -------
    dict with keys:
        query         str
        variants      list[str]          query + sub-questions
        query_vector  list[float]        max-pooled embedding
        seeds         list[dict]         Qdrant multi-seed hits
        bfs_nodes     list[ScoredNode]   raw BFS output
        rrf_nodes     list[ScoredNode]   after RRF fusion
        final_nodes   list[ScoredNode]   after semantic re-rank
        context       str                assembled Markdown for LLM

    The trace dict is designed to be inspectable: every intermediate
    result is preserved so callers can debug, log, or visualise the
    pipeline without re-running it.
    """
    # ── Step 1: embed ─────────────────────────────────────────────────────────
    query_vector, variants = embed_query(query, expand=expand)

    # ── Step 2: multi-seed search ─────────────────────────────────────────────
    seeds = multi_seed_search(query_vector, top_k=top_k, min_score=seed_min_score)
    if not seeds:
        log.warning("No seeds above threshold — retrieval aborted.")
        return _empty_result(query, variants, query_vector)

    # ── Step 3: semantic BFS ──────────────────────────────────────────────────
    bfs_nodes = semantic_bfs(seeds, query_vector, max_nodes=max_nodes)
    if not bfs_nodes:
        log.warning("BFS returned no nodes — retrieval aborted.")
        return _empty_result(query, variants, query_vector, seeds=seeds)

    # ── Step 4: RRF ───────────────────────────────────────────────────────────
    rrf_nodes = reciprocal_rank_fusion(bfs_nodes)

    # ── Step 5: semantic re-rank ──────────────────────────────────────────────
    final_nodes = semantic_rerank(rrf_nodes, query_vector)

    # ── Step 6: context assembly ──────────────────────────────────────────────
    context = build_context(final_nodes)

    return {
        "query":        query,
        "variants":     variants,
        "query_vector": query_vector,
        "seeds":        seeds,
        "bfs_nodes":    bfs_nodes,
        "rrf_nodes":    rrf_nodes,
        "final_nodes":  final_nodes,
        "context":      context,
    }


def ask(
    query:          str,
    top_k:          int   = TOP_K_SEEDS,
    max_nodes:      int   = MAX_BFS_NODES,
    seed_min_score: float = SEED_MIN_SCORE,
    expand:         bool  = True,
) -> str:
    """
    End-to-end: retrieve context then generate a cited LLM answer.

    Calls retrieve() internally; if no relevant nodes are found,
    returns a clean "not found" message without making a Sonnet API call.

    Returns
    -------
    str — the LLM's answer with inline source citations.
    """
    result = retrieve(
        query,
        top_k          = top_k,
        max_nodes       = max_nodes,
        seed_min_score  = seed_min_score,
        expand          = expand,
    )

    if not result["final_nodes"]:
        return (
            "The knowledge base does not contain enough information to answer "
            "this question.  Try rephrasing, or verify that relevant documents "
            "have been ingested."
        )

    return generate_answer(
        query,
        result["context"],
        query_variants=result["variants"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _empty_result(
    query:        str,
    variants:     list[str],
    query_vector: list[float],
    seeds:        list[dict] | None = None,
) -> dict:
    return {
        "query":        query,
        "variants":     variants,
        "query_vector": query_vector,
        "seeds":        seeds or [],
        "bfs_nodes":    [],
        "rrf_nodes":    [],
        "final_nodes":  [],
        "context":      "",
    }
