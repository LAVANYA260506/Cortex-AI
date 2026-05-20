"""
retriever/search.py
───────────────────
Step 2 of the retrieval pipeline: Multi-seed Qdrant similarity search.

Instead of picking a single "best" candidate and betting the entire pipeline
on it, this module returns ALL Qdrant hits above SEED_MIN_SCORE as seeds.
Every qualifying seed becomes a root node for the semantic BFS in graph.py.

Why multi-seed matters
----------------------
With a single seed, a mis-ranked Qdrant result (common for short or ambiguous
queries) derails the entire pipeline.  With multiple seeds the BFS has several
starting points, so the probability of having at least one correct root rises
sharply — especially after query expansion creates a richer query vector.

Each returned seed dict carries `qdrant_rank` so that fusion.py can use
Qdrant's ranking signal in RRF without re-querying.
"""

from __future__ import annotations

import logging

from .config import COLLECTION_NAME, SEED_MIN_SCORE, TOP_K_SEEDS
from .singletons import _get_qdrant

log = logging.getLogger(__name__)


def multi_seed_search(
    query_vector:  list[float],
    top_k:         int   = TOP_K_SEEDS,
    min_score:     float = SEED_MIN_SCORE,
) -> list[dict]:
    """
    Query Qdrant and return all hits that exceed *min_score*.

    Parameters
    ----------
    query_vector : list[float]
        Max-pooled query embedding from query_expander.embed_query().
    top_k : int
        Maximum number of candidates to request from Qdrant.
    min_score : float
        Cosine-similarity floor.  Candidates below this are silently dropped.

    Returns
    -------
    list[dict]  — sorted descending by score, each entry:
        {
            "node_id"     : str,
            "filename"    : str,
            "summary"     : str,
            "score"       : float,   # Qdrant cosine similarity
            "qdrant_rank" : int,     # 0 = best Qdrant hit
        }
    """
    client = _get_qdrant()

    hits = client.search(
        collection_name = COLLECTION_NAME,
        query_vector    = query_vector,
        limit           = top_k,
        with_payload    = True,
    )

    seeds = []
    for rank, hit in enumerate(hits):
        score = round(hit.score, 4)
        if score < min_score:
            log.debug(
                "Candidate rank=%d score=%.3f dropped (below floor %.2f)",
                rank, score, min_score,
            )
            continue
        seeds.append({
            "node_id":     hit.payload.get("node_id", str(hit.id)),
            "filename":    hit.payload.get("filename", ""),
            "summary":     hit.payload.get("summary", ""),
            "score":       score,
            "qdrant_rank": rank,
        })

    log.info(
        "Multi-seed search: %d/%d candidates above threshold %.2f  [scores: %s]",
        len(seeds), top_k, min_score,
        ", ".join(f"{s['score']:.3f}" for s in seeds),
    )
    return seeds
