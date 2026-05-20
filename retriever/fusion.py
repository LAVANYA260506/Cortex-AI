"""
retriever/fusion.py
───────────────────
Steps 4 and 5 of the retrieval pipeline.

Step 4 — Reciprocal Rank Fusion (RRF)
--------------------------------------
Takes the raw BFS output and fuses three independent ranked lists into a
single unified score per node:

    List A  Qdrant vector rank       (dense retrieval signal)
    List B  BFS visit order          (graph proximity to query)
    List C  Hop depth (ascending)    (structural closeness to a seed)

RRF formula for node n:
    rrf(n) = Σ  1 / (k + rank_i(n) + 1)
             i ∈ {A, B, C}

where k = RRF_K (default 60, the standard constant from the original paper).



Step 5 — Semantic re-rank
--------------------------
After RRF, the top RERANK_TOP_N nodes are re-scored with exact cosine
similarity between each node's summary vector and the query vector.

Final score = 0.6 × semantic_score + 0.4 × normalised_rrf_score

The 60/40 split:
  - Gives semantic relevance primacy (it is the ground truth signal).
  - Keeps graph structure (RRF) as a tie-breaker and diversity signal.
  - Prevents a graph-adjacent but semantically weak node from winning.

Only FINAL_TOP_N nodes survive re-rank and proceed to context assembly.
"""

from __future__ import annotations

import logging

from .config import FINAL_TOP_N, RERANK_TOP_N, RRF_K, ScoredNode
from .math_utils import cosine
from .singletons import _get_encoder

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — RRF
# ─────────────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    nodes: list[ScoredNode],
    k:     int = RRF_K,
) -> list[ScoredNode]:
    """
    Fuse Qdrant rank, BFS order, and hop depth into a single RRF score.

    Mutates node.rrf_score in place, then returns the list sorted descending
    by rrf_score.

    Parameters
    ----------
    nodes : list[ScoredNode]
        All nodes collected by graph.semantic_bfs().
    k : int
        RRF smoothing constant (default 60).

    Returns
    -------
    list[ScoredNode] — same nodes, sorted best-first by rrf_score.
    """
    if not nodes:
        return []

    # Build per-signal rank maps (rank 0 = best in that signal)
    by_qdrant = sorted(nodes, key=lambda n: n.qdrant_rank)          # lower rank = better
    by_bfs    = sorted(nodes, key=lambda n: n.bfs_order)            # visited earlier = better
    by_depth  = sorted(nodes, key=lambda n: n.hop_depth)            # shallower = better

    qdrant_rank_map = {n.node_id: i for i, n in enumerate(by_qdrant)}
    bfs_rank_map    = {n.node_id: i for i, n in enumerate(by_bfs)}
    depth_rank_map  = {n.node_id: i for i, n in enumerate(by_depth)}

    for node in nodes:
        node.rrf_score = round(
            1.0 / (k + qdrant_rank_map[node.node_id] + 1)
            + 1.0 / (k + bfs_rank_map[node.node_id]    + 1)
            + 1.0 / (k + depth_rank_map[node.node_id]  + 1),
            6,
        )

    fused = sorted(nodes, key=lambda n: n.rrf_score, reverse=True)
    log.info(
        "RRF fusion: %d nodes, top scores = [%s]",
        len(fused),
        ", ".join(f"{n.rrf_score:.5f}" for n in fused[:5]),
    )
    return fused


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Semantic re-rank
# ─────────────────────────────────────────────────────────────────────────────

def semantic_rerank(
    nodes:        list[ScoredNode],
    query_vector: list[float],
    top_n:        int = RERANK_TOP_N,
    final_n:      int = FINAL_TOP_N,
) -> list[ScoredNode]:
    """
    Re-score the top *top_n* RRF nodes with exact cosine similarity, then
    return the best *final_n*.

    The re-rank fuses semantic score and normalised RRF score:
        final = 0.6 × semantic_score + 0.4 × (rrf / max_rrf)

    This overwrites node.rrf_score with the final weighted score (so
    downstream modules always read one authoritative score field).
    node.semantic_score is preserved separately for reporting.

    Parameters
    ----------
    nodes : list[ScoredNode]
        Output of reciprocal_rank_fusion(), sorted best-first.
    query_vector : list[float]
        Max-pooled query embedding.
    top_n : int
        How many top-RRF nodes to re-score (the rest are discarded early).
    final_n : int
        How many nodes to return after re-rank.

    Returns
    -------
    list[ScoredNode] — at most *final_n* nodes, sorted by final score.
    """
    candidates = nodes[:top_n]
    if not candidates:
        return []

    encoder   = _get_encoder()
    summaries = [n.summary or n.filename for n in candidates]
    vecs      = encoder.encode(summaries, show_progress_bar=False).tolist()

    # Compute exact cosine for each candidate
    for node, vec in zip(candidates, vecs):
        node._summary_vec   = vec
        node.semantic_score = round(cosine(vec, query_vector), 4)

    # Normalise RRF scores to [0, 1] before weighting
    max_rrf = max(n.rrf_score for n in candidates) or 1.0

    for node in candidates:
        norm_rrf     = node.rrf_score / max_rrf
        node.rrf_score = round(0.6 * node.semantic_score + 0.4 * norm_rrf, 6)

    reranked = sorted(candidates, key=lambda n: n.rrf_score, reverse=True)[:final_n]

    log.info(
        "Semantic re-rank: %d → %d nodes  [final scores: %s]",
        top_n, len(reranked),
        ", ".join(f"{n.rrf_score:.4f}" for n in reranked),
    )
    return reranked
