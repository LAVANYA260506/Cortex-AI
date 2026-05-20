"""
retriever/graph.py
──────────────────
Step 3 of the retrieval pipeline: Semantic BFS with a priority queue.

Classic BFS is blind — it expands all depth-1 neighbours before any depth-2
node regardless of relevance.  This module replaces it with a best-first
search where the traversal priority is:

    priority = cosine(neighbour_summary, query_vector) × EDGE_DECAY^depth

Key properties
--------------
* Semantically relevant neighbours are visited before irrelevant ones at the
  same depth — the traversal follows the query's meaning through the graph.
* EDGE_DECAY (< 1.0) applies a geometric distance penalty so the search does
  not sprawl indefinitely into weakly related territory.
* Branches whose priority drops below SEMANTIC_DECAY_FLOOR are pruned without
  being visited, saving DB round-trips and encoder calls.
* All seeds from search.py are pushed into the same priority queue, so the
  traversal is unified across seeds — no duplicated visits.
* Bidirectional edges: both outgoing and incoming links are traversed so the
  graph is treated as undirected (Gemini CLI writes bidirectional links anyway,
  but this guards against any asymmetry).

The ScoredNode objects returned carry qdrant_rank, bfs_order, and hop_depth
so that fusion.py can use all three signals in RRF without any extra queries.
"""

from __future__ import annotations

import heapq
import logging
from pathlib import Path

from .config import (
    BFS_MAX_DEPTH,
    EDGE_DECAY,
    MAX_BFS_NODES,
    SEMANTIC_DECAY_FLOOR,
)
from .config import ScoredNode
from .math_utils import cosine
from .singletons import _get_db, _get_encoder

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_neighbours(node_id: str, conn) -> list[dict]:
    """
    Return all neighbours of *node_id* via both outgoing and incoming links.
    Each row includes `reason` (the Gemini-written link annotation) in case
    future versions want to use it for edge weighting.
    """
    rows = conn.execute(
        """
        SELECT n.id, n.filename, n.summary, l.reason
        FROM   links l
        JOIN   nodes n ON n.id = l.target_id
        WHERE  l.source_id = ?

        UNION

        SELECT n.id, n.filename, n.summary, l.reason
        FROM   links l
        JOIN   nodes n ON n.id = l.source_id
        WHERE  l.target_id = ?
        """,
        (node_id, node_id),
    ).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Core BFS
# ─────────────────────────────────────────────────────────────────────────────

def semantic_bfs(
    seeds:        list[dict],
    query_vector: list[float],
    max_nodes:    int   = MAX_BFS_NODES,
    max_depth:    int   = BFS_MAX_DEPTH,
    edge_decay:   float = EDGE_DECAY,
    floor:        float = SEMANTIC_DECAY_FLOOR,
) -> list[ScoredNode]:
    """
    Priority-queue BFS from all seeds simultaneously.

    Parameters
    ----------
    seeds : list[dict]
        Output of search.multi_seed_search(). Each dict must have
        node_id, filename, summary, score (cosine), qdrant_rank.
    query_vector : list[float]
        Max-pooled query embedding used to score neighbours.
    max_nodes : int
        Hard cap on total nodes collected (across all seeds).
    max_depth : int
        Maximum hops from any seed node.
    edge_decay : float
        Per-hop priority multiplier (e.g. 0.75 → 25% penalty per hop).
    floor : float
        Prune any node whose priority falls below this value.

    Returns
    -------
    list[ScoredNode]  — in visit order (bfs_order reflects insertion sequence).
    """
    encoder = _get_encoder()
    conn    = _get_db()

    # heap entry: (-priority, tie_breaker_int, node_id, depth, qdrant_rank)
    heap:    list[tuple] = []
    counter: int         = 0   # monotonic tie-breaker; avoids comparing ScoredNode
    visited: dict[str, ScoredNode] = {}

    # ── Push seeds ────────────────────────────────────────────────────────────
    for seed in seeds:
        heapq.heappush(heap, (
            -seed["score"],    # negate: heapq is a min-heap
            counter,
            seed["node_id"],
            0,                 # depth = 0 for seeds
            seed["qdrant_rank"],
        ))
        counter += 1

    # ── Traversal ─────────────────────────────────────────────────────────────
    try:
        while heap and len(visited) < max_nodes:
            neg_prio, _, node_id, depth, qdrant_rank = heapq.heappop(heap)
            priority = -neg_prio

            if node_id in visited:
                continue
            if priority < floor:
                log.debug(
                    "BFS pruned: priority=%.4f < floor=%.4f at depth=%d",
                    priority, floor, depth,
                )
                # Once the heap top is below floor, all remaining entries are too
                break

            # Fetch full node from DB
            row = conn.execute(
                "SELECT id, filename, summary FROM nodes WHERE id = ?",
                (node_id,),
            ).fetchone()
            if not row:
                continue

            sn = ScoredNode(
                node_id     = row["id"],
                filename    = row["filename"],
                summary     = row["summary"] or "",
                qdrant_rank = qdrant_rank,
                bfs_order   = len(visited),
                hop_depth   = depth,
            )
            visited[node_id] = sn
            log.debug(
                "BFS #%d: %s  prio=%.4f  depth=%d",
                len(visited), Path(sn.filename).name, priority, depth,
            )

            if depth >= max_depth:
                continue

            # ── Expand neighbours ─────────────────────────────────────────────
            neighbours = _fetch_neighbours(node_id, conn)
            new_nbrs   = [n for n in neighbours if n["id"] not in visited]
            if not new_nbrs:
                continue

            # Batch-encode unseen neighbour summaries in one call
            texts    = [n["summary"] or n["filename"] for n in new_nbrs]
            nbr_vecs = encoder.encode(texts, show_progress_bar=False).tolist()

            for nbr, nbr_vec in zip(new_nbrs, nbr_vecs):
                sem_score    = cosine(nbr_vec, query_vector)
                nbr_priority = sem_score * (edge_decay ** (depth + 1))
                if nbr_priority < floor:
                    continue
                heapq.heappush(heap, (
                    -nbr_priority,
                    counter,
                    nbr["id"],
                    depth + 1,
                    9999,          # not a Qdrant seed; rank stays high
                ))
                counter += 1

    finally:
        conn.close()

    result = list(visited.values())
    log.info(
        "Semantic BFS complete: %d node(s) visited from %d seed(s).",
        len(result), len(seeds),
    )
    return result
