"""
retriever/math_utils.py
───────────────────────
Pure-Python math helpers used across the pipeline.

No numpy, no torch, no external dependencies — only stdlib `math`.
These functions operate on plain Python lists of floats, which is
appropriate for the small vector counts in re-rank (≤ RERANK_TOP_N nodes).

For large batch operations the pipeline uses sentence-transformers' own
numpy-backed `.encode()` directly; these helpers are for the final
scalar comparisons where importing numpy would be overkill.
"""

from __future__ import annotations

import math


def cosine(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity between two equal-length float vectors.

    Returns a value in [-1, 1]; returns 0.0 for zero-magnitude vectors
    rather than raising ZeroDivisionError.

    Used in:
      - fusion.py  : exact re-rank of top-N node summaries vs query vector
      - graph.py   : per-neighbour priority scoring during semantic BFS
    """
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))

    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def max_pool(vectors: list[list[float]]) -> list[float]:
    """
    Element-wise maximum across a list of equal-length float vectors.

    This is the pooling strategy used in query_expander.py to merge the
    embeddings of the original query and its sub-question expansions into
    a single representative vector.

    Why max-pool instead of mean?
      Mean-pooling averages features, diluting strong signals from any one
      variant.  Max-pooling preserves the peak activation for each dimension,
      so the resulting vector has high similarity to *all* variants rather
      than being mediocre toward all of them.

    Args:
        vectors: non-empty list of vectors of identical length.

    Returns:
        A single vector of the same length.
    """
    if not vectors:
        raise ValueError("Cannot max-pool an empty list of vectors.")
    if len(vectors) == 1:
        return list(vectors[0])

    dim    = len(vectors[0])
    pooled = list(vectors[0])

    for vec in vectors[1:]:
        if len(vec) != dim:
            raise ValueError(f"Inconsistent vector lengths: expected {dim}, got {len(vec)}")
        for i, v in enumerate(vec):
            if v > pooled[i]:
                pooled[i] = v

    return pooled
