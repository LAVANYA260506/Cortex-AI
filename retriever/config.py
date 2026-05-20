"""
retriever/config.py
───────────────────
Single source of truth for every tunable constant and the shared ScoredNode
dataclass.  Import from here; never hard-code values in other modules.

Changing a value here propagates to the entire pipeline automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths & model
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME      = "all-MiniLM-L6-v2"   # must match embedder.py
QDRANT_DIR      = Path("qdrant_db")
COLLECTION_NAME = "kg_summaries"
DB_PATH         = Path("kg.db")
RAW_DIR         = Path("raw")

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Query expansion
# ─────────────────────────────────────────────────────────────────────────────

EXPANSION_ENABLED = True   # set False to skip LLM sub-question generation
MAX_SUBQUESTIONS  = 3      # extra sub-questions generated per query

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Multi-seed Qdrant search
# ─────────────────────────────────────────────────────────────────────────────

TOP_K_SEEDS    = 5     # max candidates pulled from Qdrant
SEED_MIN_SCORE = 0.28  # cosine floor; seeds below this are dropped

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Semantic BFS
# ─────────────────────────────────────────────────────────────────────────────

MAX_BFS_NODES        = 20    # hard cap on nodes visited across ALL seeds
BFS_MAX_DEPTH        = 3     # hop limit from any seed
EDGE_DECAY           = 0.75  # priority multiplied by this per hop
SEMANTIC_DECAY_FLOOR = 0.10  # prune branch when priority drops below this

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Reciprocal Rank Fusion
# ─────────────────────────────────────────────────────────────────────────────

RRF_K = 60   # standard RRF constant (higher → flatter, more robust fusion)

# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Semantic re-rank
# ─────────────────────────────────────────────────────────────────────────────

RERANK_TOP_N = 10   # nodes considered for re-rank (top slice after RRF)
FINAL_TOP_N  = 5    # nodes passed to the LLM after re-rank

# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Context assembly
# ─────────────────────────────────────────────────────────────────────────────

MAX_CONTEXT_WORDS = 6000   # word-proxy budget; lowest-scored nodes trimmed first

# ─────────────────────────────────────────────────────────────────────────────
# Logging — configure once here, every module uses getLogger(__name__)
# ─────────────────────────────────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


# ─────────────────────────────────────────────────────────────────────────────
# Shared data structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoredNode:
    """
    Carries a graph node through every stage of the retrieval pipeline.

    Scores are set incrementally:
      qdrant_rank    — set by search.py        (lower = better)
      bfs_order      — set by graph.py         (lower = better)
      hop_depth      — set by graph.py
      semantic_score — set by fusion.py        (higher = better)
      rrf_score      — set by fusion.py; overwritten with final weighted score
    """
    node_id:        str
    filename:       str
    summary:        str

    qdrant_rank:    int   = 9999
    bfs_order:      int   = 9999
    semantic_score: float = 0.0
    rrf_score:      float = 0.0
    hop_depth:      int   = 0

    # Lazily populated in fusion.py during re-rank; excluded from repr
    _summary_vec: list = field(default_factory=list, repr=False)

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ScoredNode) and self.node_id == other.node_id
