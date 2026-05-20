"""
retriever/context.py
────────────────────
Step 6 of the retrieval pipeline: Context assembly.

Reads the raw Markdown files for each re-ranked node, strips metadata noise
(YAML front-matter and overlap-context blocks), and concatenates them into
a single context string for the LLM.

Token budget
------------
MAX_CONTEXT_WORDS is a word-count proxy for token budget.  Nodes are added
in descending score order (best first) and the loop stops when the next node
would overflow the budget.  This means the LLM always sees the most relevant
content first, and lower-scored nodes are gracefully trimmed rather than
causing a hard API error.

Noise stripping
---------------
The raw .md files contain two types of metadata not useful to the LLM:

  1. YAML front-matter (---\n...\n---) — file-level metadata added by
     convert_docs_to_markdown.py (original_filename, processed_date, chunk).

  2. Overlap-context blocks (<!-- overlap-context --> ... <!-- /overlap-context -->)
     — cross-chunk overlap prepended by chunker.py.  The LLM has the full
     context across chunks via the retrieval pipeline, so this redundant
     overlap only wastes tokens.

Stripping both reduces per-node token cost by ~5-15%.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .config import MAX_CONTEXT_WORDS, ScoredNode

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Regex patterns — compiled once at import time
# ─────────────────────────────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(
    r"^---\n.*?\n---\n?",
    re.DOTALL,
)
_OVERLAP_RE = re.compile(
    r"<!-- overlap-context -->.*?<!-- /overlap-context -->\n*",
    re.DOTALL,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_metadata(text: str) -> str:
    """Remove YAML front-matter and overlap-context blocks."""
    text = _FRONTMATTER_RE.sub("", text, count=1)
    text = _OVERLAP_RE.sub("", text)
    return text.strip()


def _read_node_content(node: ScoredNode) -> str:
    """
    Read and clean the raw .md file for *node*.

    Falls back to the node's summary string if the file does not exist
    (e.g. the file was deleted after indexing).
    """
    filepath = Path(node.filename)
    if filepath.exists():
        raw = filepath.read_text(encoding="utf-8")
        return _strip_metadata(raw)

    log.warning(
        "File not found: %s — falling back to summary.",
        filepath,
    )
    return node.summary or ""


# ─────────────────────────────────────────────────────────────────────────────
# Public function
# ─────────────────────────────────────────────────────────────────────────────

def build_context(
    nodes:     list[ScoredNode],
    max_words: int = MAX_CONTEXT_WORDS,
) -> str:
    """
    Assemble a single context string from re-ranked nodes.

    Nodes are processed in score-descending order (caller should pass them
    already sorted).  Each node's content is prefixed with a header that
    includes the filename and final score so the LLM can cite sources.

    If adding the next node would exceed *max_words*, that node and all
    lower-scored nodes are dropped and a warning is logged.

    Parameters
    ----------
    nodes : list[ScoredNode]
        Re-ranked nodes from fusion.semantic_rerank(), best first.
    max_words : int
        Hard word-count budget for the assembled context.

    Returns
    -------
    str — multi-section Markdown string, separated by horizontal rules.
    """
    parts:   list[str] = []
    running: int       = 0

    for node in nodes:
        content = _read_node_content(node)
        wc      = len(content.split())

        if running + wc > max_words and parts:
            log.info(
                "Context budget reached (%d words) — dropping %s (%d words).",
                running, Path(node.filename).name, wc,
            )
            break

        header = (
            f"### [{Path(node.filename).name}]"
            f"  <!-- score={node.rrf_score:.4f}"
            f" sem={node.semantic_score:.4f}"
            f" depth={node.hop_depth} -->"
        )
        parts.append(f"{header}\n\n{content}")
        running += wc

    log.info("Context assembled: %d section(s), ~%d words.", len(parts), running)
    return "\n\n---\n\n".join(parts)
