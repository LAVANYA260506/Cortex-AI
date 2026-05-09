"""
chunker.py — content-aware markdown chunker for the knowledge graph pipeline.

Public API
----------
smart_chunk(content: str, threshold: int = 3000, soft_tolerance: float = 0.10,
            overlap_sentences: int = 2) -> list[str]

Takes a markdown *string* (not a file path) and returns a list of chunk strings.
Each chunk is ready to be written as its own .md file.

Splitting strategy (priority order):
  1. docling page markers  <!-- page N -->
  2. Markdown H2 headings  ## ...        (PPTX slide headers land here)
  3. Markdown H3 headings  ### ...
  4. Blank-line paragraphs
  5. Hard split at threshold words        (last resort — no natural boundary found)

Overlap
-------
`overlap_sentences` trailing sentences from the previous chunk are prepended to
each new chunk so that cross-boundary context is not lost at retrieval time.
Set to 0 to disable.

Token counting
--------------
Uses a word-count proxy by default (fast, zero dependencies).
Swap `_word_count` for a tiktoken/Gemini tokenizer if you need precision.
"""

from __future__ import annotations

import re
from typing import List


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    """Cheap proxy: ~0.75× real token count for English prose."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Sentence extraction for overlap
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')


def _tail_sentences(text: str, n: int) -> str:
    """Return the last `n` sentences of `text` as an overlap header."""
    if n <= 0 or not text.strip():
        return ""
    sentences = _SENTENCE_SPLIT.split(text.strip())
    tail = sentences[-n:]
    return "\n".join(tail).strip()


# ---------------------------------------------------------------------------
# Boundary detection
# ---------------------------------------------------------------------------

# Priority 1 — docling page markers
_PAGE_MARKER = re.compile(r'(?=<!-- page \d+ -->)')

# Priority 2/3 — markdown headings
_H2 = re.compile(r'(?=^## )', re.MULTILINE)
_H3 = re.compile(r'(?=^### )', re.MULTILINE)

# Priority 4 — blank-line paragraph breaks
_PARAGRAPH = re.compile(r'\n{2,}')


def _split_by_pattern(text: str, pattern: re.Pattern) -> List[str]:
    parts = pattern.split(text)
    return [p for p in parts if p.strip()]


def _find_segments(text: str) -> List[str]:
    """
    Split text into the finest natural segments available,
    trying progressively coarser boundaries until we get > 1 segment.
    """
    for pattern in (_PAGE_MARKER, _H2, _H3):
        parts = _split_by_pattern(text, pattern)
        if len(parts) > 1:
            return parts

    # Paragraph split
    parts = [p.strip() for p in _PARAGRAPH.split(text) if p.strip()]
    if len(parts) > 1:
        return parts

    # No boundary found — return the whole thing as one segment
    return [text]


# ---------------------------------------------------------------------------
# Core chunker
# ---------------------------------------------------------------------------

def smart_chunk(
    content: str,
    threshold: int = 3000,
    soft_tolerance: float = 0.10,
    overlap_sentences: int = 2,
) -> List[str]:
    """
    Split `content` (a markdown string) into chunks of at most
    `threshold * (1 + soft_tolerance)` words.

    Parameters
    ----------
    content : str
        The full markdown text to chunk. NOT a file path.
    threshold : int
        Target maximum words per chunk (default 3000).
    soft_tolerance : float
        Fractional overage allowed before forcing a split (default 10%).
    overlap_sentences : int
        How many trailing sentences from the previous chunk to prepend to the
        next one for retrieval continuity (default 2, set 0 to disable).

    Returns
    -------
    list[str]
        One or more chunk strings. Single-element list if the content fits
        within the soft ceiling.
    """
    if not content or not content.strip():
        return []

    hard_ceil = int(threshold * (1 + soft_tolerance))
    total = _word_count(content)

    # Fast path — content fits in a single chunk
    if total <= hard_ceil:
        return [content]

    segments = _find_segments(content)

    chunks: List[str] = []
    current_segs: List[str] = []
    running = 0
    prev_overlap = ""

    for seg in segments:
        seg_words = _word_count(seg)

        # Single segment exceeds threshold on its own — hard split it
        # (compare against threshold, not hard_ceil, to leave room for overlap)
        if seg_words > threshold:
            # Flush any accumulated segments first
            if current_segs:
                chunk_text = _build_chunk(current_segs, prev_overlap)
                prev_overlap = _tail_sentences(chunk_text, overlap_sentences)
                chunks.append(chunk_text)
                current_segs = []
                running = 0

            # Hard-split the oversized segment by words.
            # Each sub is its own chunk; overlap only bridges between subs,
            # not accumulated across all of them.
            sub_overlap = prev_overlap
            for sub in _hard_split(seg, threshold):
                sub_chunk = _build_chunk([sub], sub_overlap)
                sub_overlap = _tail_sentences(sub, overlap_sentences)  # tail of body only
                chunks.append(sub_chunk)
            prev_overlap = sub_overlap  # carry forward into next regular segment
            continue

        if running + seg_words > threshold and current_segs:
            # Flush current accumulation
            chunk_text = _build_chunk(current_segs, prev_overlap)
            prev_overlap = _tail_sentences(chunk_text, overlap_sentences)
            chunks.append(chunk_text)
            current_segs = [seg]
            running = seg_words
        else:
            current_segs.append(seg)
            running += seg_words

    # Flush remainder
    if current_segs:
        remainder_body = "\n\n".join(s.strip() for s in current_segs if s.strip())
        remainder_body_words = _word_count(remainder_body)

        # If the remainder body is tiny (< 20% of threshold) AND merging
        # won't blow the hard ceiling, append body-only to the previous chunk
        if (chunks
                and remainder_body_words < threshold * 0.20
                and running + remainder_body_words <= hard_ceil):
            chunks[-1] = chunks[-1].rstrip() + "\n\n" + remainder_body.lstrip()
        else:
            chunks.append(_build_chunk(current_segs, prev_overlap))

    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_chunk(segments: List[str], overlap: str) -> str:
    """Join segments, optionally prepending overlap context."""
    body = "\n\n".join(s.strip() for s in segments if s.strip())
    if overlap:
        overlap_block = (
            "<!-- overlap-context -->\n"
            + overlap
            + "\n<!-- /overlap-context -->\n\n"
        )
        return overlap_block + body
    return body


def _hard_split(text: str, threshold: int) -> List[str]:
    """
    Last-resort word-boundary split when a single segment exceeds the threshold.
    Tries to break at sentence boundaries within the word window.
    """
    words = text.split()
    parts: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + threshold, len(words))
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)

        # Try to snap back to the nearest sentence boundary
        if end < len(words):
            snap = _last_sentence_boundary(chunk)
            if snap > 0:
                chunk = chunk[:snap].rstrip()

        parts.append(chunk)
        # Advance by the actual words used (sentence-snapped)
        start += len(chunk.split())

    return parts


def _last_sentence_boundary(text: str) -> int:
    """Return the char index of the last sentence boundary in `text`, or 0."""
    for m in reversed(list(re.finditer(r'[.!?]\s', text))):
        return m.end()
    return 0
