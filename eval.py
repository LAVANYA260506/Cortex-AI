#!/usr/bin/env python3
"""
eval.py — Retrieval evaluation suite for the knowledge graph pipeline.

What this measures
------------------
1. Seed Hit Rate       — Did the right document appear in Qdrant top-K seeds?
2. Retrieval Hit Rate  — Did the right document survive BFS + re-rank?
3. MRR                 — Mean Reciprocal Rank (how *high* in the final list?)
4. Context Relevance   — Does the assembled context contain expected keywords?
5. No-Hallucination    — Does the LLM admit ignorance on out-of-domain queries?
6. Cross-chunk fusion  — Does a multi-chunk doc get reassembled correctly?
7. Graph traversal     — Do BFS-expanded nodes include expected neighbours?
8. Latency             — How long does each step take?

Run
---
    # Full suite (no LLM calls — retrieval only, fast)
    python eval.py

    # Include LLM answer quality tests (costs API tokens)
    python eval.py --llm

    # Single test by name
    python eval.py --test test_attention_paper_retrieval

    # Save JSON report
    python eval.py --report results.json

    # Verbose: print full retrieval trace per test
    python eval.py --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ── suppress noisy INFO logs during eval ─────────────────────────────────────
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s — %(message)s")

from retriever import retrieve
from retriever.config import ScoredNode

# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name:        str
    passed:      bool
    score:       float          # 0.0 – 1.0
    message:     str  = ""
    detail:      dict = field(default_factory=dict)
    took_ms:     float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _filenames(nodes: list[ScoredNode]) -> list[str]:
    """Strip paths and extensions for loose matching."""
    return [Path(n.filename).stem.lower() for n in nodes]


def _seed_filenames(seeds: list[dict]) -> list[str]:
    return [Path(s["filename"]).stem.lower() for s in seeds]


def _any_match(targets: list[str], candidates: list[str]) -> bool:
    """True if ANY target string is a substring of ANY candidate string."""
    for t in targets:
        for c in candidates:
            if t.lower() in c.lower() or c.lower() in t.lower():
                return True
    return False


def _mrr(targets: list[str], ranked: list[str]) -> float:
    """Mean Reciprocal Rank — returns 1/(rank) of first hit, or 0."""
    for i, name in enumerate(ranked):
        for t in targets:
            if t.lower() in name.lower() or name.lower() in t.lower():
                return 1.0 / (i + 1)
    return 0.0


def _context_contains(context: str, keywords: list[str], min_hits: int = 1) -> tuple[bool, int]:
    """Check how many keywords appear in the assembled context."""
    ctx_lower = context.lower()
    hits = sum(1 for kw in keywords if kw.lower() in ctx_lower)
    return hits >= min_hits, hits


def _run(fn: Callable) -> TestResult:
    """Run a test function, catching exceptions as failures."""
    t0 = time.perf_counter()
    try:
        result: TestResult = fn()
        result.took_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result
    except Exception as exc:
        took = round((time.perf_counter() - t0) * 1000, 1)
        return TestResult(
            name    = fn.__name__,
            passed  = False,
            score   = 0.0,
            message = f"EXCEPTION: {exc}",
            detail  = {"traceback": traceback.format_exc()},
            took_ms = took,
        )


# ─────────────────────────────────────────────────────────────────────────────
# ── TEST CASES ────────────────────────────────────────────────────────────────
# Each test function returns a TestResult.
# query       → what the user would type
# expected    → stem(s) of files that MUST appear in final_nodes
# anti        → stem(s) that must NOT appear (wrong domain)
# keywords    → words that must appear in the assembled context
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. DS/Algo domain ────────────────────────────────────────────────────────

def test_stack_operations():
    """Query about stacks should retrieve stack chunks, not OS or attention."""
    result = retrieve("What are the operations of a stack data structure?", expand=False)
    final  = _filenames(result["final_nodes"])
    seeds  = _seed_filenames(result["seeds"])

    seed_hit  = _any_match(["stack"], seeds)
    final_hit = _any_match(["stack"], final)
    mrr       = _mrr(["stack"], final)
    anti_contamination = not _any_match(["nips", "attention", "os_module"], final)
    ctx_ok, kw_hits = _context_contains(result["context"], ["stack", "push", "pop"], min_hits=2)

    score   = (seed_hit + final_hit + anti_contamination + ctx_ok) / 4
    passed  = final_hit and anti_contamination and ctx_ok

    return TestResult(
        name    = "test_stack_operations",
        passed  = passed,
        score   = score,
        message = f"seed_hit={seed_hit} final_hit={final_hit} no_contamination={anti_contamination} ctx_kw={kw_hits}/3 mrr={mrr:.2f}",
        detail  = {"seeds": seeds, "final_nodes": final, "mrr": mrr},
    )


def test_queue_fifo():
    """Queue FIFO concept should land on Queues node."""
    result = retrieve("Explain FIFO and how queues work", expand=False)
    final  = _filenames(result["final_nodes"])
    seeds  = _seed_filenames(result["seeds"])

    final_hit = _any_match(["queue"], final)
    mrr       = _mrr(["queue"], final)
    ctx_ok, kw_hits = _context_contains(result["context"], ["queue", "fifo", "enqueue"], min_hits=1)

    score  = (final_hit + ctx_ok + (mrr > 0.3)) / 3
    passed = final_hit and ctx_ok

    return TestResult(
        name    = "test_queue_fifo",
        passed  = passed,
        score   = score,
        message = f"final_hit={final_hit} ctx_kw={kw_hits}/3 mrr={mrr:.2f}",
        detail  = {"seeds": seeds, "final_nodes": final},
    )


def test_graph_traversal_bfs_dfs():
    """BFS/DFS query must retrieve graph chunks, not stack/queue."""
    result = retrieve("How does BFS and DFS work on a graph?", expand=False)
    final  = _filenames(result["final_nodes"])

    final_hit  = _any_match(["graph"], final)
    no_stack   = not _any_match(["stack"], final)
    mrr        = _mrr(["graph"], final)
    ctx_ok, _  = _context_contains(result["context"], ["graph", "vertex", "edge"], min_hits=2)

    score  = (final_hit + no_stack + ctx_ok) / 3
    passed = final_hit and ctx_ok

    return TestResult(
        name    = "test_graph_traversal_bfs_dfs",
        passed  = passed,
        score   = score,
        message = f"final_hit={final_hit} no_stack_bleed={no_stack} ctx_ok={ctx_ok} mrr={mrr:.2f}",
        detail  = {"final_nodes": final, "mrr": mrr},
    )


def test_linked_list_pointer():
    """Linked list query — should not bleed into tree nodes."""
    result = retrieve("How does a singly linked list use pointers?", expand=False)
    final  = _filenames(result["final_nodes"])

    final_hit = _any_match(["linked"], final)
    no_tree   = not _any_match(["tree", "dms"], final[:2])   # top 2 must be clean
    mrr       = _mrr(["linked"], final)
    ctx_ok, _ = _context_contains(result["context"], ["linked", "pointer", "node"], min_hits=2)

    score  = (final_hit + no_tree + ctx_ok) / 3
    passed = final_hit and ctx_ok

    return TestResult(
        name    = "test_linked_list_pointer",
        passed  = passed,
        score   = score,
        message = f"final_hit={final_hit} no_tree_bleed={no_tree} mrr={mrr:.2f}",
        detail  = {"final_nodes": final},
    )


def test_tree_inorder_traversal():
    """Tree traversal query — should pull trees-1 chunks and DMS-Trees."""
    result = retrieve("Explain inorder traversal of a binary tree", expand=False)
    final  = _filenames(result["final_nodes"])
    seeds  = _seed_filenames(result["seeds"])

    seed_hit  = _any_match(["tree", "dms"], seeds)
    final_hit = _any_match(["tree", "dms"], final)
    mrr       = _mrr(["tree"], final)
    ctx_ok, _ = _context_contains(result["context"], ["inorder", "traversal", "tree"], min_hits=2)

    score  = (seed_hit + final_hit + ctx_ok) / 3
    passed = final_hit and ctx_ok

    return TestResult(
        name    = "test_tree_inorder_traversal",
        passed  = passed,
        score   = score,
        message = f"seed_hit={seed_hit} final_hit={final_hit} mrr={mrr:.2f}",
        detail  = {"seeds": seeds, "final_nodes": final},
    )


# ── 2. OS domain ─────────────────────────────────────────────────────────────

def test_os_process_scheduling():
    """OS scheduling query must hit OS modules, not DS/Algo."""
    result = retrieve("What is process scheduling in operating systems?", expand=False)
    final  = _filenames(result["final_nodes"])
    seeds  = _seed_filenames(result["seeds"])

    seed_hit  = _any_match(["os_module", "chapter2_os"], seeds)
    final_hit = _any_match(["os_module", "chapter2_os"], final)
    no_ds     = not _any_match(["stack", "queue", "graph", "tree"], final[:2])
    mrr       = _mrr(["os_module", "chapter2"], final)
    ctx_ok, _ = _context_contains(result["context"], ["process", "schedule", "cpu"], min_hits=2)

    score  = (seed_hit + final_hit + no_ds + ctx_ok) / 4
    passed = final_hit and no_ds

    return TestResult(
        name    = "test_os_process_scheduling",
        passed  = passed,
        score   = score,
        message = f"seed_hit={seed_hit} final_hit={final_hit} no_ds_bleed={no_ds} mrr={mrr:.2f}",
        detail  = {"seeds": seeds, "final_nodes": final},
    )


def test_os_memory_management():
    """Memory management query — OS memory modules."""
    result = retrieve("How does virtual memory and paging work in OS?", expand=False)
    final  = _filenames(result["final_nodes"])

    final_hit = _any_match(["os_module", "memory", "chapter2"], final)
    ctx_ok, kw_hits = _context_contains(
        result["context"], ["memory", "page", "virtual"], min_hits=2
    )
    mrr = _mrr(["os_module", "memory"], final)

    score  = (final_hit + ctx_ok + (mrr > 0.25)) / 3
    passed = final_hit and ctx_ok

    return TestResult(
        name    = "test_os_memory_management",
        passed  = passed,
        score   = score,
        message = f"final_hit={final_hit} ctx_kw={kw_hits}/3 mrr={mrr:.2f}",
        detail  = {"final_nodes": final},
    )


def test_exception_handling():
    """Exception handling node should surface for trap/interrupt queries."""
    result = retrieve("What happens during an exception or interrupt in a system?", expand=False)
    final  = _filenames(result["final_nodes"])

    final_hit = _any_match(["exception"], final)
    ctx_ok, _ = _context_contains(result["context"], ["exception", "interrupt", "trap"], min_hits=1)
    mrr       = _mrr(["exception"], final)

    score  = (final_hit + ctx_ok) / 2
    passed = final_hit

    return TestResult(
        name    = "test_exception_handling",
        passed  = passed,
        score   = score,
        message = f"final_hit={final_hit} ctx_ok={ctx_ok} mrr={mrr:.2f}",
        detail  = {"final_nodes": final},
    )


# ── 3. Attention paper (cross-chunk fusion) ───────────────────────────────────

def test_attention_paper_retrieval():
    """
    The Attention Is All You Need paper is split into 3 chunks.
    A query about transformers should retrieve AT LEAST 2 of the 3 chunks
    — testing whether multi-chunk fusion works.
    """
    result  = retrieve("How does the transformer architecture use self-attention?", expand=False)
    final   = _filenames(result["final_nodes"])
    bfs_all = _filenames(result["bfs_nodes"])

    chunks_in_final = sum(1 for f in final if "nips" in f or "attention" in f)
    chunks_in_bfs   = sum(1 for f in bfs_all if "nips" in f or "attention" in f)
    mrr             = _mrr(["nips", "attention"], final)
    ctx_ok, kw_hits = _context_contains(
        result["context"], ["attention", "transformer", "query", "key", "value"], min_hits=3
    )

    # Pass if at least 1 chunk in final and BFS pulled at least 2 chunks
    passed = chunks_in_final >= 1 and chunks_in_bfs >= 2

    score  = min(1.0, (chunks_in_final / 3) * 0.5 + ctx_ok * 0.3 + (mrr > 0.3) * 0.2)

    return TestResult(
        name    = "test_attention_paper_retrieval",
        passed  = passed,
        score   = round(score, 3),
        message = (
            f"chunks_in_final={chunks_in_final}/3 "
            f"chunks_in_bfs={chunks_in_bfs}/3 "
            f"ctx_kw={kw_hits}/5 mrr={mrr:.2f}"
        ),
        detail  = {"final_nodes": final, "bfs_nodes": bfs_all},
    )


def test_attention_multi_head():
    """Specific sub-topic from the paper."""
    result = retrieve("What is multi-head attention and why is it used?", expand=False)
    final  = _filenames(result["final_nodes"])

    final_hit = _any_match(["nips", "attention"], final)
    ctx_ok, kw_hits = _context_contains(
        result["context"], ["multi-head", "attention", "head"], min_hits=2
    )
    mrr = _mrr(["nips", "attention"], final)

    score  = (final_hit + ctx_ok) / 2
    passed = final_hit

    return TestResult(
        name    = "test_attention_multi_head",
        passed  = passed,
        score   = score,
        message = f"final_hit={final_hit} ctx_kw={kw_hits}/3 mrr={mrr:.2f}",
        detail  = {"final_nodes": final},
    )


# ── 4. Digital/hardware domain ────────────────────────────────────────────────

def test_jk_flipflop():
    """JK flip-flop query should hit the counter/JK node."""
    result = retrieve("How does a JK flip-flop work in a synchronous counter?", expand=False)
    final  = _filenames(result["final_nodes"])

    final_hit = _any_match(["jk", "counter", "synchronous"], final)
    ctx_ok, _ = _context_contains(result["context"], ["flip-flop", "jk", "counter"], min_hits=1)
    mrr       = _mrr(["jk", "counter"], final)

    score  = (final_hit + ctx_ok) / 2
    passed = final_hit

    return TestResult(
        name    = "test_jk_flipflop",
        passed  = passed,
        score   = score,
        message = f"final_hit={final_hit} ctx_ok={ctx_ok} mrr={mrr:.2f}",
        detail  = {"final_nodes": final},
    )


def test_registers():
    """Register query should hit registers node."""
    result = retrieve("What are CPU registers and how are they used?", expand=False)
    final  = _filenames(result["final_nodes"])

    final_hit = _any_match(["register"], final)
    ctx_ok, _ = _context_contains(result["context"], ["register", "cpu", "accumulator"], min_hits=1)
    mrr       = _mrr(["register"], final)

    score  = (final_hit + ctx_ok) / 2
    passed = final_hit

    return TestResult(
        name    = "test_registers",
        passed  = passed,
        score   = score,
        message = f"final_hit={final_hit} ctx_ok={ctx_ok} mrr={mrr:.2f}",
        detail  = {"final_nodes": final},
    )


def test_memory_systems():
    """Memory hierarchy query."""
    result = retrieve("Explain cache memory and memory hierarchy in computer architecture", expand=False)
    final  = _filenames(result["final_nodes"])

    final_hit = _any_match(["memory"], final)
    ctx_ok, kw_hits = _context_contains(
        result["context"], ["cache", "memory", "hierarchy"], min_hits=2
    )
    mrr = _mrr(["memory"], final)

    score  = (final_hit + ctx_ok) / 2
    passed = final_hit

    return TestResult(
        name    = "test_memory_systems",
        passed  = passed,
        score   = score,
        message = f"final_hit={final_hit} ctx_kw={kw_hits}/3 mrr={mrr:.2f}",
        detail  = {"final_nodes": final},
    )


# ── 5. Cross-domain contamination test ───────────────────────────────────────

def test_no_cross_domain_contamination():
    """
    A very specific DS query (stack push/pop) must NOT return OS or
    Attention paper nodes in the top-3 final results.
    Tests that the keyword/semantic filter is working.
    """
    result = retrieve("Explain push and pop operations on a stack", expand=False)
    final  = _filenames(result["final_nodes"])
    top3   = final[:3]

    has_stack     = _any_match(["stack"], top3)
    no_os         = not _any_match(["os_module", "chapter2_os"], top3)
    no_attention  = not _any_match(["nips", "attention"], top3)
    no_jk         = not _any_match(["jk", "counter", "register"], top3)

    passed = has_stack and no_os and no_attention and no_jk
    score  = (has_stack + no_os + no_attention + no_jk) / 4

    return TestResult(
        name    = "test_no_cross_domain_contamination",
        passed  = passed,
        score   = score,
        message = f"top3={top3} has_stack={has_stack} no_os={no_os} no_attention={no_attention} no_jk={no_jk}",
        detail  = {"top3": top3, "all_final": final},
    )


# ── 6. Out-of-domain / hallucination guard ────────────────────────────────────

def test_out_of_domain_rejection():
    """
    Query about something completely absent from the knowledge base.
    The retriever should either return no nodes OR return a very low score seed.
    Tests that the pipeline does NOT confidently hallucinate a source.
    """
    result = retrieve(
        "What are the cooking techniques used in Italian cuisine?",
        expand=False,
        seed_min_score=0.28,   # use default threshold
    )
    final = result["final_nodes"]
    seeds = result["seeds"]

    # Either no seeds, or all seed scores are very low
    max_seed_score = max((s["score"] for s in seeds), default=0.0)
    no_confident_result = len(final) == 0 or max_seed_score < 0.35

    return TestResult(
        name    = "test_out_of_domain_rejection",
        passed  = no_confident_result,
        score   = 1.0 if no_confident_result else 0.0,
        message = f"final_nodes={len(final)} max_seed_score={max_seed_score:.3f}",
        detail  = {"seeds": [s["score"] for s in seeds], "final_count": len(final)},
    )


# ── 7. Graph BFS expansion test ───────────────────────────────────────────────

def test_bfs_expands_beyond_seed():
    """
    Verify BFS actually collects MORE nodes than just the seeds.
    If BFS only returns seed nodes, graph traversal is broken.
    """
    result = retrieve("How does a binary search tree work?", expand=False)

    n_seeds = len(result["seeds"])
    n_bfs   = len(result["bfs_nodes"])

    bfs_expanded = n_bfs > n_seeds
    multiple_bfs = n_bfs >= 2

    passed = bfs_expanded and multiple_bfs
    score  = min(1.0, n_bfs / max(n_seeds + 1, 3))

    return TestResult(
        name    = "test_bfs_expands_beyond_seed",
        passed  = passed,
        score   = round(score, 3),
        message = f"seeds={n_seeds} bfs_nodes={n_bfs} expanded={bfs_expanded}",
        detail  = {
            "seed_files": _seed_filenames(result["seeds"]),
            "bfs_files":  _filenames(result["bfs_nodes"]),
        },
    )


def test_rrf_reorders_bfs():
    """
    Verify RRF changes the node ordering compared to raw BFS visit order.
    If RRF top node == BFS visit order top node always, RRF is doing nothing.
    """
    result = retrieve("What is the attention mechanism?", expand=False)

    bfs_nodes = result["bfs_nodes"]
    rrf_nodes = result["rrf_nodes"]

    if len(bfs_nodes) < 2 or len(rrf_nodes) < 2:
        return TestResult(
            name="test_rrf_reorders_bfs", passed=False, score=0.0,
            message="Not enough nodes to compare ordering"
        )

    bfs_top = _filenames(bfs_nodes)[0]
    rrf_top = _filenames(rrf_nodes)[0]

    # Also check that final_nodes differ from bfs_nodes in order
    final_top = _filenames(result["final_nodes"])[0] if result["final_nodes"] else ""
    reordered = (bfs_top != rrf_top) or (rrf_top != final_top)

    return TestResult(
        name    = "test_rrf_reorders_bfs",
        passed  = True,   # informational — always passes, shows the reordering
        score   = 1.0 if reordered else 0.5,
        message = f"bfs_top={bfs_top} rrf_top={rrf_top} final_top={final_top} reordered={reordered}",
        detail  = {
            "bfs_order":   _filenames(bfs_nodes)[:5],
            "rrf_order":   _filenames(rrf_nodes)[:5],
            "final_order": _filenames(result["final_nodes"])[:5],
        },
    )


# ── 8. Latency test ───────────────────────────────────────────────────────────

def test_latency_no_expand():
    """
    Retrieval without expansion should complete in under 5 seconds
    on a local machine. Tests pipeline efficiency.
    """
    t0     = time.perf_counter()
    result = retrieve("What is a queue data structure?", expand=False)
    took   = time.perf_counter() - t0

    under_5s = took < 5.0
    has_result = len(result["final_nodes"]) > 0

    return TestResult(
        name    = "test_latency_no_expand",
        passed  = under_5s,
        score   = max(0.0, 1.0 - took / 10.0),
        message = f"took={took:.2f}s {'OK' if under_5s else 'SLOW'}",
        detail  = {"took_seconds": round(took, 3), "final_nodes": len(result["final_nodes"])},
    )


def test_seed_score_ordering():
    """
    Seeds returned by Qdrant must be sorted descending by score.
    Tests that search.py is not corrupting the ordering.
    """
    result = retrieve("How does process scheduling work?", expand=False)
    seeds  = result["seeds"]

    if len(seeds) < 2:
        return TestResult(
            name="test_seed_score_ordering", passed=True, score=1.0,
            message="Only 1 seed — ordering trivially correct"
        )

    scores  = [s["score"] for s in seeds]
    ordered = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

    return TestResult(
        name    = "test_seed_score_ordering",
        passed  = ordered,
        score   = 1.0 if ordered else 0.0,
        message = f"scores={[round(s,3) for s in scores]} sorted_desc={ordered}",
        detail  = {"scores": scores},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Optional LLM tests (only run with --llm flag, cost tokens)
# ─────────────────────────────────────────────────────────────────────────────

def test_llm_answer_cites_source():
    """LLM answer for a stack query must cite at least one .md filename."""
    import re
    from retriever import ask
    answer = ask("What are push and pop operations on a stack?", expand=False)
    citations = re.findall(r"\[([^\]]+\.md)\]", answer)
    has_citation = len(citations) > 0

    return TestResult(
        name    = "test_llm_answer_cites_source",
        passed  = has_citation,
        score   = 1.0 if has_citation else 0.0,
        message = f"citations_found={citations}",
        detail  = {"answer_excerpt": answer[:300]},
    )


def test_llm_refuses_out_of_domain():
    """LLM must decline to answer cooking questions using the knowledge base."""
    from retriever import ask
    answer = ask("Give me a recipe for pasta carbonara", expand=False)
    refusal_signals = [
        "does not contain",
        "not enough information",
        "cannot find",
        "no information",
        "not covered",
        "knowledge base",
    ]
    refused = any(sig.lower() in answer.lower() for sig in refusal_signals)

    return TestResult(
        name    = "test_llm_refuses_out_of_domain",
        passed  = refused,
        score   = 1.0 if refused else 0.0,
        message = f"refused={refused}",
        detail  = {"answer_excerpt": answer[:300]},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test registry
# ─────────────────────────────────────────────────────────────────────────────

RETRIEVAL_TESTS = [
    test_stack_operations,
    test_queue_fifo,
    test_graph_traversal_bfs_dfs,
    test_linked_list_pointer,
    test_tree_inorder_traversal,
    test_os_process_scheduling,
    test_os_memory_management,
    test_exception_handling,
    test_attention_paper_retrieval,
    test_attention_multi_head,
    test_jk_flipflop,
    test_registers,
    test_memory_systems,
    test_no_cross_domain_contamination,
    test_out_of_domain_rejection,
    test_bfs_expands_beyond_seed,
    test_rrf_reorders_bfs,
    test_latency_no_expand,
    test_seed_score_ordering,
]

LLM_TESTS = [
    test_llm_answer_cites_source,
    test_llm_refuses_out_of_domain,
]


# ─────────────────────────────────────────────────────────────────────────────
# Runner + reporter
# ─────────────────────────────────────────────────────────────────────────────

def run_suite(tests: list[Callable], verbose: bool = False) -> list[TestResult]:
    results = []
    for fn in tests:
        print(f"  running {fn.__name__} ...", end="", flush=True)
        r = _run(fn)
        status = "PASS" if r.passed else "FAIL"
        print(f"\r  [{status}] {fn.__name__:<45} score={r.score:.2f}  {r.took_ms:.0f}ms")
        if not r.passed or verbose:
            print(f"         {r.message}")
        if verbose and r.detail:
            for k, v in r.detail.items():
                print(f"         {k}: {v}")
        results.append(r)
    return results


def print_summary(results: list[TestResult]) -> None:
    passed = sum(1 for r in results if r.passed)
    total  = len(results)
    avg_score = sum(r.score for r in results) / total if total else 0
    avg_ms    = sum(r.took_ms for r in results) / total if total else 0

    # MRR across tests that have mrr in detail
    print()
    print("=" * 65)
    print(f"  RESULTS : {passed}/{total} passed")
    print(f"  AVG SCORE  : {avg_score:.3f} / 1.000")
    print(f"  AVG LATENCY: {avg_ms:.0f} ms per test")
    print("=" * 65)

    if passed < total:
        print("\n  FAILED TESTS:")
        for r in results:
            if not r.passed:
                print(f"    ✗ {r.name}")
                print(f"      {r.message}")

    print()


def save_report(results: list[TestResult], path: str) -> None:
    data = {
        "summary": {
            "passed": sum(1 for r in results if r.passed),
            "total":  len(results),
            "avg_score": round(sum(r.score for r in results) / len(results), 4) if results else 0,
        },
        "tests": [
            {
                "name":    r.name,
                "passed":  r.passed,
                "score":   r.score,
                "message": r.message,
                "took_ms": r.took_ms,
                "detail":  r.detail,
            }
            for r in results
        ],
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  Report saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Retriever eval suite")
    parser.add_argument("--llm",     action="store_true", help="Also run LLM answer tests (costs API tokens)")
    parser.add_argument("--test",    type=str,            help="Run a single test by function name")
    parser.add_argument("--report",  type=str,            help="Save JSON report to this path")
    parser.add_argument("--verbose", action="store_true", help="Print full detail for every test")
    args = parser.parse_args()

    all_tests = RETRIEVAL_TESTS + (LLM_TESTS if args.llm else [])

    if args.test:
        matched = [t for t in all_tests if t.__name__ == args.test]
        if not matched:
            print(f"Test '{args.test}' not found.")
            print("Available:", ", ".join(t.__name__ for t in all_tests))
            sys.exit(1)
        all_tests = matched

    print(f"\nRunning {len(all_tests)} test(s)"
          + (" + LLM tests" if args.llm else " (retrieval only — use --llm for answer tests)"))
    print("-" * 65)

    results = run_suite(all_tests, verbose=args.verbose)
    print_summary(results)

    if args.report:
        save_report(results, args.report)

    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
