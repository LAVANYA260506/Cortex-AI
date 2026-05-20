"""
retriever/__main__.py
─────────────────────
CLI entry point.  Invoked as:

    python -m retriever ask "What were Q2 revenue highlights?"
    python -m retriever ask "NLP tokenisation" --no-llm --verbose
    python -m retriever ask "climate findings" --top-k 6 --max-nodes 15 --no-expand

Why __main__.py instead of a script?
--------------------------------------
Putting the CLI here means:
  - `python -m retriever ask "..."` works without any PATH manipulation.
  - The package is still importable cleanly (`from retriever import ask`).
  - No sys.path hacks needed: Python adds the parent of the package to
    sys.path automatically when running with -m.

Nothing in this file does computation — it only parses args, calls the
public API from __init__.py, and formats output.  All logic lives in the
pipeline modules so this file stays thin and testable.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import ask, retrieve
from .config import MAX_BFS_NODES, SEED_MIN_SCORE, TOP_K_SEEDS


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_report(result: dict) -> None:
    W = 68
    print("\n" + "=" * W)
    print(f"  QUERY   : {result['query']}")
    variants = result.get("variants", [])
    if len(variants) > 1:
        print(f"  EXPANDED: {len(variants) - 1} sub-question(s) generated")
        for v in variants[1:]:
            print(f"            • {v}")
    print("=" * W)

    # Step 2 — seeds
    seeds = result.get("seeds", [])
    print(f"\n[Step 2] Multi-seed search — {len(seeds)} seed(s) above threshold:")
    for s in seeds:
        print(f"  {s['score']:.3f}  rank={s['qdrant_rank']}  {Path(s['filename']).name}")

    if not seeds:
        print("\n  No seeds found — query may be too vague or knowledge base empty.")
        print("=" * W + "\n")
        return

    # Step 3 — BFS
    bfs = result.get("bfs_nodes", [])
    print(f"\n[Step 3] Semantic BFS — {len(bfs)} node(s) visited:")
    shown = sorted(bfs, key=lambda n: n.bfs_order)[:10]
    for n in shown:
        print(f"  ord={n.bfs_order:<3} depth={n.hop_depth}  {Path(n.filename).name}")
    if len(bfs) > 10:
        print(f"  … and {len(bfs) - 10} more")

    # Step 4 — RRF
    rrf = result.get("rrf_nodes", [])
    print(f"\n[Step 4] After RRF fusion — top 5 of {len(rrf)}:")
    for n in rrf[:5]:
        print(f"  rrf={n.rrf_score:.5f}  {Path(n.filename).name}")

    # Step 5 — re-rank
    final = result.get("final_nodes", [])
    print(f"\n[Step 5] After semantic re-rank — {len(final)} node(s) to LLM:")
    for n in final:
        print(
            f"  final={n.rrf_score:.4f}  "
            f"sem={n.semantic_score:.4f}  "
            f"depth={n.hop_depth}  "
            f"{Path(n.filename).name}"
        )

    # Step 6 — context
    ctx_words = len(result.get("context", "").split())
    print(f"\n[Step 6] Context assembled: ~{ctx_words} words")
    print("=" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "python -m retriever",
        description = "Graph-aware semantic retrieval engine",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = (
            "Examples:\n"
            "  python -m retriever ask \"Q2 revenue highlights\"\n"
            "  python -m retriever ask \"NLP tokenisation\" --no-llm --verbose\n"
            "  python -m retriever ask \"climate\" --top-k 6 --max-nodes 15\n"
        ),
    )
    sub = parser.add_subparsers(dest="cmd")

    ap = sub.add_parser("ask", help="Ask a question against the knowledge base")
    ap.add_argument(
        "query", nargs="+",
        help="Natural language question (quote multi-word queries)",
    )
    ap.add_argument(
        "--top-k", type=int, default=TOP_K_SEEDS, metavar="N",
        help=f"Max Qdrant seed candidates (default {TOP_K_SEEDS})",
    )
    ap.add_argument(
        "--max-nodes", type=int, default=MAX_BFS_NODES, metavar="N",
        help=f"BFS node cap (default {MAX_BFS_NODES})",
    )
    ap.add_argument(
        "--min-score", type=float, default=SEED_MIN_SCORE, metavar="F",
        help=f"Seed cosine floor (default {SEED_MIN_SCORE})",
    )
    ap.add_argument(
        "--no-expand", action="store_true",
        help="Skip query expansion (faster, lower recall)",
    )
    ap.add_argument(
        "--no-llm", action="store_true",
        help="Print retrieval trace only; skip LLM answer generation",
    )
    ap.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG logging",
    )
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    if getattr(args, "verbose", False):
        logging.getLogger().setLevel(logging.DEBUG)

    if args.cmd != "ask":
        parser.print_help()
        sys.exit(0)

    query = " ".join(args.query)

    result = retrieve(
        query,
        top_k          = args.top_k,
        max_nodes       = args.max_nodes,
        seed_min_score  = args.min_score,
        expand          = not args.no_expand,
    )
    _print_report(result)

    if args.no_llm:
        return

    if not result["final_nodes"]:
        print(
            "No relevant nodes found after retrieval.\n"
            "Try lowering --min-score or ingesting more documents."
        )
        return

    print("ANSWER\n" + "-" * 68)
    from .generator import generate_answer
    answer = generate_answer(query, result["context"], result["variants"])
    print(answer)
    print()


if __name__ == "__main__":
    main()
