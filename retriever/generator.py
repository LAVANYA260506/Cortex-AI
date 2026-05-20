"""
retriever/generator.py
──────────────────────
Step 7 of the retrieval pipeline: LLM answer generation.

Uses Groq to generate a grounded, cited answer from the assembled context.

Setup
-----
Set the environment variable GROQ_API_KEY to your Groq API key.
Free tier: https://console.groq.com/
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a precise, citation-aware knowledge-base assistant.

Rules:
1. Answer using ONLY the provided context documents.
2. After each factual claim, cite the source filename in square brackets,
   e.g. [report_chunk_01.md].  If multiple documents support a claim, cite all.
3. If the context is insufficient to answer, respond with exactly:
   "The knowledge base does not contain enough information to answer this question."
4. Do not invent facts, infer beyond the text, or use outside knowledge.
5. Lead with a direct answer, then supporting detail.  Be concise."""


def generate_answer(
    query:          str,
    context:        str,
    query_variants: list[str] | None = None,
) -> str:
    """
    Call Groq to produce a grounded, cited answer.

    Parameters
    ----------
    query : str
        The original user question.
    context : str
        Assembled Markdown context from context.build_context().
    query_variants : list[str] | None
        Query + sub-questions from query_expander.  When provided and longer
        than 1 item, they are appended to the user turn so the model sees
        the full information need.

    Returns
    -------
    str — the model's answer text.

    Raises
    ------
    RuntimeError  if the groq SDK is not installed or GROQ_API_KEY is unset.
    """
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError(
            "groq SDK not installed. Run: pip install groq"
        ) from exc

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable not set. "
            "Get a free key at https://console.groq.com/"
        )

    client = Groq(api_key=api_key)

    # Build optional variants block
    variants_block = ""
    if query_variants and len(query_variants) > 1:
        variants_block = (
            "\n\n**Query variants considered during retrieval:**\n"
            + "\n".join(f"- {v}" for v in query_variants)
        )

    user_message = (
        f"## Context documents\n\n{context}"
        f"{variants_block}"
        f"\n\n---\n\n## Question\n\n{query}"
    )

    log.info("Generating answer with Groq...")
    response = client.chat.completions.create(
        model      = "llama-3.3-70b-versatile",  # fast & free on Groq
        max_tokens = 1024,
        messages   = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    answer = response.choices[0].message.content
    log.info("Answer generated (%d chars).", len(answer))
    return answer