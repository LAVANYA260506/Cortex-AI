# Cortex-Wiki

> A local-first, graph-augmented knowledge base that turns your documents into a queryable, semantically linked wiki — maintained autonomously by an LLM agent and served through a FastAPI retrieval engine.

Inspired by [Andrej Karpathy's LLM Wiki concept](https://x.com/karpathy) — the idea that an LLM should be able to maintain, link, and reason over a personal knowledge graph the way a wiki editor would, but at machine speed and scale.

---

## What it does

Drop a PDF or PPTX into a folder. Cortex-Wiki automatically converts it to Markdown, chunks it, summarises every chunk, links it to related nodes in a knowledge graph, embeds it into a vector store, and makes it queryable via a REST API — all without any manual intervention.

The graph is visualised locally in **Obsidian**, giving you a living, navigable map of everything in your knowledge base.

---

## Architecture — three independent systems

Cortex-Wiki is built as three decoupled processes that communicate only through shared file artifacts. No process calls another directly.

```
┌─────────────────────┐     raw/*.md      ┌──────────────────────────┐
│   Terminal 1        │ ───────────────►  │   Terminal 2             │
│   Document          │                   │   Gemini CLI Agent       │
│   Ingestion         │                   │   (LLM Wiki Maintainer)  │
│                     │ ◄─────────────── │                          │
│  watcher.py         │   wikilinks       │  db_utils.py             │
│  convert_docs.py    │   injected back   │  md_utils.py             │
│  chunker.py         │                   │  job_queue.py            │
└─────────────────────┘                   └──────────┬───────────────┘
                                                     │
                                          kg.db + embed_queue.jsonl
                                                     │
                                                     ▼
                                          ┌──────────────────────────┐
                                          │   Terminal 3             │
                                          │   Embedder Daemon        │
                                          │                          │
                                          │  all-MiniLM-L6-v2        │
                                          │  Qdrant (local, on-disk) │
                                          └──────────┬───────────────┘
                                                     │
                                               qdrant_db/
                                                     │
                                                     ▼
                                          ┌──────────────────────────┐
                                          │   FastAPI                │
                                          │   Retriever Service      │
                                          │                          │
                                          │  Query expand (Haiku)    │
                                          │  Multi-seed search       │
                                          │  Semantic BFS            │
                                          │  RRF + re-rank           │
                                          │  Groq LLaMA answer gen   │
                                          └──────────────────────────┘
```

### Terminal 1 — Document Ingestion

Watches `inputs/` for new `.pdf`, `.pptx`, or `.ppt` files. On detection, converts them to clean Markdown using Docling (PDF) or python-pptx (PPTX), then runs `smart_chunk()` — a content-aware splitter that respects page markers, H2/H3 headings, and paragraph boundaries before falling back to a hard word-count split. Each chunk is written as its own `.md` file under `raw/` with YAML front matter.

### Terminal 2 — Gemini CLI Agent (LLM Wiki Maintainer)

This is the core idea lifted from Karpathy's LLM Wiki method: an LLM agent that autonomously maintains a knowledge graph the way a diligent wiki editor would — summarising nodes, finding thematic connections, and injecting `[[wikilinks]]` — but doing it at machine speed over hundreds of documents.

The agent is driven by two commands:

- **`update`** — discovers new chunks in `raw/`, upserts them into `kg.db`, generates a 3–5 sentence summary per node, compares summaries in-memory to find the top 2–4 thematically related existing nodes, injects bidirectional wikilinks into the `.md` files, and writes embed jobs to `embed_queue.jsonl`. All of this happens in a single autonomous run with no confirmation gates.
- **`lint`** — audits the graph for broken links, orphan nodes, missing summaries, and stale metadata, then auto-fixes every issue it finds.

The agent never reads existing files to find links — it fetches all summaries from SQLite in one query and does all thematic comparison in-memory. This keeps latency constant regardless of graph size.

### Terminal 3 — Embedder Daemon

Polls `embed_queue.jsonl` every 10 seconds. When jobs are present, it atomically drains the queue using `os.rename()` (no locking primitives needed), encodes summaries with `all-MiniLM-L6-v2`, and upserts 384-dimensional vectors into a local Qdrant collection. If the process crashes mid-batch, the `.processing` file is recovered on restart — no job is ever lost.

---

## Retrieval pipeline — 7 steps

When a query hits the FastAPI endpoint, it runs through a multi-stage pipeline:

| Step | Module | What happens |
|------|--------|-------------|
| 1 | `query_expander.py` | Claude Haiku generates 3 sub-questions; all variants are embedded and max-pooled into one vector |
| 2 | `search.py` | Multi-seed Qdrant search — all hits above cosine threshold become BFS roots |
| 3 | `graph.py` | Semantic priority-queue BFS through `kg.db` links; neighbours scored by `cosine × decay^depth` |
| 4 | `fusion.py` | Reciprocal Rank Fusion across Qdrant rank, BFS order, and hop depth |
| 5 | `fusion.py` | Semantic re-rank: `0.6 × cosine + 0.4 × normalised_RRF` |
| 6 | `context.py` | Top nodes assembled into a Markdown context block, stripped of front matter and overlap noise |
| 7 | `generator.py` | Groq LLaMA 3.3-70B generates a cited answer grounded strictly in the retrieved context |

---

## Obsidian graph view

All `raw/*.md` files are valid Obsidian vault documents. The `[[wikilinks]]` injected by the Gemini CLI agent render as edges in Obsidian's graph view, giving you a live, navigable map of the knowledge base. The system is **local-first** — every file, database, and vector store lives on disk. No cloud sync, no external database.

Open the `raw/` folder as an Obsidian vault to explore the graph.

---

## Evaluation results

The retrieval pipeline ships with a suite of 19 behavioural tests covering graph traversal, cross-domain isolation, ranking correctness, and latency.

```
RESULTS : 19/19 passed
AVG SCORE : 0.947 / 1.000
AVG LATENCY : 2905 ms per test
```

| Test | Score | Latency |
|------|-------|---------|
| test_stack_operations | 1.00 | 29337ms |
| test_queue_fifo | 1.00 | 2052ms |
| test_graph_traversal_bfs_dfs | 0.67 | 939ms |
| test_linked_list_pointer | 1.00 | 1606ms |
| test_tree_inorder_traversal | 1.00 | 1422ms |
| test_os_process_scheduling | 1.00 | 1611ms |
| test_os_memory_management | 1.00 | 1736ms |
| test_exception_handling | 1.00 | 1707ms |
| test_attention_paper_retrieval | 1.00 | 1348ms |
| test_attention_multi_head | 1.00 | 1032ms |
| test_jk_flipflop | 1.00 | 1486ms |
| test_registers | 1.00 | 1886ms |
| test_memory_systems | 1.00 | 1584ms |
| test_no_cross_domain_contamination | 1.00 | 1849ms |
| test_out_of_domain_rejection | 1.00 | 16ms |
| test_bfs_expands_beyond_seed | 1.00 | 1142ms |
| test_rrf_reorders_bfs | 0.50 | 1127ms |
| test_latency_no_expand | 0.83 | 1732ms |
| test_seed_score_ordering | 1.00 | 1578ms |

The only sub-1.0 scores are on `test_graph_traversal_bfs_dfs` (0.67) and `test_rrf_reorders_bfs` (0.50) — both ranking-sensitivity tests where the pipeline retrieves the right documents but in a different order than the test expects. The `test_out_of_domain_rejection` completes in 16ms because the retriever correctly aborts before the LLM call when no seeds clear the cosine threshold.

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Document parsing | Docling, python-pptx, pdfplumber |
| Chunking | Custom content-aware splitter |
| Knowledge graph | SQLite (`kg.db`) |
| LLM wiki agent | Gemini CLI (autonomous, tool-use) |
| Query expansion | Claude Haiku (`claude-haiku-4-5`) |
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector store | Qdrant (local, on-disk) |
| Answer generation | Groq LLaMA 3.3-70B |
| Graph visualisation | Obsidian (local vault) |
| API | FastAPI |

---

## Setup

### Prerequisites

- Python 3.10+
- [Obsidian](https://obsidian.md/) (for graph view)
- [Gemini CLI](https://github.com/google-gemini/gemini-cli) installed and authenticated
- Groq API key — free tier at [console.groq.com](https://console.groq.com)
- Anthropic API key — for query expansion via Claude Haiku

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment

```bash
export GROQ_API_KEY=your_groq_key
export ANTHROPIC_API_KEY=your_anthropic_key
```

### Initialise the database

```bash
python scripts/db_utils.py init
```

### Terminal 1 — start the ingestion watcher

```bash
python watcher.py
```

Drop `.pdf`, `.pptx`, or `.ppt` files into `inputs/`. They will be converted, chunked, and written to `raw/` automatically.

### Terminal 2 — run the Gemini CLI agent

```bash
gemini
```

Then inside the Gemini CLI session:

```
update        # index new files, generate summaries, link nodes
lint          # audit and auto-fix the graph
```

### Terminal 3 — start the embedder daemon

```bash
python embedder.py run
```

### FastAPI — start the retriever service

```bash
uvicorn api:app --reload
```

### Obsidian graph view

Open the `raw/` directory as an Obsidian vault. The `[[wikilinks]]` injected by the agent will render as a navigable knowledge graph.

---

## Inspiration

This project is directly inspired by **Andrej Karpathy's LLM Wiki** idea — the observation that a sufficiently capable language model should be able to act as a tireless wiki editor: reading documents, writing summaries, finding connections between ideas, and maintaining a living knowledge graph that grows smarter as more content is added. Cortex-Wiki is an attempt to build that system locally, with full control over the data and the graph.

---

## Project structure

```
cortex-wiki/
├── inputs/                  # Drop documents here
├── raw/                     # Chunked .md files (also your Obsidian vault)
├── scripts/
│   ├── db_utils.py          # SQLite knowledge graph operations
│   ├── md_utils.py          # Wikilink injection, file listing
│   └── job_queue.py         # Append-only embed job queue
├── retriever/
│   ├── __init__.py          # Pipeline orchestration
│   ├── config.py            # All tunable constants
│   ├── query_expander.py    # Step 1: Haiku query expansion
│   ├── search.py            # Step 2: Multi-seed Qdrant search
│   ├── graph.py             # Step 3: Semantic BFS
│   ├── fusion.py            # Steps 4-5: RRF + re-rank
│   ├── context.py           # Step 6: Context assembly
│   └── generator.py         # Step 7: Groq answer generation
├── watcher.py               # Terminal 1: FS event handler
├── convert_docs_to_markdown.py
├── chunker.py
├── embedder.py              # Terminal 3: Embedding daemon
├── gemini.md                # Gemini CLI agent instructions
├── kg.db                    # SQLite knowledge graph
├── qdrant_db/               # Local Qdrant vector store
└── eval.py                  # 19-test retrieval evaluation suite
```

---

## License

MIT

