#!/usr/bin/env bash
# setup.sh — one-shot setup for the knowledge graph pipeline
# Run once: bash setup.sh

set -euo pipefail

echo "=== Knowledge Graph Pipeline Setup ==="

# Create required directories
mkdir -p inputs raw processed qdrant_db scripts

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install --quiet \
    watchdog \
    sentence-transformers \
    qdrant-client \
    python-pptx \
    pdfplumber \
    docling 2>/dev/null || true

echo "Core dependencies installed."

# Initialise the SQLite database
echo ""
echo "Initialising kg.db..."
python scripts/db_utils.py init

echo ""
echo "=== Setup complete. ==="
echo ""
echo "Start the three terminals:"
echo ""
echo "  Terminal 1 — Document watcher:"
echo "    python watcher.py"
echo ""
echo "  Terminal 2 — Gemini CLI:"
echo "    gemini --yolo "
echo "    (then type commands: update, lint)"
echo ""
echo "  Terminal 3 — Embedding daemon:"
echo "    python embedder.py run"
echo ""
echo "  Optional — Similarity search:"
echo "    python embedder.py search \"your query here\""
echo ""
echo "  Optional — Check queue:"
echo "    python embedder.py queue-status"
echo "    python scripts/job_queue.py peek"
