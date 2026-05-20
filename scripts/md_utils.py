#!/usr/bin/env python3
"""
md_utils.py — Markdown file helpers for the knowledge graph pipeline.

Used by Gemini CLI to read raw/*.md files, inject wikilinks, and
parse YAML front matter.

Usage:
    python scripts/md_utils.py read-frontmatter raw/report.md
    python scripts/md_utils.py inject-links raw/report.md "[[other-file]]" "[[yet-another]]"
    python scripts/md_utils.py list-files [--since <iso_timestamp>]
    python scripts/md_utils.py read raw/report.md
"""

from __future__ import annotations

import re
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

RAW_DIR = Path("raw")
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)
WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


# ---------------------------------------------------------------------------
# Front-matter parsing
# ---------------------------------------------------------------------------

def parse_frontmatter(text: str) -> dict:
    """Parse YAML-like front matter into a dict (simple key: value only)."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}
    result = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip().strip("'\"")
    return result


def strip_frontmatter(text: str) -> str:
    return FRONTMATTER_RE.sub("", text, count=1)


# ---------------------------------------------------------------------------
# Wikilink injection
# ---------------------------------------------------------------------------

def existing_wikilinks(text: str) -> set[str]:
    return set(WIKILINK_RE.findall(text))


def inject_related_section(filepath: Path, targets: list[str]) -> int:
    """
    Append or update a '## Related' section in filepath with wikilinks.
    Only adds links that are not already present anywhere in the file.
    Returns the number of new links added.
    """
    content = filepath.read_text(encoding="utf-8")
    existing = existing_wikilinks(content)
    new_targets = [t for t in targets if t not in existing]
    if not new_targets:
        return 0

    link_lines = "\n".join(f"- [[{t}]]" for t in new_targets)

    if "## Related" in content:
        # Append to existing Related section
        content = content.rstrip() + "\n" + link_lines + "\n"
    else:
        content = content.rstrip() + f"\n\n## Related\n\n{link_lines}\n"

    filepath.write_text(content, encoding="utf-8")
    return len(new_targets)


def inject_inline_links(filepath: Path, replacements: dict[str, str]) -> int:
    """
    Replace occurrences of plain text phrases with wikilinks inline.
    replacements = {"Climate change": "climate-change-report"}
    Returns count of replacements made.
    """
    content = filepath.read_text(encoding="utf-8")
    body = strip_frontmatter(content)
    frontmatter = content[: len(content) - len(body)]

    count = 0
    for phrase, target in replacements.items():
        # Only replace first occurrence to avoid spam
        new_body, n = re.subn(
            rf"\b{re.escape(phrase)}\b(?!\]\])",
            f"[[{target}|{phrase}]]",
            body,
            count=1,
            flags=re.IGNORECASE,
        )
        if n:
            body = new_body
            count += n

    filepath.write_text(frontmatter + body, encoding="utf-8")
    return count


# ---------------------------------------------------------------------------
# File listing
# ---------------------------------------------------------------------------

def list_md_files(since: str | None = None) -> list[dict]:
    """
    List all .md files in raw/.
    If `since` is an ISO timestamp string, only return files modified after it.
    """
    result = []
    for path in sorted(RAW_DIR.glob("*.md")):
        mtime = path.stat().st_mtime
        mtime_iso = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if since and mtime_iso <= since:
            continue
        fm = parse_frontmatter(path.read_text(encoding="utf-8"))
        result.append({
            "filename": str(path),
            "stem": path.stem,
            "mtime": mtime_iso,
            "original_filename": fm.get("original_filename", ""),
            "chunk": int(fm.get("chunk", 0)),
            "total_chunks": int(fm.get("total_chunks", 1)),
            "processed_date": fm.get("processed_date", ""),
        })
    return result


def read_file(filepath: Path | str) -> str:
    p = Path(filepath)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> None:
    if not argv:
        print(__doc__)
        return

    cmd, *args = argv

    if cmd == "read-frontmatter":
        text = read_file(args[0])
        print(json.dumps(parse_frontmatter(text), indent=2))

    elif cmd == "inject-links":
        filepath = Path(args[0])
        targets = args[1:]
        added = inject_related_section(filepath, targets)
        print(f"Added {added} new link(s) to {filepath.name}")

    elif cmd == "list-files":
        since = None
        if "--since" in args:
            since = args[args.index("--since") + 1]
        print(json.dumps(list_md_files(since), indent=2))

    elif cmd == "read":
        print(read_file(args[0]))

    elif cmd == "existing-links":
        text = read_file(args[0])
        print(json.dumps(sorted(existing_wikilinks(text)), indent=2))

    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
