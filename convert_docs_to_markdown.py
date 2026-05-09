#!/usr/bin/env python3
"""Convert local PDF, PPTX, and PPT files into chunked Markdown for knowledge graph ingestion."""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from chunker import smart_chunk

try:
    from docling.document_converter import DocumentConverter
except ImportError:
    DocumentConverter = None  # type: ignore

try:
    from pptx import Presentation
except ImportError:
    Presentation = None  # type: ignore

try:
    import comtypes.client
except ImportError:
    comtypes = None  # type: ignore

INPUT_DIR = Path("inputs")
OUTPUT_DIR = Path("raw")
SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".ppt"}
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def ensure_directories() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# YAML front-matter
# ---------------------------------------------------------------------------

def build_yaml_header(source_path: Path, chunk_index: int = 0, total_chunks: int = 1) -> str:
    lines = [
        "---",
        f"original_filename: '{source_path.name}'",
        f"processed_date: '{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}'",
    ]
    if total_chunks > 1:
        lines.append(f"chunk: {chunk_index + 1}")
        lines.append(f"total_chunks: {total_chunks}")
    lines.append("---\n")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Markdown cleaning
# ---------------------------------------------------------------------------

def clean_markdown(text: str) -> str:
    lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        # Normalise bullet characters
        if re.match(r"^[\u2022\-*+]\s+", line):
            lines.append("- " + line.lstrip("\u2022-*+ ").strip())
            continue
        # Normalise numbered lists to bullets
        if re.match(r"^\d+[.)]\s+", line):
            lines.append("- " + re.sub(r"^\d+[.)]\s+", "", line).strip())
            continue
        # Crude heading detection for "=== Title ===" style
        if line.startswith("=") and len(line) > 2:
            lines.append(f"## {line.strip('= ').strip()}")
            continue
        lines.append(line)

    result = "\n".join(lines).strip()
    # Collapse 3+ consecutive blank lines to 2
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result + "\n"


# ---------------------------------------------------------------------------
# PPTX extraction
# ---------------------------------------------------------------------------

def _safe_shape_text(shape) -> str:
    if not hasattr(shape, "text_frame"):
        return ""
    try:
        return shape.text_frame.text.strip()
    except Exception:
        return ""


def slide_to_markdown(slide, index: int) -> str:
    """
    Convert one slide to markdown, emitting a <!-- page N --> marker
    so that smart_chunk can split on slide boundaries for PPTX files.
    """
    # Use the proper title placeholder if present
    title_shape = getattr(slide.shapes, "title", None)
    slide_title = title_shape.text.strip() if (title_shape and title_shape.text) else None

    text_lines: List[str] = []
    skipped_shapes = 0

    for shape in slide.shapes:
        if shape.has_text_frame:
            text = _safe_shape_text(shape)
            if text:
                text_lines.extend(text.splitlines())
        else:
            skipped_shapes += 1

    if skipped_shapes:
        logging.warning(
            "Slide %d: skipped %d non-text shape(s) (images, charts, SmartArt, etc.)",
            index + 1,
            skipped_shapes,
        )

    # Emit a docling-compatible page marker so smart_chunk works uniformly
    page_marker = f"<!-- page {index + 1} -->"
    headline = f"## Slide {index + 1}: {slide_title or 'Untitled'}"
    body = clean_markdown("\n".join(text_lines))
    return f"{page_marker}\n{headline}\n\n{body}"


def extract_markdown_from_pptx(path: Path) -> str:
    if Presentation is None:
        raise RuntimeError(
            "python-pptx is required. Install with: pip install python-pptx"
        )
    prs = Presentation(str(path))
    slides_md: List[str] = [f"# {path.stem}"]
    for idx, slide in enumerate(prs.slides):
        slides_md.append(slide_to_markdown(slide, idx))
    return "\n\n".join(slides_md).strip() + "\n"


# ---------------------------------------------------------------------------
# PPT -> PPTX conversion (Windows COM, then LibreOffice fallback)
# ---------------------------------------------------------------------------

def _convert_ppt_via_com(path: Path) -> Optional[Path]:
    if comtypes is None:
        return None
    try:
        powerpoint = comtypes.client.CreateObject("PowerPoint.Application")
        powerpoint.Visible = 1
        prs = powerpoint.Presentations.Open(str(path.resolve()), WithWindow=False)
        out = path.with_suffix(".pptx")
        prs.SaveAs(str(out.resolve()), 24)  # ppSaveAsOpenXMLPresentation
        prs.Close()
        powerpoint.Quit()
        return out if out.exists() else None
    except Exception:
        logging.exception("COM conversion failed for %s", path.name)
        return None


def _convert_ppt_via_libreoffice(path: Path) -> Optional[Path]:
    """Cross-platform fallback using LibreOffice headless."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    "libreoffice",
                    "--headless",
                    "--convert-to", "pptx",
                    "--outdir", tmpdir,
                    str(path.resolve()),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                logging.warning("LibreOffice stderr: %s", result.stderr.strip())
                return None
            out_tmp = Path(tmpdir) / path.with_suffix(".pptx").name
            if out_tmp.exists():
                dest = path.with_suffix(".pptx")
                dest.write_bytes(out_tmp.read_bytes())
                return dest
    except FileNotFoundError:
        logging.warning("LibreOffice not found. Install libreoffice for .ppt support.")
    except subprocess.TimeoutExpired:
        logging.warning("LibreOffice timed out converting %s", path.name)
    except Exception:
        logging.exception("LibreOffice conversion failed for %s", path.name)
    return None


def convert_ppt_to_pptx(path: Path) -> Optional[Path]:
    converted = _convert_ppt_via_com(path)
    if converted:
        return converted
    return _convert_ppt_via_libreoffice(path)


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_markdown_from_pdf(path: Path) -> str:
    if DocumentConverter is None:
        logging.warning("docling not installed, falling back to pdfplumber for %s", path.name)
        return _fallback_pdf(path)
    try:
        converter = DocumentConverter()
        result = converter.convert(str(path))
        return clean_markdown(result.document.export_to_markdown())
    except Exception:
        logging.warning("docling failed, falling back to pdfplumber for %s", path.name)
        return _fallback_pdf(path)


def _fallback_pdf(path: Path) -> str:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError(
            "Fallback PDF processing requires pdfplumber: pip install pdfplumber"
        ) from exc

    chunks: List[str] = [f"# {path.stem}"]
    with pdfplumber.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                # Emit page markers compatible with smart_chunk
                chunks.append(f"<!-- page {page_num} -->\n## Page {page_num}\n\n{clean_markdown(text)}")
    return "\n\n".join(chunks).strip() + "\n"


# ---------------------------------------------------------------------------
# Output: write one file per chunk
# ---------------------------------------------------------------------------

def _chunk_output_path(source_stem: str, index: int, total: int) -> Path:
    """
    Single chunk  → raw/report.md
    Multiple chunks → raw/report_chunk_01.md, raw/report_chunk_02.md, …
    """
    if total == 1:
        return OUTPUT_DIR / f"{source_stem}.md"
    padded = str(index + 1).zfill(len(str(total)))
    return OUTPUT_DIR / f"{source_stem}_chunk_{padded}.md"


def _remove_existing_chunks(source_stem: str) -> None:
    """Delete any previously generated chunks for this source file."""
    for old in OUTPUT_DIR.glob(f"{source_stem}*.md"):
        old.unlink()
        logging.debug("Removed stale chunk: %s", old.name)


def write_chunks(source_path: Path, chunks: List[str]) -> None:
    stem = source_path.stem
    _remove_existing_chunks(stem)
    total = len(chunks)
    for idx, chunk_body in enumerate(chunks):
        out_path = _chunk_output_path(stem, idx, total)
        header = build_yaml_header(source_path, chunk_index=idx, total_chunks=total)
        out_path.write_text(header + chunk_body, encoding="utf-8")
        logging.info("Written: %s (%d words)", out_path.name, len(chunk_body.split()))


# ---------------------------------------------------------------------------
# Per-file orchestration
# ---------------------------------------------------------------------------

def process_file(path: Path) -> None:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        markdown_body = extract_markdown_from_pdf(path)

    elif suffix == ".pptx":
        markdown_body = extract_markdown_from_pptx(path)

    elif suffix == ".ppt":
        converted = convert_ppt_to_pptx(path)
        if converted and converted.exists():
            markdown_body = extract_markdown_from_pptx(converted)
        else:
            raise RuntimeError(
                f"Unable to convert {path.name} to PPTX. "
                "Install LibreOffice (cross-platform) or run on Windows with PowerPoint."
            )

    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    # Duplicate-stem guard
    existing = list(OUTPUT_DIR.glob(f"{path.stem}*.md"))
    if existing:
        logging.debug("Overwriting %d existing chunk(s) for %s", len(existing), path.stem)

    chunks = smart_chunk(markdown_body, threshold=3000, soft_tolerance=0.10, overlap_sentences=2)

    if not chunks:
        logging.warning("No content extracted from %s — skipping output.", path.name)
        return

    write_chunks(path, chunks)
    logging.info(
        "Processed %s → %d chunk(s)", path.name, len(chunks)
    )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def batch_process() -> None:
    ensure_directories()

    # Detect stem collisions across different extensions
    stems: dict[str, Path] = {}
    for p in INPUT_DIR.iterdir():
        if p.suffix.lower() in SUPPORTED_EXTENSIONS:
            if p.stem in stems:
                logging.warning(
                    "Stem collision: '%s' and '%s' share the stem '%s'. "
                    "The second will overwrite the first's output.",
                    stems[p.stem].name, p.name, p.stem,
                )
            stems[p.stem] = p

    input_files = sorted(stems.values())
    if not input_files:
        logging.warning(
            "No supported files found in %s. Add PDFs, PPTX, or PPT files.", INPUT_DIR
        )
        return

    for input_file in input_files:
        try:
            logging.info("Processing %s", input_file.name)
            process_file(input_file)
        except Exception:
            logging.exception("Skipping failed file: %s", input_file.name)


def main() -> None:
    setup_logging()
    logging.info("Starting batch conversion: %s → %s", INPUT_DIR, OUTPUT_DIR)
    batch_process()
    logging.info("Batch conversion complete.")


if __name__ == "__main__":
    main()
