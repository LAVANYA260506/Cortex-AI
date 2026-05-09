#!/usr/bin/env python3
"""Live ingestion watcher for document conversion pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from threading import Event

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent

from convert_docs_to_markdown import (
    INPUT_DIR,
    OUTPUT_DIR,
    SUPPORTED_EXTENSIONS,
    process_file,
    setup_logging,
)

from chunker import smart_chunk

PROCESSED_DIR = Path("processed")
WRITE_COMPLETE_DELAY = 1.5  # seconds


class DocumentIngestionHandler(FileSystemEventHandler):
    """Event handler that watches for new document files and processes them."""

    def __init__(self) -> None:
        super().__init__()
        self._processed: set[Path] = set()

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.is_directory:
            return
        self._handle_file(Path(event.src_path))

    def on_modified(self, event: FileModifiedEvent) -> None:
        if event.is_directory:
            return
        self._handle_file(Path(event.src_path))

    def _is_supported(self, path: Path) -> bool:
        return path.suffix.lower() in SUPPORTED_EXTENSIONS

    def _output_exists(self, input_path: Path) -> bool:
        output_path = OUTPUT_DIR / f"{input_path.stem}.md"
        return output_path.exists()

    def _wait_for_write_complete(self, path: Path) -> None:
        time.sleep(WRITE_COMPLETE_DELAY)

    def _handle_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            return
        if not self._is_supported(path):
            logging.debug("Ignoring unsupported file: %s", path.name)
            return

        if path in self._processed:
            logging.debug("File already being processed: %s", path.name)
            return

        self._processed.add(path)
        try:
            logging.info("New file detected: %s", path.name)

            self._wait_for_write_complete(path)

            if self._output_exists(path):
                logging.info("Skipping already processed file: %s", path.name)
                return

            logging.info("Starting processing: %s", path.name)
            process_file(path)
            logging.info("Processing complete: %s", path.name)

            
            
            self._move_to_processed(path)

        except Exception as exc:
            logging.error("Failed to process file %s: %s", path.name, exc)
        finally:
            self._processed.discard(path)

    def _move_to_processed(self, path: Path) -> None:
        try:
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            destination = PROCESSED_DIR / path.name
            counter = 1
            while destination.exists():
                stem = path.stem
                suffix = path.suffix
                destination = PROCESSED_DIR / f"{stem}_{counter}{suffix}"
                counter += 1
            path.rename(destination)
            logging.info("Moved original file to: %s", destination.name)
        except Exception as exc:
            logging.warning("Could not move file to processed/ folder: %s", exc)


def start_watcher() -> None:
    """Start the file system watcher."""
    setup_logging()
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    event_handler = DocumentIngestionHandler()
    observer = Observer()
    observer.schedule(event_handler, str(INPUT_DIR), recursive=False)
    observer.start()

    logging.info("Watcher started - monitoring: %s", INPUT_DIR)
    logging.info("Output directory: %s", OUTPUT_DIR)
    logging.info("Processed folder: %s", PROCESSED_DIR)
    logging.info("Supported extensions: %s", ", ".join(SUPPORTED_EXTENSIONS))

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down watcher...")
        observer.stop()
    observer.join()
    logging.info("Watcher stopped.")


if __name__ == "__main__":
    start_watcher()