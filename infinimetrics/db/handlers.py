#!/usr/bin/env python3
"""File system event handlers for test result files."""

import logging
import time
from pathlib import Path
from typing import Callable, Optional, Set

from watchdog.events import FileSystemEventHandler

from .importer import DataImporter
from .utils import is_dispatcher_summary_file, should_skip_file

logger = logging.getLogger(__name__)


class TestResultHandler(FileSystemEventHandler):
    """Handle file system events for test result files."""

    def __init__(
        self,
        importer: DataImporter,
        on_import: Optional[Callable[[str], None]] = None,
        delay_seconds: float = 1.0,
    ):
        self._importer = importer
        self._on_import = on_import
        self._delay_seconds = delay_seconds
        self._processed_files: Set[str] = set()

    def on_created(self, event):
        """Handle file creation event."""
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path.suffix != ".json":
            return

        self._process_file(path)

    def on_modified(self, event):
        """Handle file modification event."""
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path.suffix != ".json":
            return

        path_str = str(path)
        if path_str not in self._processed_files:
            self._process_file(path)

    def _process_file(self, path: Path):
        """Process a JSON file."""
        path_str = str(path)

        if path_str in self._processed_files:
            return

        time.sleep(self._delay_seconds)

        if is_dispatcher_summary_file(path):
            logger.info(f"Detected new dispatcher summary: {path}")
            result = self._importer.import_dispatcher_summary(path)
            imported = result.get("imported", [])

            if imported:
                self._processed_files.add(path_str)
                for run_id in imported:
                    if self._on_import:
                        self._on_import(run_id)
                logger.info(f"Auto-imported {len(imported)} tests from summary")
            else:
                logger.warning(f"No tests imported from summary: {path}")
        else:
            if should_skip_file(path):
                return

            logger.info(f"Detected new file: {path}")
            result = self._importer.import_test_result(path)

            if result:
                self._processed_files.add(path_str)
                if self._on_import:
                    self._on_import(result)
                logger.info(f"Auto-imported: {result}")
            else:
                logger.warning(f"Failed to auto-import: {path}")
