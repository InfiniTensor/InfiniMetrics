#!/usr/bin/env python3
"""File system watcher for auto-importing test results."""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from watchdog.observers import Observer

from .handlers import TestResultHandler
from .importer import DataImporter

logger = logging.getLogger(__name__)


class FileWatcher:
    """
    File system watcher for auto-importing test results.

    Can monitor multiple directories:
    - output/ directory for individual test results
    - summary_output/ directory for dispatcher summaries
    """

    def __init__(
        self,
        importer: DataImporter,
        on_import: Optional[Callable[[str], None]] = None,
    ):
        self._importer = importer
        self._on_import = on_import
        self._observers: List[Observer] = []
        self._running = False
        self._watch_dirs: List[Path] = []

    def add_watch_dir(self, watch_dir: Path) -> None:
        """Add a directory to watch."""
        watch_dir = Path(watch_dir)
        if not watch_dir.exists():
            logger.warning(f"Watch directory does not exist: {watch_dir}")
            return

        if watch_dir not in self._watch_dirs:
            self._watch_dirs.append(watch_dir)
            logger.info(f"Added watch directory: {watch_dir}")

    def start(self) -> None:
        """Start watching all configured directories."""
        if self._running:
            logger.warning("Watcher already running")
            return

        if not self._watch_dirs:
            logger.warning("No directories configured to watch")
            return

        logger.info("Performing initial scan...")
        self.run_once()

        handler = TestResultHandler(self._importer, self._on_import)

        for watch_dir in self._watch_dirs:
            observer = Observer()
            observer.schedule(handler, str(watch_dir), recursive=True)
            observer.start()
            self._observers.append(observer)
            logger.info(f"Watching {watch_dir} for new test results")

        self._running = True

    def stop(self) -> None:
        """Stop watching all directories."""
        for observer in self._observers:
            observer.stop()
            observer.join()

        self._observers = []
        self._running = False
        logger.info("Watcher stopped")

    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._running

    def run_once(
        self,
        output_dir: Optional[Path] = None,
        summary_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Perform a one-time scan without starting continuous watch."""
        dirs_to_scan = []

        if summary_dir:
            dirs_to_scan.append(("summary", Path(summary_dir)))
        if output_dir:
            dirs_to_scan.append(("output", Path(output_dir)))

        if not dirs_to_scan:
            for watch_dir in self._watch_dirs:
                if "summary" in watch_dir.name.lower():
                    dirs_to_scan.append(("summary", watch_dir))
                else:
                    dirs_to_scan.append(("output", watch_dir))

        if len(dirs_to_scan) >= 2:
            summary_d = next((d for t, d in dirs_to_scan if t == "summary"), None)
            output_d = next((d for t, d in dirs_to_scan if t == "output"), None)
            if summary_d and output_d:
                logger.info(f"One-time scan: summary_dir={summary_d}, output_dir={output_d}")
                return self._importer.import_all(
                    output_dir=output_d,
                    summary_dir=summary_d,
                )

        combined: Dict[str, Any] = {"imported": [], "skipped": [], "failed": []}

        for dir_type, scan_dir in dirs_to_scan:
            if not scan_dir.exists():
                continue

            logger.info(f"Scanning {dir_type} directory: {scan_dir}")

            if dir_type == "summary":
                for summary_file in sorted(scan_dir.glob("dispatcher_summary_*.json")):
                    result = self._importer.import_dispatcher_summary(summary_file)
                    combined["imported"].extend(result["imported"])
                    combined["skipped"].extend(result["skipped"])
                    combined["failed"].extend(result["failed"])
            else:
                result = self._importer.import_directory(
                    scan_dir,
                    recursive=True,
                    include_summaries=False,
                )
                for run_id in result["imported"]:
                    if run_id not in combined["imported"]:
                        combined["imported"].append(run_id)
                combined["skipped"].extend(result["skipped"])
                combined["failed"].extend(result["failed"])

        logger.info(
            f"Scan completed: {len(combined['imported'])} imported, "
            f"{len(combined['skipped'])} skipped, {len(combined['failed'])} failed"
        )
        return combined

    def run_forever(self) -> None:
        """Start watcher and block until interrupted."""
        self.start()
        try:
            while self.is_running():
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()


class MultiDirWatcher:
    """Convenience class for watching both output and summary directories."""

    def __init__(
        self,
        importer: DataImporter,
        output_dir: Optional[Path] = None,
        summary_dir: Optional[Path] = None,
        on_import: Optional[Callable[[str], None]] = None,
    ):
        self._watcher = FileWatcher(importer, on_import)

        if output_dir:
            self._watcher.add_watch_dir(Path(output_dir))
        if summary_dir:
            self._watcher.add_watch_dir(Path(summary_dir))

        self._output_dir = Path(output_dir) if output_dir else Path("./output")
        self._summary_dir = Path(summary_dir) if summary_dir else Path("./summary_output")

    def start(self) -> None:
        self._watcher.start()

    def stop(self) -> None:
        self._watcher.stop()

    def is_running(self) -> bool:
        return self._watcher.is_running()

    def run_once(self) -> Dict[str, Any]:
        return self._watcher.run_once(self._output_dir, self._summary_dir)

    def run_forever(self) -> None:
        self._watcher.run_forever()
