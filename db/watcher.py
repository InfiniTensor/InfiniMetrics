#!/usr/bin/env python3
"""
File watcher for auto-importing test results to MongoDB.

Usage:
    # Start watching for new files
    python -m db.watcher

    # One-time scan (import existing files)
    python -m db.watcher --scan
"""

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .client import MongoDBClient
from .config import DatabaseConfig
from .importer import DataImporter
from .repository import DispatcherSummaryRepository, TestRunRepository
from .utils import is_dispatcher_summary_file, should_skip_file

logger = logging.getLogger(__name__)


class Watcher:
    """Watch directories and auto-import test results to MongoDB.

    - output_dir: imports test result files to test_runs collection
    - summary_dir: stores summary metadata to dispatcher_summaries collection
    """

    def __init__(
        self,
        output_dir: Path = None,
        summary_dir: Path = None,
        mongo_uri: str = None,
    ):
        self.output_dir = Path(output_dir) if output_dir else Path("./output")
        self.summary_dir = (
            Path(summary_dir) if summary_dir else Path("./summary_output")
        )

        # Initialize MongoDB connection
        config = DatabaseConfig.from_env()
        if mongo_uri:
            config.mongo_uri = mongo_uri

        client = MongoDBClient(config)
        if not client.health_check():
            raise ConnectionError(f"Cannot connect to MongoDB: {config.mongo_uri}")

        # Repositories
        self.test_run_repo = TestRunRepository(
            client.get_collection(config.collection_name)
        )
        self.summary_repo = DispatcherSummaryRepository(
            client.get_collection(config.summary_collection_name)
        )
        self.importer = DataImporter(self.test_run_repo)

        self._observer = None
        self._processed: Set[str] = set()

    def scan(self) -> Dict[str, Dict[str, List[str]]]:
        """
        One-time scan and import all existing files.

        Returns:
            {
                "output": {"imported": [...], "skipped": [...], "failed": [...]},
                "summary": {"imported": [...], "skipped": [...], "failed": [...]}
            }
        """
        result = {
            "output": {"imported": [], "skipped": [], "failed": []},
            "summary": {"imported": [], "skipped": [], "failed": []},
        }

        # Import test results from output directory
        if self.output_dir.exists():
            logger.info(f"Scanning output: {self.output_dir}")
            output_result = self.importer.import_directory(
                self.output_dir, recursive=True, include_summaries=False
            )
            result["output"]["imported"].extend(output_result.get("imported", []))
            result["output"]["skipped"].extend(output_result.get("skipped", []))
            result["output"]["failed"].extend(output_result.get("failed", []))
            logger.info(
                f"Output: {len(result['output']['imported'])} imported, "
                f"{len(result['output']['skipped'])} skipped, "
                f"{len(result['output']['failed'])} failed"
            )

        # Store summary metadata (not re-import test results)
        if self.summary_dir.exists():
            logger.info(f"Scanning summaries: {self.summary_dir}")
            for summary_file in sorted(
                self.summary_dir.glob("dispatcher_summary_*.json")
            ):
                summary_result = self._import_summary_metadata(summary_file)
                result["summary"]["imported"].extend(summary_result.get("imported", []))
                result["summary"]["skipped"].extend(summary_result.get("skipped", []))
                result["summary"]["failed"].extend(summary_result.get("failed", []))
            logger.info(
                f"Summary: {len(result['summary']['imported'])} imported, "
                f"{len(result['summary']['skipped'])} skipped, "
                f"{len(result['summary']['failed'])} failed"
            )

        return result

    def _import_summary_metadata(self, summary_path: Path) -> Dict[str, List[str]]:
        """
        Import summary metadata to dispatcher_summaries collection.

        This stores the summary info (which tests ran, success/failure)
        without re-importing the actual test results.
        """
        result = {"imported": [], "skipped": [], "failed": []}

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)

            timestamp = summary_data.get("timestamp")
            if not timestamp:
                logger.warning(f"No timestamp in summary: {summary_path}")
                result["failed"].append(str(summary_path))
                return result

            # Check if already exists
            if self.summary_repo.exists(timestamp):
                logger.debug(f"Summary already exists: {timestamp}")
                result["skipped"].append(timestamp)
                return result

            # Add source file info
            summary_data.setdefault("_metadata", {})
            summary_data["_metadata"]["source_file"] = summary_path.name

            # Insert summary metadata
            self.summary_repo.upsert(summary_data)
            result["imported"].append(timestamp)
            logger.info(f"Imported summary: {timestamp}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {summary_path}: {e}")
            result["failed"].append(str(summary_path))
        except Exception as e:
            logger.error(f"Failed to import summary {summary_path}: {e}")
            result["failed"].append(str(summary_path))

        return result

    def start(self):
        """Start watching for new files."""
        logger.info("Performing initial scan...")
        self.scan()

        # Setup file event handler
        handler = _Handler(self, self._processed)

        self._observer = Observer()
        if self.output_dir.exists():
            self._observer.schedule(handler, str(self.output_dir), recursive=True)
            logger.info(f"Watching: {self.output_dir}")
        if self.summary_dir.exists():
            self._observer.schedule(handler, str(self.summary_dir), recursive=True)
            logger.info(f"Watching: {self.summary_dir}")

        self._observer.start()
        logger.info("Watcher started. Press Ctrl+C to stop.")

    def stop(self):
        """Stop watching."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            logger.info("Watcher stopped")

    def run_forever(self):
        """Start watcher and block until interrupted."""
        self.start()
        try:
            while self._observer and self._observer.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()


class _Handler(FileSystemEventHandler):
    """Internal handler for file system events."""

    def __init__(self, watcher: Watcher, processed: Set[str]):
        self.watcher = watcher
        self.processed = processed

    def on_created(self, event):
        if event.is_directory:
            return
        self._process_file(Path(event.src_path))

    def on_modified(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if str(path) not in self.processed:
            self._process_file(path)

    def _process_file(self, path: Path):
        if path.suffix != ".json":
            return

        path_str = str(path)
        if path_str in self.processed:
            return

        # Wait for file to be fully written
        time.sleep(0.5)

        if is_dispatcher_summary_file(path):
            logger.info(f"New summary: {path.name}")
            result = self.watcher._import_summary_metadata(path)
            if result["imported"]:
                self.processed.add(path_str)
                logger.info(f"Imported summary metadata")
        elif not should_skip_file(path):
            logger.info(f"New test result: {path.name}")
            run_id = self.watcher.importer.import_test_result(path)
            if run_id:
                self.processed.add(path_str)
                logger.info(f"Imported: {run_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Watch for test results and import to MongoDB"
    )
    parser.add_argument("--scan", action="store_true", help="One-time scan only")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument(
        "--summary-dir", default="./summary_output", help="Summary directory"
    )
    parser.add_argument("--mongo-uri", help="MongoDB connection URI")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        watcher = Watcher(
            output_dir=args.output_dir,
            summary_dir=args.summary_dir,
            mongo_uri=args.mongo_uri,
        )

        if args.scan:
            result = watcher.scan()
            print("\n=== Output Files (test_runs) ===")
            print(f"Imported: {len(result['output']['imported'])}")
            print(f"Skipped:  {len(result['output']['skipped'])}")
            print(f"Failed:   {len(result['output']['failed'])}")
            print("\n=== Summary Files (dispatcher_summaries) ===")
            print(f"Imported: {len(result['summary']['imported'])}")
            print(f"Skipped:  {len(result['summary']['skipped'])}")
            print(f"Failed:   {len(result['summary']['failed'])}")
        else:
            watcher.run_forever()

    except ConnectionError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
