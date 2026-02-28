#!/usr/bin/env python3
"""Data importer for loading JSON/CSV test results into MongoDB."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .repository import TestRunRepository
from .utils import (
    get_csv_base_dir,
    is_dispatcher_summary,
    is_dispatcher_summary_file,
    is_valid_test_result,
    load_csv_data,
    resolve_csv_path,
    resolve_result_file_path,
    should_skip_file,
)

logger = logging.getLogger(__name__)


class DataImporter:
    """
    Import JSON/CSV test results to MongoDB.

    Supports hierarchical structure:
    - Dispatcher summary files (summary_output/dispatcher_summary_*.json)
    - Individual test result files (output/*_results.json)
    """

    def __init__(self, repository: TestRunRepository, base_dir: Optional[Path] = None):
        self._repository = repository
        self._base_dir = Path(base_dir) if base_dir else Path.cwd()

    def import_dispatcher_summary(
        self, summary_path: Path, overwrite: bool = False
    ) -> Dict[str, Any]:
        """Import a dispatcher summary file and all referenced test results."""
        summary: Dict[str, Any] = {
            "imported": [],
            "skipped": [],
            "failed": [],
            "summary_file": str(summary_path),
        }

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)

            if not is_dispatcher_summary(summary_data):
                logger.debug(f"Not a dispatcher summary: {summary_path}")
                return summary

            logger.info(
                f"Processing dispatcher summary: {summary_path} "
                f"({summary_data.get('total_tests', 0)} tests)"
            )

            for result_info in summary_data.get("results", []):
                result_file = result_info.get("result_file")
                if not result_file:
                    continue

                result_path = resolve_result_file_path(
                    result_file, summary_path, self._base_dir
                )

                if not result_path or not result_path.exists():
                    logger.warning(f"Result file not found: {result_file}")
                    summary["failed"].append(result_file)
                    continue

                imported_run_id, status = self.import_test_result(
                    result_path,
                    dispatcher_info={
                        "summary_file": str(summary_path),
                        "summary_timestamp": summary_data.get("timestamp"),
                        "total_tests": summary_data.get("total_tests"),
                    },
                    overwrite=overwrite,
                )

                if status == self.STATUS_IMPORTED:
                    summary["imported"].append(imported_run_id)
                elif status == self.STATUS_SKIPPED:
                    summary["skipped"].append(imported_run_id or result_info.get("run_id"))
                else:
                    summary["failed"].append(str(result_path))

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {summary_path}: {e}")
        except Exception as e:
            logger.error(f"Failed to process summary {summary_path}: {e}")

        logger.info(
            f"Summary processed: {len(summary['imported'])} imported, "
            f"{len(summary['skipped'])} skipped, {len(summary['failed'])} failed"
        )
        return summary

    # Return status constants
    STATUS_IMPORTED = "imported"
    STATUS_SKIPPED = "skipped"
    STATUS_FAILED = "failed"

    def import_test_result(
        self,
        result_path: Path,
        dispatcher_info: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> tuple[str, str]:
        """
        Import a single test result file to MongoDB.

        Returns:
            Tuple of (run_id, status) where status is one of:
            - "imported": Successfully imported
            - "skipped": Already exists or not a test result
            - "failed": Error during import
        """
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not is_valid_test_result(data):
                logger.debug(f"Skipping non-test file: {result_path}")
                return (None, self.STATUS_SKIPPED)

            run_id = data.get("run_id")
            if not run_id:
                logger.warning(f"No run_id in {result_path}")
                return (None, self.STATUS_SKIPPED)

            if self._repository.exists(run_id) and not overwrite:
                logger.debug(f"Skipping existing run_id: {run_id}")
                return (run_id, self.STATUS_SKIPPED)

            self._embed_csv_data(data, result_path)

            data.setdefault("_metadata", {})
            data["_metadata"]["source_file"] = str(result_path)
            data["_metadata"]["version"] = "1.0"

            if dispatcher_info:
                data["_metadata"]["dispatcher"] = dispatcher_info

            if overwrite and self._repository.exists(run_id):
                inserted_id = self._repository.upsert(data)
            else:
                inserted_id = self._repository.insert(data)

            return (run_id, self.STATUS_IMPORTED if inserted_id else self.STATUS_FAILED)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {result_path}: {e}")
            return (None, self.STATUS_FAILED)
        except Exception as e:
            logger.error(f"Failed to import {result_path}: {e}")
            return (None, self.STATUS_FAILED)

    def import_directory(
        self,
        directory: Path,
        recursive: bool = True,
        overwrite: bool = False,
        include_summaries: bool = True,
    ) -> Dict[str, Any]:
        """Import all JSON files from a directory."""
        summary: Dict[str, Any] = {"imported": [], "skipped": [], "failed": []}

        directory = Path(directory)
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return summary

        pattern = "**/*.json" if recursive else "*.json"
        json_files = list(directory.glob(pattern))

        logger.info(f"Found {len(json_files)} JSON files in {directory}")

        for json_file in json_files:
            if is_dispatcher_summary_file(json_file):
                if include_summaries:
                    result = self.import_dispatcher_summary(json_file, overwrite)
                    summary["imported"].extend(result["imported"])
                    summary["skipped"].extend(result["skipped"])
                    summary["failed"].extend(result["failed"])
                continue

            if should_skip_file(json_file):
                continue

            run_id, status = self.import_test_result(json_file, overwrite=overwrite)
            if status == self.STATUS_IMPORTED:
                summary["imported"].append(run_id)
            elif status == self.STATUS_SKIPPED:
                if run_id:
                    summary["skipped"].append(run_id)
            else:
                summary["failed"].append(str(json_file))

        logger.info(
            f"Import completed: {len(summary['imported'])} imported, "
            f"{len(summary['skipped'])} skipped, {len(summary['failed'])} failed"
        )
        return summary

    def import_all(
        self,
        output_dir: Optional[Path] = None,
        summary_dir: Optional[Path] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Import from both output and summary directories."""
        combined: Dict[str, Any] = {"imported": [], "skipped": [], "failed": []}

        if summary_dir:
            summary_dir = Path(summary_dir)
            if summary_dir.exists():
                logger.info(f"Importing from summary directory: {summary_dir}")
                for summary_file in sorted(summary_dir.glob("dispatcher_summary_*.json")):
                    result = self.import_dispatcher_summary(summary_file, overwrite)
                    combined["imported"].extend(result["imported"])
                    combined["skipped"].extend(result["skipped"])
                    combined["failed"].extend(result["failed"])

        if output_dir:
            output_dir = Path(output_dir)
            if output_dir.exists():
                logger.info(f"Importing from output directory: {output_dir}")
                result = self.import_directory(
                    output_dir, recursive=True, overwrite=overwrite, include_summaries=False
                )
                for run_id in result["imported"]:
                    if run_id not in combined["imported"]:
                        combined["imported"].append(run_id)
                combined["skipped"].extend(result["skipped"])
                combined["failed"].extend(result["failed"])

        logger.info(
            f"Total import: {len(combined['imported'])} imported, "
            f"{len(combined['skipped'])} skipped, {len(combined['failed'])} failed"
        )
        return combined

    def import_json_file(self, json_path: Path, overwrite: bool = False) -> Optional[str]:
        """Import a single JSON result file (legacy method)."""
        run_id, status = self.import_test_result(json_path, overwrite=overwrite)
        return run_id if status == self.STATUS_IMPORTED else None

    def _embed_csv_data(self, data: Dict[str, Any], json_path: Path) -> None:
        """Load CSV files and embed data into metrics."""
        base_dir = get_csv_base_dir(data, json_path)

        for metric in data.get("metrics", []):
            csv_url = metric.get("raw_data_url")
            if csv_url and not csv_url.startswith("http"):
                csv_path = resolve_csv_path(csv_url, base_dir)
                if csv_path and csv_path.exists():
                    try:
                        csv_data = load_csv_data(csv_path)
                        metric["data"] = csv_data
                        metric["data_columns"] = list(csv_data[0].keys()) if csv_data else []
                        logger.debug(f"Embedded CSV data from {csv_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load CSV {csv_path}: {e}")
