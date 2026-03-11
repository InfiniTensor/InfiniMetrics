#!/usr/bin/env python3
"""Data source implementations for InfiniMetrics dashboard."""

import json
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .data_utils import extract_accelerator_types, extract_run_info, load_summary_file
from db.utils import get_csv_base_dir, resolve_csv_path

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract data source for test results."""

    @abstractmethod
    def list_test_runs(self, test_type: str = None) -> List[Dict[str, Any]]:
        """List all test runs."""
        pass

    @abstractmethod
    def load_test_result(self, identifier) -> Dict[str, Any]:
        """Load a single test result with full data."""
        pass

    @abstractmethod
    def load_summaries(self) -> List[Dict[str, Any]]:
        """Load dispatcher summaries."""
        pass

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Return the data source type name."""
        pass


class FileDataSource(DataSource):
    """File-based data source (reads from JSON/CSV files)."""

    def __init__(self, results_dir: str = "../output"):
        self.results_dir = Path(results_dir)

    @property
    def source_type(self) -> str:
        return "file"

    def list_test_runs(self, test_type: str = None) -> List[Dict[str, Any]]:
        """List all test runs, filtering out summary files."""
        runs = []

        for json_file in self.results_dir.rglob("*.json"):
            try:
                if (
                    "summary" in json_file.name.lower()
                    or "dispatcher" in json_file.name.lower()
                ):
                    continue

                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if not self._is_test_result_file(data):
                    continue

                testcase = data.get("testcase", "")
                if test_type and not testcase.startswith(test_type):
                    continue

                run_info = extract_run_info(data, json_file)
                run_info["accelerator_types"] = extract_accelerator_types(data)
                runs.append(run_info)

            except Exception as e:
                logger.debug(f"Skipping file {json_file}: {e}")

        runs.sort(key=lambda x: x["time"], reverse=True)
        return runs

    def load_test_result(self, json_path: Path) -> Dict[str, Any]:
        """Load a single test result with all data."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for metric in data.get("metrics", []):
            csv_url = metric.get("raw_data_url")
            if csv_url and not csv_url.startswith("http"):
                base_dir = get_csv_base_dir(data, json_path)
                csv_path = resolve_csv_path(csv_url, base_dir)

                if csv_path and csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        metric["data"] = df
                        metric["data_columns"] = list(df.columns)
                        metric["csv_path"] = str(csv_path)
                    except Exception as e:
                        logger.warning(f"Failed to load CSV {csv_path}: {e}")
                        metric["data"] = None
                else:
                    logger.debug(f"CSV not found: {csv_url} (base: {base_dir})")
                    metric["data"] = None

        return data

    def _is_test_result_file(self, data: Dict[str, Any]) -> bool:
        """Check if JSON file is a test result (not a summary)."""
        required = ["run_id", "testcase", "config"]
        return all(key in data for key in required) and "metrics" in data

    def load_summaries(self) -> List[Dict[str, Any]]:
        """Load dispatcher summary files from summary_output directory."""
        summary_dir = self.results_dir.parent / "summary_output"
        return load_summary_file(str(summary_dir))


class MongoDataSource(DataSource):
    """MongoDB-based data source."""

    def __init__(self, config=None):
        self._config = config
        self._client = None
        self._repository = None
        self._connected = False

    def _connect(self):
        """Lazy connection to MongoDB."""
        if self._connected:
            return self._connected

        try:
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            from db import MongoDBClient, TestRunRepository

            if self._config:
                self._client = MongoDBClient(self._config)
            else:
                self._client = MongoDBClient()

            if self._client.health_check():
                from db.config import DatabaseConfig

                config = self._config or DatabaseConfig.from_env()
                self._repository = TestRunRepository(
                    self._client.get_collection(config.collection_name)
                )
                self._connected = True
                logger.info("Connected to MongoDB data source")
            else:
                logger.warning("MongoDB health check failed")

        except Exception as e:
            logger.warning(f"Failed to connect to MongoDB: {e}")
            self._connected = False

        return self._connected

    @property
    def source_type(self) -> str:
        return "mongodb"

    def is_connected(self) -> bool:
        """Check if MongoDB is connected."""
        return self._connected or self._connect()

    def list_test_runs(self, test_type: str = None) -> List[Dict[str, Any]]:
        """List all test runs from MongoDB."""
        if not self._connect():
            logger.warning("MongoDB not connected, returning empty list")
            return []

        runs = self._repository.list_test_runs(test_type=test_type)
        result = []

        for run in runs:
            run_info = extract_run_info(run)
            run_info["accelerator_types"] = extract_accelerator_types(run)
            result.append(run_info)

        result.sort(key=lambda x: x["time"], reverse=True)
        return result

    def load_test_result(self, run_id: str) -> Dict[str, Any]:
        """Load a single test result with full data from MongoDB."""
        if not self._connect():
            logger.warning("MongoDB not connected")
            return {}

        data = self._repository.find_by_run_id(run_id)
        if not data:
            return {}

        for metric in data.get("metrics", []):
            if "data" in metric and isinstance(metric["data"], list):
                if metric["data"]:
                    metric["data"] = pd.DataFrame(metric["data"])
                    if "data_columns" not in metric:
                        metric["data_columns"] = list(metric["data"].columns)

        data.pop("_id", None)
        data.pop("_metadata", None)

        return data

    def load_summaries(self) -> List[Dict[str, Any]]:
        """Load dispatcher summaries from MongoDB."""
        if not self._connect():
            logger.warning("MongoDB not connected, returning empty list")
            return []

        try:
            from db import DispatcherSummaryRepository
            from db.config import DatabaseConfig

            config = self._config or DatabaseConfig.from_env()
            summary_collection = self._client.get_collection(
                config.summary_collection_name
            )
            summary_repo = DispatcherSummaryRepository(summary_collection)
            summaries = summary_repo.list_summaries()

            for s in summaries:
                s.pop("_id", None)
                s.pop("_metadata", None)

            return summaries
        except Exception as e:
            logger.warning(f"Failed to load summaries from MongoDB: {e}")
            return []
