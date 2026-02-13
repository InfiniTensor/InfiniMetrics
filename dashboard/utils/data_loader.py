#!/usr/bin/env python3
"""Data loading utilities for InfiniMetrics dashboard."""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

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

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Return the data source type name."""
        pass


class FileDataSource(DataSource):
    """File-based data source (reads from JSON/CSV files)."""

    def __init__(self, results_dir: str = "./test_output"):
        self.results_dir = Path(results_dir)

    @property
    def source_type(self) -> str:
        return "file"

    def list_test_runs(self, test_type: str = None) -> List[Dict[str, Any]]:
        """List all test runs, filtering out summary files."""
        runs = []

        # Search for JSON result files
        for json_file in self.results_dir.rglob("*.json"):
            try:
                # Skip summary files and dispatcher files
                if (
                    "summary" in json_file.name.lower()
                    or "dispatcher" in json_file.name.lower()
                ):
                    continue

                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Filter: must be a test result file, not a summary file
                if not self._is_test_result_file(data):
                    continue

                # Filter by test type if specified
                testcase = data.get("testcase", "")
                if test_type and not testcase.startswith(test_type):
                    continue

                # Extract basic info
                run_info = self._extract_run_info(data, json_file)

                # Extract the accelerator card type
                run_info["accelerator_types"] = extract_accelerator_types(data)
                runs.append(run_info)

            except Exception as e:
                logger.debug(f"Skipping file {json_file}: {e}")

        # Sort by time (newest first)
        runs.sort(key=lambda x: x["time"], reverse=True)
        return runs

    def load_test_result(self, json_path: Path) -> Dict[str, Any]:
        """Load a single test result with all data."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load associated CSV files
        for metric in data.get("metrics", []):
            csv_url = metric.get("raw_data_url")
            if csv_url and not csv_url.startswith("http"):
                # Get the correct base directory
                base_dir = self._get_csv_base_dir(data, json_path)
                csv_path = self._resolve_csv_path(csv_url, base_dir)

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
        if not all(key in data for key in required):
            return False
        if "metrics" not in data:
            return False
        return True

    def _extract_run_info(
        self, data: Dict[str, Any], json_path: Path
    ) -> Dict[str, Any]:
        """Extract run info from test result data."""
        config = data.get("config", {})
        resolved = data.get("resolved", {})

        device_used = (
            resolved.get("device_used")
            or config.get("device_used")
            or config.get("device_involved", 1)
        )

        nodes = resolved.get("nodes") or data.get("environment", {}).get(
            "cluster_scale", 1
        )

        result_code = data.get("result_code", 1)
        success = result_code == 0

        metrics = data.get("metrics", [])
        metric_types = [
            m.get("name", "").split(".")[0] for m in metrics if m.get("name")
        ]

        return {
            "path": json_path,
            "testcase": data.get("testcase", "unknown"),
            "run_id": data.get("run_id", "unknown"),
            "time": data.get("time", ""),
            "success": success,
            "result_code": result_code,
            "test_type": self._extract_test_type(data.get("testcase", "")),
            "operation": self._extract_operation(data.get("testcase", "")),
            "config": config,
            "resolved": resolved,
            "device_used": device_used,
            "nodes": nodes,
            "metrics_count": len(metrics),
            "metric_types": list(set(metric_types)),
        }

    def _get_csv_base_dir(self, json_data: Dict[str, Any], json_path: Path) -> Path:
        """Get the correct base directory for CSV files."""
        config = json_data.get("config", {})
        output_dir = config.get("output_dir")

        if output_dir:
            output_path = Path(output_dir)
            if output_path.is_absolute():
                return output_path
            return json_path.parent / output_dir

        return json_path.parent

    def _resolve_csv_path(self, csv_url: str, base_dir: Path) -> Optional[Path]:
        """Resolve CSV path from raw_data_url and base_dir."""
        try:
            if not csv_url:
                return None

            rel = csv_url[2:] if csv_url.startswith("./") else csv_url
            rel_path = Path(rel)

            candidates = [
                base_dir / rel_path,
                base_dir / rel_path.name,
                base_dir.parent / rel_path,
                base_dir.parent / rel_path.name,
            ]

            for p in candidates:
                if p.exists():
                    return p

            return None
        except Exception:
            return None

    def _extract_test_type(self, testcase: str) -> str:
        """Extract test type from testcase string."""
        parts = testcase.split(".")
        if len(parts) > 0:
            return parts[0]
        return "unknown"

    def _extract_operation(self, testcase: str) -> str:
        """Extract operation from testcase string."""
        parts = testcase.split(".")
        if len(parts) > 2:
            return parts[2]
        return "unknown"


class MongoDataSource(DataSource):
    """MongoDB-based data source."""

    def __init__(self, config=None):
        """
        Initialize MongoDB data source.

        Args:
            config: Optional DatabaseConfig. If None, loads from environment.
        """
        self._config = config
        self._client = None
        self._repository = None
        self._connected = False

    def _connect(self):
        """Lazy connection to MongoDB."""
        if self._connected:
            return self._connected

        try:
            from infinimetrics.db import MongoDBClient, TestRunRepository

            if self._config:
                self._client = MongoDBClient(self._config)
            else:
                self._client = MongoDBClient()

            if self._client.health_check():
                from infinimetrics.db.config import DatabaseConfig

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
            run_info = self._extract_run_info(run)
            run_info["accelerator_types"] = extract_accelerator_types(run)
            result.append(run_info)

        # Sort by time (newest first)
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

        # Convert embedded data arrays to DataFrames for compatibility
        for metric in data.get("metrics", []):
            if "data" in metric and isinstance(metric["data"], list):
                if metric["data"]:
                    metric["data"] = pd.DataFrame(metric["data"])
                    if "data_columns" not in metric:
                        metric["data_columns"] = list(metric["data"].columns)

        # Remove MongoDB internal fields
        data.pop("_id", None)
        data.pop("_metadata", None)

        return data

    def _extract_run_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract run info from MongoDB document."""
        config = data.get("config", {})
        resolved = data.get("resolved", {})

        device_used = (
            resolved.get("device_used")
            or config.get("device_used")
            or config.get("device_involved", 1)
        )

        nodes = resolved.get("nodes") or data.get("environment", {}).get(
            "cluster_scale", 1
        )

        result_code = data.get("result_code", 1)
        success = result_code == 0

        metrics = data.get("metrics", [])
        metric_types = [
            m.get("name", "").split(".")[0] for m in metrics if m.get("name")
        ]

        testcase = data.get("testcase", "")

        return {
            "path": None,  # No file path for MongoDB
            "run_id": data.get("run_id", "unknown"),
            "testcase": testcase,
            "time": data.get("time", ""),
            "success": success,
            "result_code": result_code,
            "test_type": self._extract_test_type(testcase),
            "operation": self._extract_operation(testcase),
            "config": config,
            "resolved": resolved,
            "device_used": device_used,
            "nodes": nodes,
            "metrics_count": len(metrics),
            "metric_types": list(set(metric_types)),
        }

    def _extract_test_type(self, testcase: str) -> str:
        """Extract test type from testcase string."""
        parts = testcase.split(".")
        if len(parts) > 0:
            return parts[0]
        return "unknown"

    def _extract_operation(self, testcase: str) -> str:
        """Extract operation from testcase string."""
        parts = testcase.split(".")
        if len(parts) > 2:
            return parts[2]
        return "unknown"


class InfiniMetricsDataLoader:
    """
    Unified data loader supporting multiple sources.

    Supports:
    - File-based loading (default)
    - MongoDB-based loading
    - Automatic fallback from MongoDB to files
    """

    def __init__(
        self,
        results_dir: str = "./test_output",
        use_mongodb: bool = False,
        mongo_config=None,
        fallback_to_files: bool = True,
    ):
        """
        Initialize the data loader.

        Args:
            results_dir: Directory containing test result files
            use_mongodb: If True, use MongoDB as primary data source
            mongo_config: Optional MongoDB configuration
            fallback_to_files: If True, fall back to file loading if MongoDB fails
        """
        self.results_dir = Path(results_dir)
        self._fallback_to_files = fallback_to_files
        self._use_mongodb = use_mongodb
        self._mongo_config = mongo_config
        self._source: Optional[DataSource] = None

        if use_mongodb:
            self._init_mongodb_source()
        else:
            self._source = FileDataSource(results_dir)

    def _init_mongodb_source(self):
        """Initialize MongoDB data source with optional fallback."""
        mongo_source = MongoDataSource(self._mongo_config)

        if mongo_source.is_connected():
            self._source = mongo_source
        elif self._fallback_to_files:
            logger.warning(
                "MongoDB unavailable, falling back to file-based loading"
            )
            self._source = FileDataSource(str(self.results_dir))
            self._use_mongodb = False
        else:
            raise RuntimeError("MongoDB connection failed and fallback is disabled")

    @property
    def source_type(self) -> str:
        """Get the current data source type."""
        return self._source.source_type if self._source else "none"

    @property
    def is_using_mongodb(self) -> bool:
        """Check if currently using MongoDB."""
        return self._use_mongodb and self._source and self._source.source_type == "mongodb"

    def switch_to_mongodb(self, mongo_config=None) -> bool:
        """
        Switch to MongoDB data source.

        Returns:
            True if switch was successful
        """
        if mongo_config:
            self._mongo_config = mongo_config

        mongo_source = MongoDataSource(self._mongo_config)

        if mongo_source.is_connected():
            self._source = mongo_source
            self._use_mongodb = True
            return True
        elif self._fallback_to_files:
            logger.warning("Failed to switch to MongoDB, keeping current source")
            return False
        else:
            raise RuntimeError("MongoDB connection failed")

    def switch_to_files(self, results_dir: str = None):
        """Switch to file-based data source."""
        if results_dir:
            self.results_dir = Path(results_dir)
        self._source = FileDataSource(str(self.results_dir))
        self._use_mongodb = False

    def list_test_runs(self, test_type: str = None) -> List[Dict[str, Any]]:
        """List all test runs."""
        if self._source is None:
            return []
        return self._source.list_test_runs(test_type)

    def load_test_result(self, identifier) -> Dict[str, Any]:
        """
        Load a single test result with all data.

        Args:
            identifier: For file source, a Path to JSON file.
                       For MongoDB source, a run_id string.
        """
        if self._source is None:
            return {}
        return self._source.load_test_result(identifier)

    # Keep backward compatibility methods
    def load_csv_data(
        self, csv_url: str, json_data: Dict[str, Any], json_path: Path
    ) -> Optional[pd.DataFrame]:
        """Load CSV data file using proper path resolution (file source only)."""
        if isinstance(self._source, FileDataSource):
            try:
                if csv_url.startswith("http"):
                    return None

                base_dir = self._source._get_csv_base_dir(json_data, json_path)
                csv_path = self._source._resolve_csv_path(csv_url, base_dir)

                if csv_path and csv_path.exists():
                    return pd.read_csv(csv_path)
            except Exception as e:
                logger.error(f"Failed to load CSV {csv_url}: {e}")
        return None


def load_summary_file(summary_path: str = "./summary_output") -> List[Dict[str, Any]]:
    """Load dispatcher summary files."""
    summaries = []
    summary_dir = Path(summary_path)

    if summary_dir.exists():
        for json_file in sorted(
            summary_dir.glob("dispatcher_summary_*.json"), reverse=True
        ):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["file"] = json_file.name
                data["timestamp"] = json_file.stem.replace("dispatcher_summary_", "")
                summaries.append(data)
            except Exception as e:
                logger.warning(f"Failed to load summary {json_file}: {e}")

    return summaries


def get_friendly_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def extract_accelerator_types(result_json: dict) -> list[str]:
    """Extract the accelerator card type from result_json."""
    types = set()
    try:
        clusters = result_json.get("environment", {}).get("cluster", [])
        for node in clusters:
            accs = node.get("machine", {}).get("accelerators", [])
            for acc in accs:
                if "type" in acc:
                    types.add(acc["type"])
    except Exception:
        pass
    return list(types)
