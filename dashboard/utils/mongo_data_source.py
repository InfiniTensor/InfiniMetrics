#!/usr/bin/env python3
"""MongoDB data source for InfiniMetrics dashboard."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .data_sources import DataSource
from .data_utils import extract_accelerator_types, extract_run_info

logger = logging.getLogger(__name__)


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
