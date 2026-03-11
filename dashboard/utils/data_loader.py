#!/usr/bin/env python3
"""Unified data loader for InfiniMetrics dashboard."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .data_sources import DataSource, FileDataSource, MongoDataSource
from .data_utils import extract_accelerator_types, extract_run_info, get_friendly_size

logger = logging.getLogger(__name__)


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
        results_dir: str = "../output",
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
            logger.warning("MongoDB unavailable, falling back to file-based loading")
            self._source = FileDataSource(str(self.results_dir))
            self._use_mongodb = False
        else:
            raise RuntimeError("MongoDB connection failed and fallback is disabled")

    @property
    def source_type(self) -> str:
        """Get the current data source type."""
        return self._source.source_type if self._source else "none"

    @property
    def is_connected(self) -> bool:
        """Check if data source is available."""
        return self._source is not None

    @property
    def is_using_mongodb(self) -> bool:
        """Check if currently using MongoDB."""
        return (
            self._use_mongodb and self._source and self._source.source_type == "mongodb"
        )

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

    def load_summaries(self) -> List[Dict[str, Any]]:
        """Load dispatcher summaries from the current data source."""
        if self._source is None:
            return []
        return self._source.load_summaries()

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


# Re-export from sibling modules
from .data_sources import DataSource, FileDataSource, MongoDataSource
from .data_utils import (
    get_friendly_size,
    extract_accelerator_types,
    extract_run_info,
)

__all__ = [
    "InfiniMetricsDataLoader",
    "get_friendly_size",
    "extract_accelerator_types",
    "extract_run_info",
]
