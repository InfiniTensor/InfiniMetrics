#!/usr/bin/env python3
"""Repository layer for test_runs collection."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pymongo import ASCENDING, DESCENDING
from pymongo.collection import Collection

logger = logging.getLogger(__name__)


class TestRunRepository:
    """Repository for test_runs collection operations."""

    def __init__(self, collection: Collection):
        self._collection = collection
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """Create indexes if they don't exist."""
        try:
            # Primary lookup index (unique)
            self._collection.create_index("run_id", unique=True)

            # Testcase index for filtering
            self._collection.create_index("testcase")

            # Time index for sorting (descending)
            self._collection.create_index([("time", DESCENDING)])

            # Compound index for common queries
            self._collection.create_index([("testcase", ASCENDING), ("time", DESCENDING)])

            logger.debug("Indexes created/verified")
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")

    def insert(self, test_run: Dict[str, Any]) -> str:
        """
        Insert a test run document.

        Args:
            test_run: Test run document to insert

        Returns:
            Inserted run_id
        """
        # Add metadata
        test_run.setdefault("_metadata", {})
        test_run["_metadata"]["imported_at"] = datetime.utcnow().isoformat()

        result = self._collection.insert_one(test_run)
        logger.debug(f"Inserted test run: {test_run.get('run_id')}")
        return test_run["run_id"]

    def upsert(self, test_run: Dict[str, Any]) -> str:
        """
        Insert or update a test run document.

        Args:
            test_run: Test run document to insert/update

        Returns:
            Upserted run_id
        """
        run_id = test_run.get("run_id")
        if not run_id:
            raise ValueError("test_run must have a run_id")

        # Add metadata
        test_run.setdefault("_metadata", {})
        test_run["_metadata"]["imported_at"] = datetime.utcnow().isoformat()

        self._collection.update_one(
            {"run_id": run_id}, {"$set": test_run}, upsert=True
        )
        logger.debug(f"Upserted test run: {run_id}")
        return run_id

    def find_by_run_id(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a test run by run_id.

        Args:
            run_id: Unique run identifier

        Returns:
            Test run document or None
        """
        return self._collection.find_one({"run_id": run_id})

    def find_all(
        self,
        testcase_prefix: Optional[str] = None,
        result_code: Optional[int] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Query test runs with filters.

        Args:
            testcase_prefix: Filter by testcase prefix
            result_code: Filter by result code
            limit: Maximum number of results
            skip: Number of results to skip

        Returns:
            List of test run documents
        """
        query: Dict[str, Any] = {}

        if testcase_prefix:
            query["testcase"] = {"$regex": f"^{testcase_prefix}"}
        if result_code is not None:
            query["result_code"] = result_code

        cursor = (
            self._collection.find(query)
            .sort("time", DESCENDING)
            .skip(skip)
            .limit(limit)
        )
        return list(cursor)

    def list_test_runs(
        self, test_type: str = None, limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        List all test runs (compatible with file-based loader format).

        Args:
            test_type: Filter by test type (e.g., 'infer', 'comm', 'hardware')
            limit: Maximum number of results

        Returns:
            List of test run summaries
        """
        query: Dict[str, Any] = {}
        if test_type:
            query["testcase"] = {"$regex": f"^{test_type}"}

        projection = {
            "run_id": 1,
            "testcase": 1,
            "time": 1,
            "result_code": 1,
            "config": 1,
            "resolved": 1,
            "environment": 1,
            "metrics": 1,
        }

        cursor = (
            self._collection.find(query, projection)
            .sort("time", DESCENDING)
            .limit(limit)
        )
        return list(cursor)

    def load_test_result(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Load full test result with embedded data.

        Args:
            run_id: Unique run identifier

        Returns:
            Full test run document or None
        """
        return self._collection.find_one({"run_id": run_id})

    def delete_by_run_id(self, run_id: str) -> bool:
        """
        Delete a test run by run_id.

        Args:
            run_id: Unique run identifier

        Returns:
            True if deleted, False if not found
        """
        result = self._collection.delete_one({"run_id": run_id})
        deleted = result.deleted_count > 0
        if deleted:
            logger.debug(f"Deleted test run: {run_id}")
        return deleted

    def count(self, **filters) -> int:
        """
        Count documents matching filters.

        Returns:
            Number of matching documents
        """
        return self._collection.count_documents(filters)

    def exists(self, run_id: str) -> bool:
        """
        Check if a test run already exists.

        Args:
            run_id: Unique run identifier

        Returns:
            True if exists, False otherwise
        """
        return self._collection.count_documents({"run_id": run_id}) > 0

    def get_distinct_testcases(self) -> List[str]:
        """Get list of distinct testcase values."""
        return self._collection.distinct("testcase")

    def get_distinct_test_types(self) -> List[str]:
        """Get list of distinct test type prefixes."""
        testcases = self.get_distinct_testcases()
        types = set()
        for tc in testcases:
            if tc and "." in tc:
                types.add(tc.split(".")[0])
        return sorted(list(types))
