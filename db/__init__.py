#!/usr/bin/env python3
"""
InfiniMetrics MongoDB integration module.

This module provides:
- MongoDB connection management
- Test result repository
- Dispatcher summary repository

Usage:
    from db import MongoDBClient, DatabaseConfig, TestRunRepository

    # Connect to MongoDB
    client = MongoDBClient()
    db = client.get_database()

    # Get repositories
    test_runs = TestRunRepository(client.get_collection("test_runs"))
    summaries = DispatcherSummaryRepository(client.get_collection("dispatcher_summaries"))

    # Query test runs
    runs = test_runs.find_all(limit=10)
"""

from .client import MongoDBClient, MongoDBConnectionError
from .config import DatabaseConfig
from .repository import DispatcherSummaryRepository, TestRunRepository

__all__ = [
    "DatabaseConfig",
    "MongoDBClient",
    "MongoDBConnectionError",
    "TestRunRepository",
    "DispatcherSummaryRepository",
]
