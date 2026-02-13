#!/usr/bin/env python3
"""
InfiniMetrics MongoDB integration module.

This module provides:
- MongoDB connection management
- Test result repository
- Data import from JSON/CSV files
- File system watcher for auto-import

Usage:
    from infinimetrics.db import MongoDBClient, TestRunRepository, DataImporter, FileWatcher

    # Connect to MongoDB
    client = MongoDBClient()
    repo = TestRunRepository(client.get_collection())

    # Import data
    importer = DataImporter(repo)
    importer.import_directory(Path("./output"))

    # Watch for new files
    watcher = FileWatcher(Path("./output"), importer)
    watcher.start()
"""

from .client import MongoDBClient, MongoDBConnectionError
from .config import DatabaseConfig
from .importer import DataImporter
from .repository import TestRunRepository
from .watcher import FileWatcher, MultiDirWatcher

__all__ = [
    "DatabaseConfig",
    "MongoDBClient",
    "MongoDBConnectionError",
    "TestRunRepository",
    "DataImporter",
    "FileWatcher",
    "MultiDirWatcher",
]
