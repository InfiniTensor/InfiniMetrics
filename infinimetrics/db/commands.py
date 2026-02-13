#!/usr/bin/env python3
"""CLI command implementations for InfiniMetrics MongoDB operations."""

import logging
import signal
import sys
from pathlib import Path
from typing import Any, Dict

from .client import MongoDBClient, MongoDBConnectionError
from .config import DatabaseConfig
from .importer import DataImporter
from .repository import TestRunRepository
from .watcher import MultiDirWatcher

logger = logging.getLogger(__name__)


def get_client_repo_importer(config: DatabaseConfig, base_dir: Path = None):
    """Get MongoDB client, repository, and importer."""
    client = MongoDBClient(config)

    if not client.health_check():
        raise MongoDBConnectionError("Cannot connect to MongoDB")

    repo = TestRunRepository(client.get_collection(config.collection_name))
    importer = DataImporter(repo, base_dir=base_dir)
    return client, repo, importer


def cmd_import(args) -> int:
    """Import test results to MongoDB."""
    config = DatabaseConfig.from_env()
    base_dir = Path(args.base_dir) if args.base_dir else Path.cwd()

    try:
        _, _, importer = get_client_repo_importer(config, base_dir)
    except MongoDBConnectionError as e:
        print(f"Error: {e}")
        return 1

    if args.summary_dir:
        output_dir = Path(args.output_dir) if args.output_dir else base_dir / "output"
        summary_dir = Path(args.summary_dir)

        print(f"Importing from:")
        print(f"  Output directory:  {output_dir}")
        print(f"  Summary directory: {summary_dir}")

        summary = importer.import_all(
            output_dir=output_dir,
            summary_dir=summary_dir,
            overwrite=args.overwrite,
        )
    else:
        directory = Path(args.directory)
        if not directory.exists():
            print(f"Error: Directory not found: {directory}")
            return 1

        print(f"Importing from: {directory}")

        summary = importer.import_directory(
            directory,
            recursive=not args.no_recursive,
            overwrite=args.overwrite,
            include_summaries=args.include_summaries,
        )

    _print_summary(summary, args.verbose)
    return 0


def cmd_watch_start(args) -> int:
    """Start the file watcher daemon."""
    config = DatabaseConfig.from_env()
    base_dir = Path(args.base_dir) if args.base_dir else Path.cwd()

    try:
        _, _, importer = get_client_repo_importer(config, base_dir)
    except MongoDBConnectionError as e:
        print(f"Error: {e}")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "output"
    summary_dir = Path(args.summary_dir) if args.summary_dir else base_dir / "summary_output"

    watcher = MultiDirWatcher(importer, output_dir=output_dir, summary_dir=summary_dir)

    def signal_handler(sig, frame):
        print("\nShutting down watcher...")
        watcher.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"Starting file watcher:")
    print(f"  Output directory:  {output_dir}")
    print(f"  Summary directory: {summary_dir}")
    print("Press Ctrl+C to stop.\n")

    watcher.run_forever()
    return 0


def cmd_watch_scan(args) -> int:
    """Perform a one-time scan without daemon mode."""
    config = DatabaseConfig.from_env()
    base_dir = Path(args.base_dir) if args.base_dir else Path.cwd()

    try:
        _, _, importer = get_client_repo_importer(config, base_dir)
    except MongoDBConnectionError as e:
        print(f"Error: {e}")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "output"
    summary_dir = Path(args.summary_dir) if args.summary_dir else base_dir / "summary_output"

    watcher = MultiDirWatcher(importer, output_dir=output_dir, summary_dir=summary_dir)

    print(f"One-time scan:")
    print(f"  Output directory:  {output_dir}")
    print(f"  Summary directory: {summary_dir}\n")

    summary = watcher.run_once()

    print(f"\nScan completed:")
    print(f"  Imported: {len(summary['imported'])}")
    print(f"  Skipped:  {len(summary['skipped'])}")
    print(f"  Failed:   {len(summary['failed'])}")

    if args.verbose and summary["imported"]:
        print("\nImported run_ids:")
        for run_id in summary["imported"]:
            print(f"  - {run_id}")

    return 0


def cmd_list(args) -> int:
    """List test runs in MongoDB."""
    config = DatabaseConfig.from_env()

    try:
        _, repo, _ = get_client_repo_importer(config)
    except MongoDBConnectionError as e:
        print(f"Error: {e}")
        return 1

    runs = repo.list_test_runs(test_type=args.test_type, limit=args.limit)

    if not runs:
        print("No test runs found.")
        return 0

    print(f"Found {len(runs)} test runs:\n")
    print(f"{'Time':<20} {'Run ID':<45} {'Testcase'}")
    print("-" * 100)

    for run in runs:
        time_str = run.get("time", "")[:19] if run.get("time") else "N/A"
        run_id = run.get("run_id", "unknown")[:43]
        testcase = run.get("testcase", "unknown")
        print(f"{time_str:<20} {run_id:<45} {testcase}")

    return 0


def cmd_info(args) -> int:
    """Show detailed info for a test run."""
    config = DatabaseConfig.from_env()

    try:
        _, repo, _ = get_client_repo_importer(config)
    except MongoDBConnectionError as e:
        print(f"Error: {e}")
        return 1

    run = repo.find_by_run_id(args.run_id)
    if not run:
        print(f"Not found: {args.run_id}")
        return 1

    _print_run_info(run)
    return 0


def cmd_delete(args) -> int:
    """Delete a test run from MongoDB."""
    config = DatabaseConfig.from_env()

    try:
        _, repo, _ = get_client_repo_importer(config)
    except MongoDBConnectionError as e:
        print(f"Error: {e}")
        return 1

    if not args.force:
        confirm = input(f"Delete run '{args.run_id}'? [y/N]: ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return 0

    if repo.delete_by_run_id(args.run_id):
        print(f"Deleted: {args.run_id}")
        return 0
    else:
        print(f"Not found: {args.run_id}")
        return 1


def _print_summary(summary: Dict[str, Any], verbose: bool = False):
    """Print import summary."""
    print(f"\nImport completed:")
    print(f"  Imported: {len(summary['imported'])}")
    print(f"  Skipped:  {len(summary['skipped'])}")
    print(f"  Failed:   {len(summary['failed'])}")

    if verbose:
        if summary["imported"]:
            print("\nImported run_ids:")
            for run_id in summary["imported"]:
                print(f"  - {run_id}")

        if summary["failed"]:
            print("\nFailed items:")
            for f in summary["failed"]:
                print(f"  - {f}")


def _print_run_info(run: Dict[str, Any]):
    """Print detailed run info."""
    print(f"Run ID:     {run.get('run_id')}")
    print(f"Time:       {run.get('time')}")
    print(f"Testcase:   {run.get('testcase')}")
    print(f"Result:     {run.get('result_code')}")

    metadata = run.get("_metadata", {})
    if metadata and metadata.get("dispatcher"):
        disp = metadata["dispatcher"]
        print(f"\nMetadata:")
        print(f"  Summary file:  {disp.get('summary_file', 'N/A')}")
        print(f"  Summary time:  {disp.get('summary_timestamp', 'N/A')}")
        print(f"  Total tests:   {disp.get('total_tests', 'N/A')}")

    env = run.get("environment", {})
    if env.get("cluster"):
        machine = env["cluster"][0].get("machine", {})
        print(f"\nMachine:")
        print(f"  CPU:       {machine.get('cpu_model', 'N/A')}")
        print(f"  Memory:    {machine.get('memory_gb', 'N/A')} GB")

        accs = machine.get("accelerators", [])
        if accs:
            print(f"  Accelerators:")
            for acc in accs:
                print(f"    - {acc.get('model', 'N/A')} ({acc.get('type', 'N/A')})")

    metrics = run.get("metrics", [])
    print(f"\nMetrics ({len(metrics)}):")
    for m in metrics[:10]:
        name = m.get("name", "unknown")
        mtype = m.get("type", "unknown")
        unit = m.get("unit", "")
        value = m.get("value", "")
        if mtype == "scalar":
            print(f"  - {name}: {value} {unit}")
        else:
            data_len = len(m.get("data", []))
            print(f"  - {name}: [{data_len} data points] {unit}")

    if len(metrics) > 10:
        print(f"  ... and {len(metrics) - 10} more metrics")
