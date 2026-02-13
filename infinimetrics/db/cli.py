#!/usr/bin/env python3
"""Command-line interface for InfiniMetrics MongoDB operations."""

import argparse
import logging
import sys

from . import commands

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="InfiniMetrics MongoDB CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import from both output and summary directories (recommended)
  python -m infinimetrics.db.cli import --output-dir ./output --summary-dir ./summary_output

  # Import from a single directory
  python -m infinimetrics.db.cli import ./output

  # Start file watcher daemon
  python -m infinimetrics.db.cli watch start

  # One-time scan
  python -m infinimetrics.db.cli watch scan

  # List all test runs
  python -m infinimetrics.db.cli list

  # Show detailed info
  python -m infinimetrics.db.cli info <run_id>
        """,
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_import_parser(subparsers)
    _add_watch_parser(subparsers)
    _add_list_parser(subparsers)
    _add_info_parser(subparsers)
    _add_delete_parser(subparsers)

    return parser


def _add_import_parser(subparsers):
    """Add import subcommand parser."""
    p = subparsers.add_parser("import", help="Import test result files to MongoDB")
    p.add_argument("directory", nargs="?", default="./output", help="Directory to import")
    p.add_argument("--output-dir", help="Output directory containing test result files")
    p.add_argument("--summary-dir", help="Summary directory containing dispatcher summaries")
    p.add_argument("--base-dir", help="Base directory for resolving relative paths")
    p.add_argument("--no-recursive", action="store_true", help="Don't recurse into subdirs")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing documents")
    p.add_argument("--include-summaries", action="store_true", help="Process dispatcher summaries")
    p.set_defaults(func=commands.cmd_import)


def _add_watch_parser(subparsers):
    """Add watch subcommand parser."""
    watch = subparsers.add_parser("watch", help="File watcher commands")
    watch_sub = watch.add_subparsers(dest="watch_command", required=True)

    start = watch_sub.add_parser("start", help="Start watcher daemon")
    start.add_argument("--output-dir", help="Output directory to watch")
    start.add_argument("--summary-dir", help="Summary directory to watch")
    start.add_argument("--base-dir", help="Base directory for resolving paths")
    start.set_defaults(func=commands.cmd_watch_start)

    scan = watch_sub.add_parser("scan", help="One-time scan")
    scan.add_argument("--output-dir", help="Output directory to scan")
    scan.add_argument("--summary-dir", help="Summary directory to scan")
    scan.add_argument("--base-dir", help="Base directory for resolving paths")
    scan.set_defaults(func=commands.cmd_watch_scan)


def _add_list_parser(subparsers):
    """Add list subcommand parser."""
    p = subparsers.add_parser("list", help="List test runs")
    p.add_argument("--test-type", help="Filter by test type prefix")
    p.add_argument("--limit", type=int, default=50, help="Max results")
    p.set_defaults(func=commands.cmd_list)


def _add_info_parser(subparsers):
    """Add info subcommand parser."""
    p = subparsers.add_parser("info", help="Show detailed info for a run")
    p.add_argument("run_id", help="Run ID to show")
    p.set_defaults(func=commands.cmd_info)


def _add_delete_parser(subparsers):
    """Add delete subcommand parser."""
    p = subparsers.add_parser("delete", help="Delete a test run")
    p.add_argument("run_id", help="Run ID to delete")
    p.add_argument("--force", "-f", action="store_true", help="Don't ask for confirmation")
    p.set_defaults(func=commands.cmd_delete)


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)

    try:
        return args.func(args)
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
