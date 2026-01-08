#!/usr/bin/env python3
"""Main entry point for InfiniMetrics test framework."""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any

from infinimetrics.dispatcher import Dispatcher

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_input_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load input from single file, returns list of test inputs.

    Each input file contains test specifications with fields like:
    - run_id: Test run identifier
    - testcase: Test case name (e.g., 'infer.InfiniLM.Direct')
    - config: Test configuration dict
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def load_inputs_from_paths(input_paths: List[str]) -> List[Dict[str, Any]]:
    """Load all test inputs from files and directories."""
    all_inputs = []

    for path_str in input_paths:
        path = Path(path_str)
        try:
            if path.is_file():
                inputs = load_input_file(path)
                all_inputs.extend(inputs)
                logger.info(f"Loaded {len(inputs)} input(s) from {path_str}")
            elif path.is_dir():
                for json_file in sorted(path.glob("*.json")):
                    inputs = load_input_file(json_file)
                    all_inputs.extend(inputs)
                logger.info(
                    f"Loaded {len(list(path.glob('*.json')))} file(s) from {path_str}"
                )
            else:
                logger.warning(f"Path not found: {path_str}")
        except Exception as e:
            logger.error(f"Error loading {path_str}: {e}")

    return all_inputs


def run_tests(
        inputs: List[Dict[str, Any]], default_config: Dict[str, Any]
    ) -> Dict[str, Any]:
    """Run tests using Dispatcher."""
    for test_input in inputs:
        test_input.setdefault("config", {}).setdefault(
            "output_dir", default_config.get("output_dir", "./output")
        )

    dispatcher = Dispatcher()
    return dispatcher.dispatch(inputs)


def print_summary(results: Dict[str, Any]) -> None:
    """Print test result summary."""
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    total = results["total_tests"]
    successful = results["successful_tests"]
    failed = results["failed_tests"]

    print(f"Total tests:   {total}")
    print(f"Successful:    {successful}")
    print(f"Failed:        {failed}")

    if total > 0:
        print(f"Success rate:  {(successful/total)*100:.1f}%")

    if failed > 0:
        print("\nFailed tests:")
        for r in results["results"]:
            if r["success"] == 0:
                print(f"  - {r['testcase']}: {r.get('error', 'Unknown')}")

    print("=" * 60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="InfiniMetrics - Test Orchestration Framework",
        epilog="Examples:\n  python main.py input.json\n  python main.py /path/to/inputs/\n  python main.py input.json --output ./results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input files or directories containing test specifications",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        inputs = load_inputs_from_paths(args.inputs)
        if not inputs:
            logger.error("No test inputs loaded")
            return 1

        default_config = {"output_dir": args.output}
        results = run_tests(inputs, default_config)
        print_summary(results)

        return 1 if results["failed_tests"] > 0 else 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
