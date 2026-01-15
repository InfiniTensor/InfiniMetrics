#!/usr/bin/env python3
"""Test script for HardwareTestAdapter"""

import json
import logging
from pathlib import Path

from infinimetrics.dispatcher import Dispatcher
from infinimetrics.input import TestInput

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def test_hardware_adapter():
    """Test hardware adapter with format_input_*.json files."""

    # Find all hardware test input files
    test_files = [
        "format_input_memory_bandwidth.json",
        "format_input_stream_benchmark.json",
        "format_input_cache_benchmark.json",
        "format_input_comprehensive_hardware.json",
    ]

    dispatcher = Dispatcher()

    for test_file in test_files:
        file_path = Path(test_file)
        if not file_path.exists():
            logger.warning(f"Test file not found: {test_file}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Testing with: {test_file}")
        logger.info(f"{'='*60}")

        # Load test input
        with open(file_path, "r") as f:
            test_input = json.load(f)

        try:
            # Process test
            result = dispatcher.dispatch(test_input)

            # Print results
            logger.info(f"\nResults for {test_file}:")
            logger.info(f"Result code: {result.get('result_code')}")
            logger.info(f"Time: {result.get('time')}")

            if result.get("error_msg"):
                logger.error(f"Error: {result['error_msg']}")

            # Print CSV file path if available
            if result.get("csv_file"):
                logger.info(f"Detailed CSV results: {result['csv_file']}")

            metrics = result.get("metrics", [])
            if metrics:
                logger.info(f"\nMetrics ({len(metrics)}):")
                for metric in metrics:
                    if isinstance(metric, dict):
                        name = metric.get("name", "unknown")
                        value = metric.get("value", "N/A")
                        unit = metric.get("unit", "")
                        logger.info(f"  {name}: {value} {unit}")

            # Save results to file
            output_file = f"output_{test_file}"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"\nResults saved to: {output_file}")

        except Exception as e:
            logger.error(f"Failed to process {test_file}: {e}", exc_info=True)


if __name__ == "__main__":
    test_hardware_adapter()
