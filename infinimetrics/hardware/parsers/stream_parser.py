"""Parser for STREAM benchmark test output."""

import re
from typing import Any, Dict, List

from .base import BaseOutputParser


class StreamBenchmarkParser(BaseOutputParser):
    """Parser for STREAM benchmark test results."""

    # STREAM operations
    OPERATIONS = ["copy", "scale", "add", "triad"]

    # Pre-compiled pattern
    STREAM_PATTERN = re.compile(r"STREAM_(?P<op>COPY|SCALE|ADD|TRIAD)\s+(?P<value>\d+\.\d+)")

    def __init__(self, metric_prefix: str = "hardware"):
        """
        Initialize STREAM parser.

        Args:
            metric_prefix: Prefix for metric names
        """
        super().__init__(metric_prefix)

    def parse(self, output: str, run_id: str = None) -> List[Dict[str, Any]]:
        """
        Parse STREAM benchmark output.

        Args:
            output: Raw test output
            run_id: Test run identifier (unused for STREAM)

        Returns:
            List of scalar metrics
        """
        metrics = []

        for op in self.OPERATIONS:
            # Match both uppercase and lowercase operation names
            match = re.search(rf"STREAM_{op.upper()}\s+(\d+\.\d+)", output)
            if match:
                value = float(match.group(1))
                metrics.append(
                    self.create_metric(
                        f"stream_{op}", value, metric_type="scalar", unit="GB/s"
                    )
                )

        return metrics
