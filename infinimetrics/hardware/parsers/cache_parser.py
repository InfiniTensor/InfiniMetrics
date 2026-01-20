"""Parser for cache bandwidth test output."""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from infinimetrics.common.csv_utils import save_csv

from .base import BaseOutputParser

logger = logging.getLogger(__name__)


class CacheBandwidthParser(BaseOutputParser):
    """Parser for CUDA cache bandwidth test results."""

    # Pre-compiled patterns for performance
    L1_PATTERN = re.compile(
        r"L1 Cache\s*-+\s*Working Set\s+Exec Time\s+Spread\s+Eff\. Bandwidth\s*-+\s*"
        r"(?P<data>.*?)\s*(?=L2 Cache|\Z)",
        re.DOTALL,
    )

    L2_PATTERN = re.compile(
        r"L2 Cache\s*-+\s*Working Set\s+Execution Data\s+Exec Time\s+Spread\s+Eff\. Bandwidth\s*-+\s*"
        r"(?P<data>.*?)\s*(?=\Z)",
        re.DOTALL,
    )

    # CSV field definitions
    L1_FIELDS = ["data_set", "exec_time", "spread", "eff_bw"]
    L2_FIELDS = ["data_set", "exec_data", "exec_time", "spread", "eff_bw"]

    def __init__(self, output_dir: Path, metric_prefix: str = "hardware"):
        """
        Initialize cache bandwidth parser.

        Args:
            output_dir: Directory to save CSV files
            metric_prefix: Prefix for metric names
        """
        super().__init__(metric_prefix)
        self.output_dir = output_dir

    def parse(self, output: str, run_id: str) -> List[Dict[str, Any]]:
        """
        Parse cache bandwidth test output.

        Args:
            output: Raw test output
            run_id: Test run identifier

        Returns:
            List of timeseries metrics
        """
        metrics = []

        # Parse L1 cache
        l1_data = self._parse_l1_cache(output)
        if l1_data:
            metrics.append(
                self._create_cache_metric("gpu_cache_l1", l1_data, run_id, "l1")
            )

        # Parse L2 cache
        l2_data = self._parse_l2_cache(output)
        if l2_data:
            metrics.append(
                self._create_cache_metric("gpu_cache_l2", l2_data, run_id, "l2")
            )

        return metrics

    def _parse_l1_cache(self, output: str) -> List[Dict]:
        """Parse L1 cache data."""
        match = self.L1_PATTERN.search(output)
        if not match:
            return []

        csv_data = []
        for line in match.group("data").strip().split("\n"):
            parts = line.split()
            if len(parts) >= 5:
                try:
                    csv_data.append(
                        {
                            "data_set": parts[0] + " " + parts[1],
                            "_sort_key": float(parts[0]),
                            "exec_time": parts[2],
                            "spread": parts[3],
                            "eff_bw": float(parts[4].rstrip("GB/s")),
                        }
                    )
                except (ValueError, IndexError):
                    continue

        return csv_data

    def _parse_l2_cache(self, output: str) -> List[Dict]:
        """Parse L2 cache data."""
        match = self.L2_PATTERN.search(output)
        if not match:
            return []

        csv_data = []
        for line in match.group("data").strip().split("\n"):
            parts = line.split()
            if len(parts) >= 7:
                try:
                    csv_data.append(
                        {
                            "data_set": parts[0] + " " + parts[1],
                            "exec_data": parts[2] + " " + parts[3],
                            "_sort_key": float(parts[2].replace("kB", "")),
                            "exec_time": parts[4],
                            "spread": parts[5],
                            "eff_bw": float(parts[6].rstrip("GB/s")),
                        }
                    )
                except (ValueError, IndexError):
                    continue

        return csv_data

    def _create_cache_metric(
        self, name: str, data: List[Dict], run_id: str, cache_level: str
    ) -> Dict[str, Any]:
        """Create a timeseries metric with CSV file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"cache_{cache_level}_bandwidth_{run_id}_{timestamp}.csv"
        csv_path = self.output_dir / csv_filename

        fields = self.L1_FIELDS if cache_level == "l1" else self.L2_FIELDS
        save_csv(data, csv_path, fields, sort_by="data_set")

        return self.create_timeseries_metric(name, f"./{csv_filename}", unit="GB/s")
