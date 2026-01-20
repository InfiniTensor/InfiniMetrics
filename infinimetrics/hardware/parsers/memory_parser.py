"""Parser for memory bandwidth test output."""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from infinimetrics.common.csv_utils import save_csv

from .base import BaseOutputParser

logger = logging.getLogger(__name__)


class MemoryBandwidthParser(BaseOutputParser):
    """Parser for CUDA memory bandwidth test results."""

    # Pre-compile regex patterns for performance
    SWEEP_PATTERN = re.compile(
        r"(?P<direction>Host to Device|Device to Host|Device to Device).*?"
        r"Size \(MB\)\s+Time \(ms\)Bandwidth \(GB/s\)\s+CV \(%\)\s*-+\s*"
        r"(?P<data>.*?)\s*(?==(?:Host to Device|Device to Host|Device to Device)|Cache|STREAM|\Z)",
        re.DOTALL,
    )

    BW_PATTERN = re.compile(
        r"(?P<direction>Host to Device|Device to Host|Device to Device).*?"
        r"Bandwidth Test.*?Transfer Size \(Bytes\)\s+Bandwidth\(GB/s\)\s*-+\s*"
        r"(?P<data>.*?)\s*(?==(?:Host to Device|Device to Host|Device to Device)|Cache|STREAM|\Z)",
        re.DOTALL,
    )

    # Memory direction mapping
    DIRECTION_MAP = {
        "Host to Device": "h2d",
        "Device to Host": "d2h",
        "Device to Device": "d2d",
    }

    def __init__(
        self,
        output_dir: Path,
        metric_prefix: str = "hardware",
        is_sweep: bool = True,
    ):
        """
        Initialize memory bandwidth parser.

        Args:
            output_dir: Directory to save CSV files
            metric_prefix: Prefix for metric names
            is_sweep: True for sweep tests, False for bandwidth tests
        """
        super().__init__(metric_prefix)
        self.output_dir = output_dir
        self.is_sweep = is_sweep

    def parse(self, output: str, run_id: str) -> List[Dict[str, Any]]:
        """
        Parse memory bandwidth test output.

        Args:
            output: Raw test output
            run_id: Test run identifier

        Returns:
            List of metrics (scalar or timeseries)
        """
        metrics = []

        for direction_label, direction_key in self.DIRECTION_MAP.items():
            csv_data = self._parse_direction_data(output, direction_label)

            if not csv_data:
                continue

            metric_name = f"mem_{self._get_test_type()}_{direction_key}"

            if self.is_sweep:
                metrics.append(
                    self._create_timeseries_from_data(
                        metric_name, csv_data, run_id, direction_key
                    )
                )
            else:
                # For non-sweep tests, return max bandwidth as scalar
                max_bw = max(row["bandwidth_gbps"] for row in csv_data)
                metrics.append(
                    self.create_metric(
                        metric_name,
                        round(max_bw, 2),
                        metric_type="scalar",
                        unit="GB/s",
                    )
                )

        return metrics

    def _get_test_type(self) -> str:
        """Get test type identifier for metric naming."""
        return "sweep" if self.is_sweep else "bw"

    def _parse_direction_data(self, output: str, direction: str) -> List[Dict]:
        """Parse bandwidth data for a specific transfer direction."""
        csv_data = []

        # Try sweep format first
        sweep_match = None
        for match in self.SWEEP_PATTERN.finditer(output):
            if match.group("direction") == direction:
                sweep_match = match
                break

        # Try bandwidth test format
        bw_match = None
        if not sweep_match:
            for match in self.BW_PATTERN.finditer(output):
                if match.group("direction") == direction:
                    bw_match = match
                    break

        # Parse data block
        data_block = None
        parser = None

        if sweep_match:
            data_block = sweep_match.group("data")
            parser = self._parse_sweep_line
        elif bw_match:
            data_block = bw_match.group("data")
            parser = self._parse_bw_line

        if data_block and parser:
            for line in data_block.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("-"):
                    result = parser(line)
                    if result:
                        csv_data.append(result)

        return csv_data

    def _parse_sweep_line(self, line: str) -> Optional[Dict]:
        """Parse a line from sweep format output."""
        parts = line.split()
        if len(parts) >= 3:
            try:
                return {"size_mb": float(parts[0]), "bandwidth_gbps": float(parts[2])}
            except (ValueError, IndexError):
                pass
        return None

    def _parse_bw_line(self, line: str) -> Optional[Dict]:
        """Parse a line from bandwidth test format output."""
        parts = line.split()
        if len(parts) >= 2:
            try:
                size_bytes = float(parts[0])
                return {
                    "size_mb": size_bytes / (1024 * 1024),
                    "bandwidth_gbps": float(parts[1]),
                }
            except (ValueError, IndexError):
                pass
        return None

    def _create_timeseries_from_data(
        self,
        name: str,
        data: List[Dict],
        run_id: str,
        direction: str,
    ) -> Dict[str, Any]:
        """Create a timeseries metric with CSV file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"mem_{self._get_test_type()}_{direction}_{run_id}_{timestamp}.csv"
        csv_path = self.output_dir / csv_filename

        fieldnames = ["size_mb", "bandwidth_gbps"]
        save_csv(data, csv_path, fieldnames, sort_by=fieldnames[0])

        return self.create_timeseries_metric(name, f"./{csv_filename}", unit="GB/s")
