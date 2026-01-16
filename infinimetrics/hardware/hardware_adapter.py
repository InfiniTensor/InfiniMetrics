#!/usr/bin/env python3
"""Hardware Test Adapter for CUDA Unified Benchmark Suite"""

import logging
import subprocess
import re
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from infinimetrics.adapter import BaseAdapter

logger = logging.getLogger(__name__)


class HardwareTestAdapter(BaseAdapter):
    """Adapter for CUDA Unified hardware performance tests."""

    def __init__(
        self,
        cuda_perf_path: str = None,
        output_dir: str = "./output",
        mock_mode: bool = False,
        mock_output_file: str = None
    ):
        """
        Initialize HardwareTestAdapter.

        Args:
            cuda_perf_path: Path to cuda_perf_suite executable.
            output_dir: Directory to save CSV files with detailed results
            mock_mode: If True, use mock output instead of running real tests
            mock_output_file: Path to file containing real cuda_perf_suite output for mock mode
        """
        if cuda_perf_path is None:
            default_path = Path(__file__).parent / "cuda-memory-benchmark" / "build" / "cuda_perf_suite"
            self.cuda_perf_path = str(default_path)
        else:
            self.cuda_perf_path = cuda_perf_path

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mock_mode = mock_mode
        self.mock_output_file = mock_output_file

    def process(self, test_input: Any) -> Dict[str, Any]:
        """Execute the hardware test and return results with CSV file path."""
        # Convert to dict
        if hasattr(test_input, "to_dict"):
            test_input = test_input.to_dict()
        elif not isinstance(test_input, dict):
            return {
                "result_code": 1,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error_msg": f"Invalid test_input type: {type(test_input)}",
                "metrics": [],
            }

        testcase = test_input.get("testcase", "unknown")
        config = test_input.get("config", {})
        run_id = test_input.get("run_id", "unknown")

        logger.info(f"HardwareTestAdapter: Processing {testcase}")

        # Update output_dir
        self.output_dir = Path(config.get("output_dir", "./output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Mock mode for testing adapter logic
            if self.mock_mode:
                output = self._get_mock_output(config.get("test_type"))
                metrics, csv_file = self._parse_output(output, testcase, run_id)
            else:
                cmd = self._build_command(config)
                output = self._execute_test(cmd, config.get("test_type"))
                metrics, csv_file = self._parse_output(output, testcase, run_id)

            result = {
                "result_code": 0,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metrics": metrics,
            }

            if csv_file:
                result["csv_file"] = str(csv_file)

            if self.mock_mode:
                result["mock_mode"] = True

            return result

        except Exception as e:
            logger.error(f"Hardware test failed: {e}")
            return {
                "result_code": 1,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error_msg": str(e),
                "metrics": [],
            }

    def _build_command(self, config: Dict[str, Any]) -> list[str]:
        """Build cuda_perf_suite command from config."""
        cmd = [self.cuda_perf_path, f"--{config.get('test_type', 'all')}"]
        cmd.extend(["--device", str(config.get("device_id", 0))])
        cmd.extend(["--iterations", str(config.get("iterations", 10))])

        if "array_size" in config:
            cmd.extend(["--array-size", str(config["array_size"])])
        if "buffer_size_mb" in config:
            cmd.extend(["--buffer-size", str(config["buffer_size_mb"])])

        return cmd

    def _execute_test(self, cmd: list[str], test_type: str) -> str:
        """Execute cuda_perf_suite and return stdout."""
        if not Path(self.cuda_perf_path).exists():
            raise RuntimeError(
                f"cuda_perf_suite not found at: {self.cuda_perf_path}\n"
                f"Build it first: cd infinimetrics/hardware/cuda-memory-benchmark && ./build.sh"
            )

        logger.info(f"Executing: {' '.join(cmd)}")

        # Cache tests take longer, use 30 min timeout
        timeout = 1800 if test_type == "cache" else 600

        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=timeout
        )
        return result.stdout

    def _get_mock_output(self, test_type: str) -> str:
        """Load real cuda_perf_suite output from file for testing."""
        if self.mock_output_file and Path(self.mock_output_file).exists():
            logger.info(f"Using mock output from: {self.mock_output_file}")
            with open(self.mock_output_file, "r") as f:
                return f.read()

        # Default: use output.log in the same directory
        default_mock_file = Path(__file__).parent / "cuda-memory-benchmark" / "output.log"
        if default_mock_file.exists():
            logger.info(f"Using mock output from: {default_mock_file}")
            with open(default_mock_file, "r") as f:
                return f.read()

        raise RuntimeError(
            f"Mock output file not found. Please provide mock_output_file or place output.log at {default_mock_file}"
        )

    def _parse_output(
        self, output: str, testcase: str, run_id: str
    ) -> tuple[list[Dict], Path | None]:
        """Parse output and extract metrics with CSV export."""
        metrics = []
        csv_file = None

        # Memory bandwidth
        if any(k in testcase for k in ["memory.bandwidth", "MemoryBandwidth", "comprehensive"]):
            mem_metrics, mem_csv = self._parse_memory_bandwidth(output, run_id)
            metrics.extend(mem_metrics)
            csv_file = mem_csv or csv_file

        # STREAM
        if any(k in testcase for k in ["stream", "STREAM", "comprehensive"]):
            metrics.extend(self._parse_stream_benchmark(output))

        # Cache
        if any(k in testcase for k in ["cache", "Cache", "comprehensive"]):
            cache_metrics, cache_csv = self._parse_cache_bandwidth(output, run_id)
            metrics.extend(cache_metrics)
            csv_file = cache_csv or csv_file

        return metrics, csv_file

    def _save_csv(self, data: list[Dict], filename: str, fieldnames: list[str] = None) -> Path:
        """Save data to CSV file and return path."""
        csv_path = self.output_dir / filename

        if not fieldnames:
            fieldnames = sorted(set().union(*data))

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        logger.info(f"CSV saved: {csv_path}")
        return csv_path

    def _parse_bandwidth_table(
        self, output: str, pattern: str, extractor: callable
    ) -> tuple[list[Dict], list[Dict]]:
        """Generic bandwidth table parser.

        Args:
            output: Raw output text
            pattern: Regex to find the table
            extractor: Function to extract data from table line

        Returns:
            (peak_metrics, csv_data)
        """
        match = re.search(pattern, output, re.DOTALL)
        if not match:
            return [], []

        csv_data = []
        lines = match.group(1).strip().split("\n")

        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    csv_data.append(extractor(parts))
                except (ValueError, IndexError):
                    continue

        return csv_data

    def _parse_memory_bandwidth(
        self, output: str, run_id: str
    ) -> tuple[list[Dict], Path | None]:
        """Parse memory bandwidth and save to CSV."""
        metrics = []
        csv_data = []

        directions = {
            "Host to Device": "host_to_device",
            "Device to Host": "device_to_host",
            "Device to Device": "device_to_device"
        }

        for direction, key in directions.items():
            # Extract peak bandwidth (value at 1024MB)
            peak_match = re.search(
                rf"{direction}.*?1024\.00.*?(\d+\.\d+)\s+GB/s", output, re.DOTALL
            )
            if peak_match:
                metrics.append({
                    "name": f"memory.bandwidth.{key}",
                    "value": float(peak_match.group(1)),
                    "type": "scalar",
                    "unit": "GB/s",
                })

            # Parse table data
            pattern = rf"Direction:\s+{direction}.*?Size\(MB\).*?={10,}"
            table_match = re.search(pattern, output, re.DOTALL)

            if table_match:
                for line in table_match.group(0).split("\n"):
                    parts = line.split()
                    if len(parts) >= 3 and parts[0].replace(".", "").isdigit():
                        try:
                            csv_data.append({
                                "direction": key,
                                "size_mb": float(parts[0]),
                                "bandwidth_gbps": float(parts[2])
                            })
                        except (ValueError, IndexError):
                            continue

        if csv_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = self._save_csv(
                csv_data,
                f"memory_bandwidth_{run_id}_{timestamp}.csv",
                ["direction", "size_mb", "bandwidth_gbps"]
            )
            return metrics, csv_file

        return metrics, None

    def _parse_stream_benchmark(self, output: str) -> list[Dict]:
        """Parse STREAM benchmark results."""
        metrics = []
        operations = ["copy", "scale", "add", "triad"]

        for op in operations:
            match = re.search(rf"STREAM_{op.capitalize()}\s+(\d+\.\d+)\s+GB/s", output)
            if match:
                metrics.append({
                    "name": f"stream.bandwidth.{op}",
                    "value": float(match.group(1)),
                    "type": "scalar",
                    "unit": "GB/s",
                })

        return metrics

    def _parse_cache_bandwidth(
        self, output: str, run_id: str
    ) -> tuple[list[Dict], Path | None]:
        """Parse cache bandwidth and save to CSV."""
        metrics = []
        csv_data = []

        # Parse L1 cache
        l1_match = re.search(r"L1 Cache Bandwidth.*?data set.*?Eff\. bw(.*?)={10,}L2", output, re.DOTALL)
        if l1_match:
            max_l1 = 0
            for line in l1_match.group(1).strip().split("\n"):
                parts = line.split()
                if len(parts) >= 4 and parts[0].replace(".", "").isdigit():
                    try:
                        csv_data.append({
                            "cache_level": "l1",
                            "size_kb": float(parts[0]),
                            "exec_time_ms": float(parts[1].rstrip("ms")),
                            "spread_pct": float(parts[2].rstrip("%")),
                            "bandwidth_gbps": float(parts[3])
                        })
                        max_l1 = max(max_l1, float(parts[3]))
                    except (ValueError, IndexError):
                        continue

            if max_l1 > 0:
                metrics.append({
                    "name": "cache.bandwidth.l1",
                    "value": round(max_l1, 2),
                    "type": "scalar",
                    "unit": "GB/s",
                })

        # Parse L2 cache
        l2_match = re.search(r"L2 Cache Bandwidth.*?data set.*?Eff\. bw(.*?)={10,}BANDWIDTH", output, re.DOTALL)
        valid_l2 = []

        if l2_match:
            for line in l2_match.group(1).strip().split("\n"):
                parts = line.split()
                if len(parts) >= 5 and parts[0].replace(".", "").isdigit():
                    try:
                        size_kb = int(float(parts[0]))
                        bandwidth = float(parts[4])

                        csv_data.append({
                            "cache_level": "l2",
                            "size_kb": size_kb,
                            "exec_data_kb": float(parts[1]),
                            "exec_time_ms": float(parts[2]),
                            "spread_pct": float(parts[3].rstrip("%")),
                            "bandwidth_gbps": bandwidth
                        })

                        if 512 <= size_kb <= 4096:
                            valid_l2.append(bandwidth)
                    except (ValueError, IndexError):
                        continue

            if valid_l2:
                metrics.append({
                    "name": "cache.bandwidth.l2",
                    "value": round(sum(valid_l2) / len(valid_l2), 2),
                    "type": "scalar",
                    "unit": "GB/s",
                })

        if csv_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = self._save_csv(
                csv_data,
                f"cache_bandwidth_{run_id}_{timestamp}.csv"
            )
            return metrics, csv_file

        return metrics, None
