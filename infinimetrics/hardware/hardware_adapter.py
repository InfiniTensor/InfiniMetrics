#!/usr/bin/env python3
"""Hardware Test Adapter for CUDA Unified Benchmark Suite"""

import logging
import subprocess
import re
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from infinimetrics.adapter import BaseAdapter
from .constants import (
    TEST_TYPE_MAP,
    MEMORY_DIRECTIONS,
    STREAM_OPERATIONS,
    LOG_FILES_MAP,
    MEMORY_CSV_FIELDS,
    L1_CACHE_CSV_FIELDS,
    L2_CACHE_CSV_FIELDS,
    L1_CACHE_PATTERN,
    L2_CACHE_PATTERN,
    CACHE_TEST_TIMEOUT,
    DEFAULT_TEST_TIMEOUT,
    METRIC_PREFIX_MEM_SWEEP,
    METRIC_PREFIX_MEM_BW,
)

logger = logging.getLogger(__name__)


class HardwareTestAdapter(BaseAdapter):
    """Adapter for CUDA Unified hardware performance tests."""

    def __init__(
        self,
        cuda_perf_path: str = None,
        output_dir: str = "./output",
        input_outputs_dir: str = None,
    ):
        self.cuda_perf_path = cuda_perf_path or str(
            Path(__file__).parent
            / "cuda-memory-benchmark"
            / "build"
            / "cuda_perf_suite"
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_outputs_dir = (
            Path(input_outputs_dir)
            if input_outputs_dir
            else Path(__file__).parent / "outputs"
        )
        self.build_dir = Path(__file__).parent / "cuda-memory-benchmark"
        self.build_script = self.build_dir / "build.sh"

    def setup(self, config: Dict[str, Any]) -> None:
        """Initialize resources before running tests."""
        device = config.get("device", "cuda").lower()
        if (
            device == "cpu"
            or config.get("skip_build", False)
            or Path(self.cuda_perf_path).exists()
        ):
            return
        self._build_cuda_project()

    def process(self, test_input: Any) -> Dict[str, Any]:
        """Process test input and return results."""
        if not isinstance(test_input, dict):
            test_input = (
                test_input.to_dict() if hasattr(test_input, "to_dict") else None
            )
        if not test_input:
            return self._error_response(f"Invalid test_input type: {type(test_input)}")

        testcase, config, run_id = (
            test_input.get("testcase", "unknown"),
            test_input.get("config", {}),
            test_input.get("run_id", "unknown"),
        )
        logger.info(f"HardwareTestAdapter: Processing {testcase}")

        self.output_dir = Path(config.get("output_dir", "./output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        device, test_type = config.get("device", "cuda").lower(), config.get(
            "test_type", "comprehensive"
        )

        try:
            if device == "cpu":
                logger.info("CPU mode: Reading from %s", self.input_outputs_dir)
                metrics = self._process_existing_outputs(test_type, run_id)
            else:
                logger.info("GPU mode (device=%s): Executing CUDA tests", device)
                cmd = self._build_command(config)
                output = self._execute_test(cmd, test_type)
                metrics = self._parse_output(output, test_type, run_id)

            return {
                "result_code": 0,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "run_id": run_id,
                "testcase": testcase,
                "config": config,
                "metrics": metrics,
            }
        except Exception as e:
            logger.error(f"Hardware test failed: {e}", exc_info=True)
            return self._error_response(str(e))

    def _build_cuda_project(self) -> None:
        """Build CUDA project if needed."""
        if not self.build_dir.exists():
            raise FileNotFoundError(
                f"CUDA benchmark directory not found: {self.build_dir}"
            )
        if not self.build_script.exists():
            raise FileNotFoundError(f"Build script not found: {self.build_script}")
        logger.info("Building CUDA project in: %s", self.build_dir)
        try:
            result = subprocess.run(
                ["bash", str(self.build_script)],
                cwd=str(self.build_dir),
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to build CUDA project:\n{result.stderr}")
            logger.info("CUDA project build completed successfully")
        except subprocess.TimeoutExpired:
            raise RuntimeError("CUDA project build timed out after 5 minutes")

    @staticmethod
    def _error_response(error_msg: str) -> Dict[str, Any]:
        """Generate error response."""
        return {
            "result_code": 1,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error_msg": error_msg,
            "metrics": [],
        }

    def _build_command(self, config: Dict[str, Any]) -> list[str]:
        """Build command for CUDA test suite."""
        test_type = config.get("test_type", "all")
        cuda_test_type = TEST_TYPE_MAP.get(test_type, test_type.lower())
        cmd = [self.cuda_perf_path, f"--{cuda_test_type}"]

        # Add optional parameters
        for key, param_name in [
            ("device_id", "--device"),
            ("iterations", "--iterations"),
            ("array_size", "--array-size"),
            ("buffer_size_mb", "--buffer-size"),
        ]:
            if key in config:
                cmd.extend([param_name, str(config[key])])
        return cmd

    def _execute_test(self, cmd: list[str], test_type: str) -> str:
        """Execute CUDA test and return output."""
        if not Path(self.cuda_perf_path).exists():
            raise RuntimeError(f"cuda_perf_suite not found: {self.cuda_perf_path}")
        logger.info("Executing: %s", " ".join(cmd))

        timeout = (
            CACHE_TEST_TIMEOUT if test_type.lower() == "cache" else DEFAULT_TEST_TIMEOUT
        )
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=timeout
        )
        return result.stdout

    def _parse_output(self, output: str, test_type: str, run_id: str) -> list[Dict]:
        """Parse test output based on test type."""
        if test_type == "Comprehensive":
            return (
                self._parse_memory_bandwidth(output, run_id, METRIC_PREFIX_MEM_SWEEP)
                + self._parse_memory_bandwidth(output, run_id, METRIC_PREFIX_MEM_BW)
                + self._parse_stream_benchmark(output)
                + self._parse_cache_bandwidth(output, run_id)
            )

        # Single test type
        metric_map = {
            "MemSweep": (METRIC_PREFIX_MEM_SWEEP, self._parse_memory_bandwidth),
            "MemBw": (METRIC_PREFIX_MEM_BW, self._parse_memory_bandwidth),
            "Stream": (None, self._parse_stream_benchmark),
            "Cache": (None, self._parse_cache_bandwidth),
        }

        if test_type in metric_map:
            prefix, parser = metric_map[test_type]
            return parser(output, run_id, prefix) if prefix else parser(output, run_id)
        return []

    def _save_csv(
        self,
        data: list[Dict],
        filename: str,
        fieldnames: list[str] = None,
        sort_by: str = None,
    ) -> Path:
        """Save data to CSV file with optional sorting and index column."""
        csv_path = self.output_dir / filename
        if not fieldnames:
            fieldnames = sorted(set().union(*data))

        # Add index and prepare data
        data_with_index = [{"index": idx, **row} for idx, row in enumerate(data)]

        # Sort if needed
        if sort_by:

            def sort_key(x):
                if "_sort_key" in x:
                    return float(x["_sort_key"])
                val = x.get(sort_by, "0")
                return float(str(val).replace(" kB", "").replace(" MB", ""))

            data_with_index = sorted(data_with_index, key=sort_key)

        # Remove internal fields and prepare fieldnames
        final_data = [
            {k: v for k, v in row.items() if k != "_sort_key"}
            for row in data_with_index
        ]
        fieldnames = ["index"] + [
            f for f in fieldnames if f not in ("index", "_sort_key")
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(final_data)
        logger.info("CSV saved: %s", csv_path)
        return csv_path

    def _parse_memory_bandwidth(
        self, output: str, run_id: str, metric_prefix: str
    ) -> list[Dict]:
        """Parse memory bandwidth test output."""
        metrics = []
        is_sweep = "sweep" in metric_prefix

        for direction_label, key in MEMORY_DIRECTIONS:
            csv_data = self._parse_bandwidth_data(output, direction_label)

            if not csv_data:
                continue

            metric_name = f"{metric_prefix}_{key}"

            if is_sweep:
                metrics.append(
                    self._create_timeseries_metric(
                        metric_name,
                        csv_data,
                        f"mem_sweep_{key}_{run_id}",
                        MEMORY_CSV_FIELDS,
                    )
                )
            else:
                max_bw = max(row["bandwidth_gbps"] for row in csv_data)
                metrics.append(
                    {
                        "name": metric_name,
                        "value": round(max_bw, 2),
                        "type": "scalar",
                        "unit": "GB/s",
                    }
                )

        return metrics

    def _parse_bandwidth_data(self, output: str, direction: str) -> list[Dict]:
        """Parse bandwidth data for a specific direction."""
        csv_data = []

        # Try sweep format
        sweep_pattern = rf"{direction}.*?Size \(MB\)\s+Time \(ms\)Bandwidth \(GB/s\)\s+CV \(%\)\s*-+\s*(.*?)\s*(?=={direction}|Device to Device|Cache|STREAM|\Z)"
        sweep_match = re.search(sweep_pattern, output, re.DOTALL)

        # Try bandwidth test format
        bw_pattern = rf"{direction}.*?Bandwidth Test.*?Transfer Size \(Bytes\)\s+Bandwidth\(GB/s\)\s*-+\s*(.*?)\s*(?=={direction}|Device to Device|Cache|STREAM|\Z)"
        bw_match = re.search(bw_pattern, output, re.DOTALL)

        data_block = None
        parser = None

        if sweep_match:
            data_block = sweep_match.group(1)
            parser = self._parse_sweep_line
        elif bw_match:
            data_block = bw_match.group(1)
            parser = self._parse_bw_line

        if data_block and parser:
            for line in data_block.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("-"):
                    result = parser(line)
                    if result:
                        csv_data.append(result)

        return csv_data

    @staticmethod
    def _parse_sweep_line(line: str) -> Optional[Dict]:
        """Parse a line from sweep format output."""
        parts = line.split()
        if len(parts) >= 3:
            try:
                return {"size_mb": float(parts[0]), "bandwidth_gbps": float(parts[2])}
            except (ValueError, IndexError):
                pass
        return None

    @staticmethod
    def _parse_bw_line(line: str) -> Optional[Dict]:
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

    def _create_timeseries_metric(
        self, name: str, data: list[Dict], base_filename: str, fields: list[str]
    ) -> Dict:
        """Create a timeseries metric with CSV file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{base_filename}_{timestamp}.csv"
        csv_file = self._save_csv(data, csv_filename, fields, sort_by=fields[0])
        return {
            "name": name,
            "type": "timeseries",
            "raw_data_url": f"./{csv_file.name}",
            "unit": "GB/s",
        }

    def _parse_stream_benchmark(self, output: str, run_id: str = None) -> list[Dict]:
        """Parse STREAM benchmark output."""
        metrics = []
        for op in STREAM_OPERATIONS:
            match = re.search(rf"STREAM_{op.capitalize()}\s+(\d+\.\d+)", output)
            if match:
                metrics.append(
                    {
                        "name": f"hardware.stream_{op}",
                        "value": float(match.group(1)),
                        "type": "scalar",
                        "unit": "GB/s",
                    }
                )
        return metrics

    def _parse_cache_bandwidth(self, output: str, run_id: str) -> list[Dict]:
        """Parse cache bandwidth sweep test output."""
        metrics = []

        # Parse L1
        l1_match = re.search(L1_CACHE_PATTERN, output, re.DOTALL)
        if l1_match:
            l1_data = []
            self._extract_cache_metrics(l1_match.group(1), l1_data)
            if l1_data:
                metrics.append(
                    self._create_timeseries_metric(
                        "hardware.gpu_cache_l1",
                        l1_data,
                        f"cache_l1_bandwidth_{run_id}",
                        L1_CACHE_CSV_FIELDS,
                    )
                )

        # Parse L2
        l2_match = re.search(L2_CACHE_PATTERN, output, re.DOTALL)
        if l2_match:
            l2_data = []
            self._extract_l2_metrics(l2_match.group(1), l2_data)
            if l2_data:
                metrics.append(
                    self._create_timeseries_metric(
                        "hardware.gpu_cache_l2",
                        l2_data,
                        f"cache_l2_bandwidth_{run_id}",
                        L2_CACHE_CSV_FIELDS,
                    )
                )

        return metrics

    def _extract_cache_metrics(self, text: str, csv_data: list[Dict]) -> None:
        """Extract L1 cache metrics."""
        for line in text.strip().split("\n"):
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

    def _extract_l2_metrics(self, text: str, csv_data: list[Dict]) -> None:
        """Extract L2 cache metrics."""
        for line in text.strip().split("\n"):
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

    def _process_existing_outputs(self, test_type: str, run_id: str) -> list[Dict]:
        """Process existing output log files."""
        if not self.input_outputs_dir.exists():
            raise FileNotFoundError(
                f"Input outputs directory not found: {self.input_outputs_dir}"
            )

        log_files = LOG_FILES_MAP.get(test_type, ["comprehensive_test.log"])
        all_metrics = []

        for log_file_name in log_files:
            log_path = self.input_outputs_dir / log_file_name
            if not log_path.exists():
                logger.warning("Log file not found: %s", log_path)
                continue

            logger.info("Processing log file: %s", log_path)
            with open(log_path, "r") as f:
                output = f.read()

            # Map log files to parsers
            parser_map = {
                "memory_test.log": (
                    METRIC_PREFIX_MEM_SWEEP,
                    self._parse_memory_bandwidth,
                ),
                "bandwidth_test.log": (
                    METRIC_PREFIX_MEM_BW,
                    self._parse_memory_bandwidth,
                ),
                "stream_test.log": (None, self._parse_stream_benchmark),
                "cache_test.log": (None, self._parse_cache_bandwidth),
                "comprehensive_test.log": (None, self._parse_comprehensive),
            }

            if log_file_name in parser_map:
                prefix, parser = parser_map[log_file_name]
                metrics = (
                    parser(output, run_id, prefix) if prefix else parser(output, run_id)
                )
                all_metrics.extend(metrics)

        return all_metrics

    def _parse_comprehensive(self, output: str, run_id: str) -> list[Dict]:
        """Parse comprehensive test output."""
        return (
            self._parse_memory_bandwidth(output, run_id, METRIC_PREFIX_MEM_SWEEP)
            + self._parse_memory_bandwidth(output, run_id, METRIC_PREFIX_MEM_BW)
            + self._parse_cache_bandwidth(output, run_id)
            + self._parse_stream_benchmark(output, run_id)
        )
