#!/usr/bin/env python3
"""Hardware Test Adapter for CUDA Unified Benchmark Suite"""

import logging
import subprocess
import re
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from infinimetrics.adapter import BaseAdapter

logger = logging.getLogger(__name__)


class HardwareTestAdapter(BaseAdapter):
    """Adapter for CUDA Unified hardware performance tests."""

    def __init__(
        self,
        cuda_perf_path: str = None,
        output_dir: str = "./output",
        mock_mode: bool = False,
        mock_output_file: str = None,
        input_outputs_dir: str = None
    ):
        """
        Initialize HardwareTestAdapter.

        Args:
            cuda_perf_path: Path to cuda_perf_suite executable.
            output_dir: Directory to save converted results (CSV/JSON)
            mock_mode: If True, use mock output instead of running real tests
            mock_output_file: Path to file containing real cuda_perf_suite output for mock mode
            input_outputs_dir: Directory containing cuda_perf_suite output log files to parse
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

        # Set input outputs directory (where cuda_perf_suite logs are stored)
        if input_outputs_dir is None:
            self.input_outputs_dir = Path(__file__).parent / "outputs"
        else:
            self.input_outputs_dir = Path(input_outputs_dir)

        # Store build directory path
        self.build_dir = Path(__file__).parent / "cuda-memory-benchmark"
        self.build_script = self.build_dir / "build.sh"

    def setup(self, config: Dict[str, Any]) -> None:
        """
        Initialize resources before running tests.

        This will check the device type and decide whether to build CUDA project:

        1. If device is "cpu" or "mock": Skip CUDA build
        2. If device is "cuda" or "gpu": Check and build CUDA project if needed

        Args:
            config: Configuration dict containing 'device' and 'skip_build' flags
        """
        # Get device type from config
        device = config.get("device", "cuda").lower()

        # Skip CUDA build for non-GPU devices
        if device in ["cpu", "mock"]:
            logger.info(f"Device is '{device}': Skipping CUDA project build")
            return

        # Skip setup in mock mode
        if self.mock_mode:
            logger.info("Mock mode: Skipping CUDA project build")
            return

        # Check if we should skip building
        skip_build = config.get("skip_build", False)
        if skip_build:
            logger.info("Skipping CUDA project build (skip_build=True)")
            return

        # Check if executable already exists
        if Path(self.cuda_perf_path).exists():
            logger.info(f"cuda_perf_suite already exists at: {self.cuda_perf_path}")
            return

        # Build the CUDA project
        logger.info("cuda_perf_suite not found. Building CUDA project...")
        self._build_cuda_project()

    def _build_cuda_project(self) -> None:
        """
        Build the CUDA memory benchmark project.

        This will:
        1. Navigate to the cuda-memory-benchmark directory
        2. Run the build.sh script
        3. Verify the executable was created
        """
        if not self.build_dir.exists():
            raise FileNotFoundError(
                f"CUDA benchmark directory not found: {self.build_dir}\n"
                f"Please ensure the project structure is correct."
            )

        if not self.build_script.exists():
            raise FileNotFoundError(
                f"Build script not found: {self.build_script}\n"
                f"Please ensure build.sh exists in {self.build_dir}"
            )

        logger.info(f"Building CUDA project in: {self.build_dir}")

        try:
            # Run build script
            result = subprocess.run(
                ["bash", str(self.build_script)],
                cwd=str(self.build_dir),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                error_msg = (
                    f"Failed to build CUDA project:\n"
                    f"STDOUT: {result.stdout}\n"
                    f"STDERR: {result.stderr}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            logger.info("CUDA project build completed successfully")
            logger.debug(f"Build output:\n{result.stdout}")

            # Verify executable was created
            if not Path(self.cuda_perf_path).exists():
                raise RuntimeError(
                    f"Build completed but executable not found at: {self.cuda_perf_path}\n"
                    f"Please check the build configuration."
                )

            logger.info(f"cuda_perf_suite successfully built at: {self.cuda_perf_path}")

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"CUDA project build timed out after 5 minutes.\n"
                f"You can manually build it later with: cd {self.build_dir} && ./build.sh"
            )
        except Exception as e:
            raise RuntimeError(
                f"Error building CUDA project: {e}\n"
                f"You can manually build it later with: cd {self.build_dir} && ./build.sh"
            )

    def process(self, test_input: Any) -> Dict[str, Any]:
        """Execute the hardware test and return results with converted file paths."""
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

        # Get device type and test type
        device = config.get("device", "cuda").lower()
        test_type = config.get("test_type", "comprehensive")

        try:
            # Decide execution mode based on device
            if device in ["cpu", "mock"] or self.mock_mode:
                # Mock mode: read from existing outputs directory
                logger.info(f"Mock mode enabled (device={device}, mock_mode={self.mock_mode})")
                logger.info(f"Reading from existing outputs directory: {self.input_outputs_dir}")
                metrics, converted_files = self._process_existing_outputs(test_type, testcase, run_id)
            else:
                # Normal mode: execute real CUDA tests
                logger.info(f"Normal mode enabled (device={device}): Executing CUDA tests")
                cmd = self._build_command(config)
                output = self._execute_test(cmd, test_type)
                metrics, csv_file = self._parse_output(output, testcase, run_id)
                converted_files = {"csv": str(csv_file)} if csv_file else {}

            result = {
                "result_code": 0,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metrics": metrics,
                "converted_files": converted_files,
            }

            if self.mock_mode:
                result["mock_mode"] = True

            return result

        except Exception as e:
            logger.error(f"Hardware test failed: {e}", exc_info=True)
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
            # Look for the bandwidth test section
            # Pattern: "Device to Device Bandwidth Test" or similar
            section_pattern = rf"{direction}.*?Bandwidth Test.*?Transfer Size \(Bytes\)\s+Bandwidth\(GB/s\)\s*-+\s+(\d+)\s+(\d+\.\d+)"
            section_match = re.search(section_pattern, output, re.DOTALL)

            if section_match:
                size_bytes = float(section_match.group(1))
                bandwidth = float(section_match.group(2))

                metrics.append({
                    "name": f"memory.bandwidth.{key}",
                    "value": bandwidth,
                    "type": "scalar",
                    "unit": "GB/s",
                })

                csv_data.append({
                    "direction": key,
                    "size_bytes": size_bytes,
                    "bandwidth_gbps": bandwidth
                })
            else:
                # Try alternative pattern for older format
                alt_pattern = rf"{direction}.*?Bandwidth.*?(\d+)\s+(\d+\.\d+)"
                alt_match = re.search(alt_pattern, output, re.DOTALL)
                if alt_match:
                    size_bytes = float(alt_match.group(1))
                    bandwidth = float(alt_match.group(2))

                    metrics.append({
                        "name": f"memory.bandwidth.{key}",
                        "value": bandwidth,
                        "type": "scalar",
                        "unit": "GB/s",
                    })

                    csv_data.append({
                        "direction": key,
                        "size_bytes": size_bytes,
                        "bandwidth_gbps": bandwidth
                    })

        if csv_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = self._save_csv(
                csv_data,
                f"memory_bandwidth_{run_id}_{timestamp}.csv",
                ["direction", "size_bytes", "bandwidth_gbps"]
            )
            return metrics, csv_file

        return metrics, None

    def _parse_stream_benchmark(self, output: str) -> list[Dict]:
        """Parse STREAM benchmark results."""
        metrics = []
        operations = ["copy", "scale", "add", "triad"]

        for op in operations:
            # Match STREAM_Operation pattern and extract bandwidth (first number after operation name)
            match = re.search(rf"STREAM_{op.capitalize()}\s+(\d+\.\d+)", output)
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
        l1_pattern = r"L1 Cache Bandwidth Sweep Test.*?data set\s+exec time\s+spread\s+Eff\. bw\s*-+(.*?)(?=={10,}|L2 Cache)"
        l1_match = re.search(l1_pattern, output, re.DOTALL)
        if l1_match:
            max_l1 = 0
            for line in l1_match.group(1).strip().split("\n"):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        size_str = parts[0]
                        if size_str.replace('.', '').replace('k', '').replace('B', '').isdigit():
                            bandwidth = float(parts[3])
                            csv_data.append({
                                "cache_level": "l1",
                                "size_kb": size_str,
                                "exec_time_ms": float(parts[1].rstrip("ms")),
                                "spread_pct": float(parts[2].rstrip("%")),
                                "bandwidth_gbps": bandwidth
                            })
                            max_l1 = max(max_l1, bandwidth)
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
        l2_pattern = r"L2 Cache Bandwidth Sweep Test.*?data set\s+exec data\s+exec time\s+spread\s+Eff\. bw\s*-+(.*?)(?=={10,}|$)"
        l2_match = re.search(l2_pattern, output, re.DOTALL)
        valid_l2 = []

        if l2_match:
            for line in l2_match.group(1).strip().split("\n"):
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        size_str = parts[0]
                        # Extract numeric part from size string (e.g., "512" from "512 kB")
                        size_match = re.search(r'(\d+)', size_str)
                        if size_match:
                            size_kb = int(size_match.group(1))
                            bandwidth = float(parts[4])

                            csv_data.append({
                                "cache_level": "l2",
                                "size_kb": size_str,
                                "exec_data_kb": float(parts[1].replace('kB', '').replace('kB', '')),
                                "exec_time_ms": float(parts[2].rstrip("ms")),
                                "spread_pct": float(parts[3].rstrip("%")),
                                "bandwidth_gbps": bandwidth
                            })

                            # Filter L2 cache range (512KB - 4MB typically in L2)
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

    def _process_existing_outputs(
        self, test_type: str, testcase: str, run_id: str
    ) -> Tuple[list[Dict], Dict[str, str]]:
        """
        Process existing output files from the outputs directory.

        Args:
            test_type: Type of test (bandwidth, stream, cache, comprehensive)
            testcase: Test case name
            run_id: Run ID for file naming

        Returns:
            Tuple of (metrics list, converted_files dict)
        """
        if not self.input_outputs_dir.exists():
            raise FileNotFoundError(f"Input outputs directory not found: {self.input_outputs_dir}")

        # Determine which log files to read based on test type
        log_files = self._get_log_files_for_test_type(test_type)
        all_metrics = []
        converted_files = {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for log_file_name in log_files:
            log_path = self.input_outputs_dir / log_file_name
            if not log_path.exists():
                logger.warning(f"Log file not found: {log_path}")
                continue

            logger.info(f"Processing log file: {log_path}")

            # Read the log file content
            with open(log_path, 'r') as f:
                output = f.read()

            # Parse based on log file type
            if "bandwidth" in log_file_name:
                metrics, csv_file = self._parse_memory_bandwidth(output, run_id)
            elif "stream" in log_file_name:
                metrics = self._parse_stream_benchmark(output)
                csv_file = self._save_stream_to_csv(metrics, f"stream_{run_id}_{timestamp}.csv")
            elif "cache" in log_file_name:
                metrics, csv_file = self._parse_cache_bandwidth(output, run_id)
            elif "comprehensive" in log_file_name:
                # Parse all types from comprehensive test
                metrics_list = []
                mem_metrics, mem_csv = self._parse_memory_bandwidth(output, run_id)
                metrics_list.extend(mem_metrics)
                if mem_csv:
                    converted_files["bandwidth_csv"] = str(mem_csv)

                stream_metrics = self._parse_stream_benchmark(output)
                metrics_list.extend(stream_metrics)
                stream_csv = self._save_stream_to_csv(stream_metrics, f"stream_{run_id}_{timestamp}.csv")
                if stream_csv:
                    converted_files["stream_csv"] = str(stream_csv)

                cache_metrics, cache_csv = self._parse_cache_bandwidth(output, run_id)
                metrics_list.extend(cache_metrics)
                if cache_csv:
                    converted_files["cache_csv"] = str(cache_csv)

                metrics = metrics_list
            else:
                logger.warning(f"Unknown log file type: {log_file_name}")
                continue

            all_metrics.extend(metrics)

            # Copy original log file to output directory
            log_copy_path = self.output_dir / f"{log_path.stem}_{run_id}_{timestamp}.log"
            shutil.copy2(log_path, log_copy_path)
            converted_files[f"{log_path.stem}_log"] = str(log_copy_path)
            logger.info(f"Copied log file to: {log_copy_path}")

        # Save all metrics to JSON
        json_file = self._save_metrics_to_json(all_metrics, testcase, run_id, timestamp)
        converted_files["metrics_json"] = str(json_file)

        return all_metrics, converted_files

    def _get_log_files_for_test_type(self, test_type: str) -> list[str]:
        """Get list of log files to read based on test type."""
        log_files_map = {
            "bandwidth": ["bandwidth_test.log"],
            "stream": ["stream_test.log"],
            "cache": ["cache_test.log"],
            "memory.bandwidth": ["memory_test.log", "bandwidth_test.log"],
            "comprehensive": ["comprehensive_test.log"],
        }

        return log_files_map.get(test_type, ["comprehensive_test.log"])

    def _save_stream_to_csv(self, metrics: list[Dict], filename: str) -> Optional[Path]:
        """Save STREAM benchmark metrics to CSV file."""
        if not metrics:
            return None

        csv_data = []
        for metric in metrics:
            if "stream.bandwidth" in metric.get("name", ""):
                operation = metric["name"].split(".")[-1]
                csv_data.append({
                    "operation": operation,
                    "bandwidth_gbps": metric["value"],
                    "unit": metric["unit"]
                })

        if csv_data:
            return self._save_csv(csv_data, filename, ["operation", "bandwidth_gbps", "unit"])
        return None

    def _save_metrics_to_json(
        self, metrics: list[Dict], testcase: str, run_id: str, timestamp: str
    ) -> Path:
        """Save all metrics to a JSON file."""
        safe_name = testcase.replace(".", "_").replace("/", "_")
        filename = f"{safe_name}_{run_id}_{timestamp}_metrics.json"
        json_path = self.output_dir / filename

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "testcase": testcase,
                "run_id": run_id,
                "timestamp": timestamp,
                "metrics": metrics
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved metrics JSON to: {json_path}")
        return json_path
