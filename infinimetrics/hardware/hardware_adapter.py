#!/usr/bin/env python3
"""Hardware Test Adapter for CUDA Unified Benchmark Suite."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from infinimetrics.adapter import BaseAdapter
from infinimetrics.common.build_manager import BuildManager, BuildError
from infinimetrics.common.command_runner import CommandRunner
from infinimetrics.common.device_manager import DeviceManager

from .constants import (
    TEST_TYPE_MAP,
    CACHE_TEST_TIMEOUT,
    DEFAULT_TEST_TIMEOUT,
    LOG_FILES_MAP,
)
from .parsers import (
    MemoryBandwidthParser,
    StreamBenchmarkParser,
    CacheBandwidthParser,
)

logger = logging.getLogger(__name__)


class HardwareTestAdapter(BaseAdapter):
    """Adapter for CUDA unified hardware performance tests."""

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

        # Initialize utilities
        self.command_runner = CommandRunner(default_timeout=DEFAULT_TEST_TIMEOUT)
        self.build_manager = BuildManager(build_timeout=300)

    def setup(self, config: Dict[str, Any]) -> None:
        """Initialize resources before running tests."""
        # Skip build if CPU mode or explicitly requested
        if DeviceManager.should_skip_build(config):
            return

        # Check if build artifact already exists
        if Path(self.cuda_perf_path).exists():
            return

        # Build CUDA project
        self._build_cuda_project()

    def process(self, test_input: Any) -> Dict[str, Any]:
        """Process test input and return results."""
        # Convert test_input to dict if needed
        test_input_dict = self._ensure_dict(test_input)
        if not test_input_dict:
            return self._error_response(f"Invalid test_input type: {type(test_input)}")

        # Extract common fields
        testcase, config, run_id = self._extract_test_fields(test_input_dict)
        logger.info("HardwareTestAdapter: Processing %s", testcase)

        # Update output directory from config
        self.output_dir = Path(config.get("output_dir", "./output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get test configuration
        device = DeviceManager.get_device_type(config)
        test_type = config.get("test_type", "comprehensive")

        try:
            if DeviceManager.is_cpu_mode(config):
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
            logger.error("Hardware test failed: %s", e, exc_info=True)
            return self._error_response(str(e))

    def _build_cuda_project(self) -> None:
        """Build CUDA project if needed."""
        try:
            self.build_manager.build_script(self.build_dir, str(self.build_script))
        except BuildError as e:
            logger.error("CUDA build failed: %s", e)
            raise

    @staticmethod
    def _ensure_dict(test_input: Any) -> Dict[str, Any]:
        """Convert test input to dictionary if possible."""
        if isinstance(test_input, dict):
            return test_input
        if hasattr(test_input, "to_dict"):
            return test_input.to_dict()
        return None

    @staticmethod
    def _extract_test_fields(test_input: Dict[str, Any]) -> tuple:
        """Extract common test fields from input."""
        return (
            test_input.get("testcase", "unknown"),
            test_input.get("config", {}),
            test_input.get("run_id", "unknown"),
        )

    @staticmethod
    def _error_response(error_msg: str) -> Dict[str, Any]:
        """Generate error response."""
        return {
            "result_code": 1,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error_msg": error_msg,
            "metrics": [],
        }

    def _build_command(self, config: Dict[str, Any]) -> List[str]:
        """Build command for CUDA test suite."""
        test_type = config.get("test_type", "all")
        cuda_test_type = TEST_TYPE_MAP.get(test_type, test_type.lower())
        cmd = [self.cuda_perf_path, f"--{cuda_test_type}"]

        # Add optional parameters
        param_mappings = {
            "device_id": "--device",
            "iterations": "--iterations",
            "array_size": "--array-size",
            "buffer_size_mb": "--buffer-size",
        }

        for key, param_name in param_mappings.items():
            if key in config:
                cmd.extend([param_name, str(config[key])])

        return cmd

    def _execute_test(self, cmd: List[str], test_type: str) -> str:
        """Execute CUDA test and return output."""
        if not Path(self.cuda_perf_path).exists():
            raise RuntimeError(f"cuda_perf_suite not found: {self.cuda_perf_path}")

        # Determine timeout based on test type
        timeout = (
            CACHE_TEST_TIMEOUT if test_type.lower() == "cache" else DEFAULT_TEST_TIMEOUT
        )

        return self.command_runner.run_get_output(cmd, timeout=timeout)

    def _parse_output(self, output: str, test_type: str, run_id: str) -> List[Dict]:
        """Parse test output based on test type."""
        # Handle comprehensive test (all sub-tests)
        if test_type == "Comprehensive":
            return (
                self._parse_with_parser(
                    MemoryBandwidthParser,
                    output,
                    run_id,
                    is_sweep=True,
                    prefix="mem_sweep",
                )
                + self._parse_with_parser(
                    MemoryBandwidthParser,
                    output,
                    run_id,
                    is_sweep=False,
                    prefix="mem_bw",
                )
                + self._parse_with_parser(StreamBenchmarkParser, output, run_id)
                + self._parse_with_parser(CacheBandwidthParser, output, run_id)
            )

        # Map test types to parsers
        parser_configs = {
            "MemSweep": (
                MemoryBandwidthParser,
                {"is_sweep": True, "prefix": "mem_sweep"},
            ),
            "MemBw": (MemoryBandwidthParser, {"is_sweep": False, "prefix": "mem_bw"}),
            "Stream": (StreamBenchmarkParser, {}),
            "Cache": (CacheBandwidthParser, {}),
        }

        if test_type in parser_configs:
            parser_class, parser_config = parser_configs[test_type]
            return self._parse_with_parser(
                parser_class, output, run_id, **parser_config
            )

        return []

    def _parse_with_parser(
        self, parser_class, output: str, run_id: str, **parser_config
    ) -> List[Dict]:
        """
        Parse output using a specific parser class.

        Args:
            parser_class: Parser class to instantiate
            output: Raw test output
            run_id: Test run identifier
            **parser_config: Additional configuration for the parser

        Returns:
            List of parsed metrics
        """
        # Prepare parser initialization arguments
        init_args = {"output_dir": self.output_dir, "metric_prefix": "hardware"}

        # Handle parser-specific configurations
        if parser_class == MemoryBandwidthParser:
            init_args["is_sweep"] = parser_config.get("is_sweep", True)
            if "prefix" in parser_config:
                # Update metric prefix based on test type
                init_args["metric_prefix"] = f"hardware.{parser_config['prefix']}"

        # Instantiate parser and parse output
        parser = parser_class(**init_args)
        return parser.parse(output, run_id)

    def _process_existing_outputs(self, test_type: str, run_id: str) -> List[Dict]:
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
            parser_configs = {
                "memory_test.log": (
                    MemoryBandwidthParser,
                    {"is_sweep": True, "prefix": "mem_sweep"},
                ),
                "bandwidth_test.log": (
                    MemoryBandwidthParser,
                    {"is_sweep": False, "prefix": "mem_bw"},
                ),
                "stream_test.log": (StreamBenchmarkParser, {}),
                "cache_test.log": (CacheBandwidthParser, {}),
                "comprehensive_test.log": (None, None),  # Special handling
            }

            if log_file_name in parser_configs:
                parser_class, parser_config = parser_configs[log_file_name]

                if log_file_name == "comprehensive_test.log":
                    # Handle comprehensive test - run all parsers
                    all_metrics.extend(
                        self._parse_with_parser(
                            MemoryBandwidthParser,
                            output,
                            run_id,
                            is_sweep=True,
                            prefix="mem_sweep",
                        )
                    )
                    all_metrics.extend(
                        self._parse_with_parser(
                            MemoryBandwidthParser,
                            output,
                            run_id,
                            is_sweep=False,
                            prefix="mem_bw",
                        )
                    )
                    all_metrics.extend(
                        self._parse_with_parser(StreamBenchmarkParser, output, run_id)
                    )
                    all_metrics.extend(
                        self._parse_with_parser(CacheBandwidthParser, output, run_id)
                    )
                elif parser_class:
                    metrics = self._parse_with_parser(
                        parser_class, output, run_id, **parser_config
                    )
                    all_metrics.extend(metrics)

        return all_metrics
