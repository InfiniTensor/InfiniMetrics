#!/usr/bin/env python3
"""
Executor - Universal Test Execution Framework
"""

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from infinimetrics.adapter import BaseAdapter
from infinimetrics.common.error_handler import ErrorHandler
from infinimetrics.common.hardware_info import collect_hardware_info
from infinimetrics.utils.path_utils import sanitize_filename
from infinimetrics.common.constants import ErrorCode, TEST_CATEGORIES


logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """
    Standardized test result structure.

    Used throughout the execution lifecycle and returned to Dispatcher.

    Note:
        result_code: 0 = success, non-zero = error code (following Linux convention)
    """

    run_id: str
    testcase: str
    result_code: int  # 0 = success, non-zero = error code
    result_file: Optional[str] = None
    skipped: bool = False
    config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to lightweight dictionary format for Dispatcher aggregation."""
        return {
            "run_id": self.run_id,
            "testcase": self.testcase,
            "result_code": self.result_code,
            "result_file": self.result_file,
            "skipped": self.skipped,
            "config": self.config,
        }


class Executor:
    """
    Universal test executor for all test types.

    Responsibilities:
        1. Manage adapter lifecycle (setup -> process -> teardown)
        2. Save results to disk
        3. Return result summary
    """

    def __init__(self, payload: Dict[str, Any], adapter: BaseAdapter):
        """
        Initialize executor.

        Args:
            payload: Test payload with testcase, config, etc.
            adapter: Configured adapter instance
        """
        self.payload = payload
        self.adapter = adapter
        self.testcase = payload.get("testcase", "unknown")
        self.run_id = payload.get("run_id", "")
        self.test_input = None

        config = payload.get("config", {})
        output_dir = config.get("output_dir", "./output")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Executor initialized: testcase={self.testcase}")

    def setup(self) -> None:
        """
        Setup phase - initialize adapter.

        This should be called before execute().
        """
        config = self.payload.get("config", {})

        # Inject testcase, run_id, and other metadata into config
        config["_testcase"] = self.payload.get("testcase", "")
        config["_run_id"] = self.payload.get("run_id", "")
        config["_time"] = self.payload.get("time", None)

        # Initialize test_input from payload
        self.test_input = self.payload

        self.adapter.setup(config)

        logger.debug(f"Executor: Setup complete for {self.testcase}")

    def teardown(self, result: Any) -> str:
        """
        Teardown phase - cleanup adapter, collect metrics, and save results.

        This should be called after process() completes.

        Args:
            result:

        Returns:
            Path to saved result file
        """
        # Always cleanup adapter
        try:
            self.adapter.teardown()
        except Exception as teardown_error:
            logger.warning(
                f"Executor: Teardown failed for {self.testcase}: {teardown_error}"
            )

        # TODO: Add metrics calculation method

        # Save result to disk
        result_file = self._save_result(result)

        logger.debug(f"Executor: Teardown complete for {self.testcase}")
        return result_file

    def execute(self) -> TestResult:
        """
        Execute the complete test with proper lifecycle management.

        Lifecycle:
            1. adapter.setup(config)
            2. adapter.process(payload)
            3. adapter.teardown() - includes saving results
            4. Return TestResult

        Returns:
            TestResult object with result_code and file path.
        """
        logger.info(f"Executor: Running {self.testcase}")

        # Initialize TestResult directly (default: result_code=0)
        config = self.payload.get("config", {})
        test_result = TestResult(
            run_id=self.run_id,
            testcase=self.testcase,
            result_code=0,  # Default to success
            result_file=None,
            config=config,
        )

        response: Dict[str, Any] = {}

        try:
            # Phase 1: Setup
            self.setup()

            # Phase 2: Process
            logger.debug(f"Executor: Calling adapter.process()")
            response = self.adapter.process(self.test_input)

            # Enrich environment ONLY if missing
            if isinstance(response, dict) and "environment" not in response:
                env = self._build_environment(response)

                # rebuild ordered dict (py3.7+ preserves insertion order)
                ordered: Dict[str, Any] = {}
                for k in [
                    "run_id",
                    "time",
                    "testcase",
                    "success",
                    "environment",
                    "result_code",
                    "config",
                    "metrics",
                ]:
                    if k == "environment":
                        ordered["environment"] = env
                    elif k in response:
                        ordered[k] = response[k]

                # append remaining keys in original order (skip those already set)
                for k, v in response.items():
                    if k not in ordered:
                        ordered[k] = v

                response = ordered

            # Phase 3: Teardown (cleanup, save result)
            result_file = self.teardown(response)
            test_result.result_file = result_file

            logger.info(
                f"Executor: {self.testcase} completed with code={test_result.result_code}"
            )

            return test_result

        except subprocess.TimeoutExpired as e:
            # Timeout errors (possible hardware hang)
            test_result.result_code = ErrorCode.TIMEOUT
            ErrorHandler.log_error(self.testcase, e, ErrorCode.TIMEOUT)
            response = self._build_error_response(str(e), ErrorCode.TIMEOUT, config)

        except ValueError as e:
            # Configuration or input validation errors
            test_result.result_code = ErrorCode.CONFIG
            ErrorHandler.log_error(self.testcase, e, ErrorCode.CONFIG)
            response = self._build_error_response(str(e), ErrorCode.CONFIG, config)

        except RuntimeError as e:
            # RuntimeError: analyze error message for specific patterns
            error_code = ErrorHandler.classify_runtime_error(str(e).lower())
            test_result.result_code = error_code
            ErrorHandler.log_error(self.testcase, e, error_code)
            response = self._build_error_response(str(e), error_code, config)

        except Exception as e:
            # Unexpected exceptions
            test_result.result_code = ErrorCode.GENERIC
            logger.error(
                f"Executor: {self.testcase} failed with unexpected exception: {e}",
                exc_info=True,
            )
            response = self._build_error_response(str(e), ErrorCode.GENERIC, config)

        finally:
            # Always save result (even on failure)
            self._finalize_result(test_result, response)

        return test_result

    def _build_error_response(
        self, error_msg: str, result_code: int, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build a response dict containing error information for saving to disk.

        Args:
            error_msg: Error message string
            result_code: Error result code

        Returns:
            Dictionary with basic test info and error details
        """
        response = ErrorHandler.build_error_response(
            self.run_id, self.testcase, error_msg, result_code, config
        )
        response["resolved"] = self._extract_device_info(config)
        return response

    def _extract_device_info(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract device information from config."""
        device_used = 0
        gpus_per_node = 0
        nodes = 1

        # Try device_involved
        if "device_involved" in config:
            try:
                device_used = int(config.get("device_involved", 0) or 0)
            except (ValueError, TypeError):
                device_used = 0

        # Try single_node config
        if isinstance(config.get("single_node"), dict):
            single_node = config["single_node"]
            device_ids = single_node.get("device_ids", [])
            if device_ids:
                device_used = len(device_ids)
            gpus_per_node = device_used
        else:
            gpus_per_node = device_used

        # Try multi_node config
        if "multi_node" in config:
            try:
                nodes = int(config.get("multi_node", {}).get("num_nodes", 1) or 1)
            except (ValueError, TypeError):
                nodes = 1

        return {
            "nodes": nodes,
            "gpus_per_node": gpus_per_node,
            "device_used": device_used,
        }

    def _build_environment(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a unified environment block
        """
        resolved = response.get("resolved", {}) if isinstance(response, dict) else {}
        nodes = int(resolved.get("nodes", 1) or 1)
        device_used = int(resolved.get("device_used", 0) or 0)
        gpn = int(resolved.get("gpus_per_node", 0) or 0)

        # Fallback to config hints if adapter didn't provide
        cfg = self.payload.get("config", {}) or {}

        accel_type = (
            (cfg.get("accelerator_type") or cfg.get("device_type") or "")
            .strip()
            .lower()
        )  # optional
        device_ids = cfg.get("device_ids")

        if device_ids is None and isinstance(cfg.get("single_node"), dict):
            device_ids = cfg["single_node"].get("device_ids")

        if device_used <= 0:
            try:
                device_used = int(cfg.get("device_involved", 0) or 0)
            except Exception:
                device_used = 0

        if nodes <= 1:
            topo = f"{device_used}x1 ring mesh"
        else:
            topo = f"{nodes}x{(gpn or max(1, device_used // nodes))} ring mesh"

        hw = collect_hardware_info(accel_type=accel_type, device_ids=device_ids)

        return {
            "cluster_scale": nodes,
            "topology": topo,
            "cluster": [
                {
                    "machine": {
                        "cpu_model": hw.get("cpu_model", "Unknown"),
                        "memory_gb": hw.get("memory_gb", 0),
                        "accelerators": [
                            {
                                "model": hw.get("gpu_model", "Unknown"),
                                "count": device_used,
                                "memory_gb_per_card": hw.get("gpu_memory_gb", 0),
                                "driver": hw.get("driver_version", "Unknown"),
                                "cuda": hw.get("cuda_version", "Unknown"),
                                # reserved: type of platform
                                "type": hw.get("accelerator_type", "generic"),
                            }
                        ],
                    },
                    "framework": [{"name": "unknown", "version": "unknown"}],
                }
            ],
        }

    def _finalize_result(
        self, test_result: TestResult, response: Dict[str, Any]
    ) -> None:
        """Save result file if not already saved."""
        if not test_result.result_file:
            try:
                result_file = self._save_result(response)
                test_result.result_file = result_file
            except Exception as teardown_error:
                logger.error(f"Executor: Failed to save result: {teardown_error}")

    def _save_result(self, result: Dict[str, Any]) -> str:
        """
        Save detailed result to disk as JSON.

        Args:
            result: Complete result dict with data and metrics

        Returns:
            Absolute path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1) Prefer run_id in result (adapter may generate a richer run_id)
        run_id = None
        if isinstance(result, dict):
            run_id = result.get("run_id") or result.get("raw_result", {}).get("run_id")

        # 2) Fallback to payload run_id
        if not run_id:
            run_id = self.payload.get("run_id") or self.run_id

        # 3) Determine output subdirectory based on testcase
        testcase = self.payload.get("testcase", "")

        # Extract category from testcase (e.g., "hardware.cudaUnified.Comprehensive" -> "hardware")
        category = "other"  # default fallback
        for prefix, subdir in TEST_CATEGORIES.items():
            if testcase.startswith(prefix):
                category = subdir
                break

        if run_id:
            safe_run_id = sanitize_filename(run_id)

            # Put results in category-specific subdirectory
            category_dir = self.output_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{safe_run_id}_results.json"
            output_file = category_dir / filename
        else:
            # Final fallback to old naming (in root output_dir)
            safe_name = self.testcase.replace(".", "_").replace("/", "_")
            filename = f"{safe_name}_{timestamp}_results.json"
            output_file = self.output_dir / filename

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.debug(f"Executor: Results saved to {output_file}")
        return str(output_file)
