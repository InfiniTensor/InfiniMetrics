#!/usr/bin/env python3
"""
Executor - Universal Test Execution Framework
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from infinimetrics.adapter import BaseAdapter
from infinimetrics.input import TestInput

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """
    Standardized test result structure.

    Used throughout the execution lifecycle and returned to Dispatcher.

    Note:
        success: 0 = success, non-zero = failure code (following Linux convention)
    """

    run_id: str
    testcase: str
    success: int  # 0 = success, non-zero = failure code
    result_file: Optional[str] = None
    skipped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to lightweight dictionary format for Dispatcher aggregation."""
        return {
            "run_id": self.run_id,
            "testcase": self.testcase,
            "success": self.success,
            "result_file": self.result_file,
            "skipped": self.skipped,
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

        # Setup output directory from config
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

        # Convert payload to TestInput object
        self.test_input = TestInput.from_dict(self.payload)

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
            TestResult object with success flag and file path.
        """
        logger.info(f"Executor: Running {self.testcase}")

        # Initialize TestResult directly (default: success=0)
        test_result = TestResult(
            run_id=self.run_id,
            testcase=self.testcase,
            success=0,  # Default to success
            result_file=None,
        )

        try:
            # Phase 1: Setup
            self.setup()

            # Phase 2: Process
            logger.debug(f"Executor: Calling adapter.process()")
            response = self.adapter.process(self.test_input)

            # Process response (0 = success, non-zero = failure)
            test_result.success = response.get("success", 1)

            if test_result.success != 0:
                logger.warning(
                    f"Executor: Adapter failed with error code {test_result.success}"
                )

            # Phase 3: Teardown (cleanup, save result)
            result_file = self._save_result(response)
            test_result.result_file = result_file

            logger.info(
                f"Executor: {self.testcase} completed success={test_result.success}"
            )

            return test_result

        except Exception as e:
            logger.error(f"Executor: {self.testcase} failed: {e}", exc_info=True)

            # Still run teardown on failure
            self._save_result(None)
            test_result.success = 1  # Failure

            return test_result

    def _save_result(self, result: Dict[str, Any]) -> str:
        """
        Save detailed result to disk as JSON.

        Args:
            result: Complete result dict with data and metrics

        Returns:
            Absolute path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = self.testcase.replace(".", "_").replace("/", "_")
        filename = f"{safe_name}_{timestamp}_results.json"
        output_file = self.output_dir / filename

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.debug(f"Executor: Results saved to {output_file}")
        return str(output_file)
