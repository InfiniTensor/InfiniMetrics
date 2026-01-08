#!/usr/bin/env python3
"""Dispatcher - Test Orchestration Framework"""

import logging
import json
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

from infinimetrics.adapter import BaseAdapter
from infinimetrics.executor import Executor, TestResult

logger = logging.getLogger(__name__)

# Adapter registry: maps (test_type, framework) -> adapter factory
_ADAPTER_REGISTRY = {
    ("operator", "infinicore"): lambda: _create_infinicore_adapter(),
}


def _create_infinicore_adapter():
    """Create InfiniCore adapter (lazy import)."""
    from infinimetrics.operators.infinicore_adapter import InfiniCoreAdapter

    return InfiniCoreAdapter()


class Dispatcher:
    """Test orchestration dispatcher for managing test executions."""

    def validate_input(self, test_input: Dict[str, Any]) -> bool:
        """
        Validate configuration. Override this method to add custom validation.

        Returns:
            True if test_input is valid, False otherwise
        """
        return "testcase" in test_input

    def dispatch(self, inputs: Any) -> Dict[str, Any]:
        """
        Route payloads to appropriate adapters and execute tests.

        Two-phase execution:
        1. Validation phase: Create all adapters and validate inputs
        2. Execution phase: Execute tests with valid adapters

        Args:
            inputs: Single dict or list of dicts with 'testcase' field

        Returns:
            Aggregated results dictionary
        """
        # Normalize to list
        if isinstance(inputs, dict):
            inputs = [inputs]
        elif not isinstance(inputs, list):
            raise ValueError(f"Invalid inputs type: {type(inputs)}")

        # Filter valid inputs
        valid_test_inputs = [
            test_input
            for test_input in inputs
            if isinstance(test_input, dict) and self.validate_input(test_input)
        ]
        skipped = len(inputs) - len(valid_test_inputs)
        logger.info(
            f"Processing {len(valid_test_inputs)} valid inputs (skipped {skipped} invalid)"
        )

        # Phase 1: Validation - Create all adapters
        valid_executions = []
        skipped_results = []

        for test_input in valid_test_inputs:
            testcase = test_input["testcase"]
            test_type, framework = self._parse_testcase(testcase)

            try:
                adapter = self._create_adapter(test_type, framework)
                valid_executions.append((test_input, adapter))
                logger.debug(f"Validated {testcase} - adapter ready")
            except ValueError as e:
                logger.error(f"Skipping {testcase}: {e}")
                skipped_results.append(
                    TestResult(
                        run_id=test_input.get("run_id", "unknown"),
                        testcase=testcase,
                        result_code=1,  # non-zero = error code
                        result_file=None,
                        skipped=True,
                    ).to_dict()
                )

        logger.info(
            f"Validation complete: {len(valid_executions)} valid, {len(skipped_results)} skipped"
        )

        # Phase 2: Execution - Run all valid tests
        all_results = []
        for idx, (test_input, adapter) in enumerate(valid_executions, 1):
            testcase = test_input["testcase"]
            logger.info(f"[{idx}/{len(valid_executions)}] Executing {testcase}")

            executor = Executor(test_input, adapter)
            test_result = executor.execute()
            all_results.append(test_result.to_dict())

        # Add skipped results
        all_results.extend(skipped_results)

        # Aggregate and save
        aggregated = self._aggregate_results(all_results)
        self._save_summary(aggregated)
        return aggregated

    def _create_adapter(self, test_type: str, framework: str) -> BaseAdapter:
        """Create adapter based on test type and framework."""
        key = (test_type, framework)
        if key in _ADAPTER_REGISTRY:
            adapter_factory = _ADAPTER_REGISTRY[key]
            return adapter_factory()

        raise ValueError(
            f"Adapter not registered: test_type={test_type}, framework={framework}"
        )

    def _parse_testcase(self, testcase: str) -> tuple[str, str]:
        """
        Parse testcase to extract test_type and framework.

        Args:
            testcase: Test case name in format 'test_type.Framework.Something'

        Returns:
            Tuple of (test_type, framework)

        Examples:
            'infer.InfiniLM.Direct' -> ('infer', 'infinilm')
            'operator.InfiniCore.Conv' -> ('operator', 'infinicore')
        """
        parts = testcase.split(".")

        if len(parts) < 2:
            logger.warning(f"Invalid testcase format: {testcase}, using defaults")
            return "operator", "infinicore"

        # First part is test_type
        test_type = parts[0].lower()
        # Second part is framework
        framework = parts[1].lower()

        return test_type, framework

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from executors."""
        total = len(results)
        successful = sum(1 for r in results if r["result_code"] == 0)

        return {
            "total_tests": total,
            "successful_tests": successful,
            "failed_tests": total - successful,
            "results": [
                {
                    "run_id": r["run_id"],
                    "testcase": r["testcase"],
                    "result_code": r["result_code"],
                    "result_file": r["result_file"],
                    "skipped": r.get("skipped", False),
                }
                for r in results
            ],
            "timestamp": datetime.now().isoformat(),
        }

    def _save_summary(self, aggregated: Dict[str, Any]) -> None:
        """Save aggregated results summary to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dispatcher_summary_{timestamp}.json"

        # Save summary to a separate directory
        summary_dir = Path("./summary_output")
        summary_dir.mkdir(parents=True, exist_ok=True)

        with open(summary_dir / filename, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)

        logger.info(f"Summary saved to {summary_dir / filename}")
