#!/usr/bin/env python3
"""Dispatcher - Test Orchestration Framework"""

import logging
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

from infinimetrics.adapter import BaseAdapter
from infinimetrics.executor import Executor, TestResult

logger = logging.getLogger(__name__)


class Dispatcher:
    """Test orchestration dispatcher for managing test executions."""

    def validate_input(self, input: Dict[str, Any]) -> bool:
        """
        Validate configuration. Override this method to add custom validation.

        Returns:
            True if input is valid, False otherwise
        """
        return 'testcase' in input

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
        payloads = [input for input in inputs if isinstance(input, dict) and self.validate_input(input)]
        skipped = len(inputs) - len(payloads)
        logger.info(f"Processing {len(payloads)} valid inputs (skipped {skipped} invalid)")

        # Phase 1: Validation - Create all adapters
        valid_executions = []
        skipped_results = []

        for payload in payloads:
            testcase = payload['testcase']
            test_type, framework = self._parse_testcase(testcase)

            try:
                adapter = self._create_adapter(test_type, framework)
                valid_executions.append((payload, adapter))
                logger.debug(f"Validated {testcase} - adapter ready")
            except ValueError as e:
                logger.error(f"Skipping {testcase}: {e}")
                skipped_results.append(TestResult(
                    run_id=payload.get('run_id', 'unknown'),
                    testcase=testcase,
                    success=1,  # non-zero = failure
                    result_file=None,
                    skipped=True
                ).to_dict())

        logger.info(f"Validation complete: {len(valid_executions)} valid, {len(skipped_results)} skipped")

        # Phase 2: Execution - Run all valid tests
        all_results = []
        for idx, (payload, adapter) in enumerate(valid_executions, 1):
            testcase = payload['testcase']
            logger.info(f"[{idx}/{len(valid_executions)}] Executing {testcase}")

            executor = Executor(payload, adapter)
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
        if test_type == 'operator':
            from infinimetrics.operators.infinicore_adapter import InfiniCoreAdapter
            return InfiniCoreAdapter()

        if test_type == 'inference':
            if framework == 'infinilm':
                from infinimetrics.inference.adapters.infinilm_adapter import InfiniLMAdapter
                from infinimetrics.inference.infer_config import InferConfig
                # InfiniLMAdapter needs InferConfig object
                return InfiniLMAdapter(None)  # Config will be set in setup()
            raise ValueError(f"{framework} adapter not implemented")

        raise ValueError(f"{test_type} adapter not implemented")

    def _parse_testcase(self, testcase: str) -> tuple[str, str]:
        """
        Parse testcase to extract test_type and framework.

        Args:
            testcase: Test case name in format 'test_type.Framework.Something'

        Returns:
            Tuple of (test_type, framework)

        Examples:
            'infer.InfiniLM.Direct' -> ('inference', 'infinilm')
            'train.Operator.Conv' -> ('operator', 'operator')
        """
        parts = testcase.split('.')

        if len(parts) < 2:
            logger.warning(f"Invalid testcase format: {testcase}, using defaults")
            return 'operator', 'operator'

        # First part is test_type (needs mapping)
        test_type_part = parts[0].lower()
        test_type_mapping = {
            'infer': 'inference',
            'train': 'operator',
            'eval': 'operator'
        }
        test_type = test_type_mapping.get(test_type_part, 'operator')

        # Second part is framework
        framework = parts[1].lower()

        return test_type, framework

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from executors."""
        total = len(results)
        successful = sum(1 for r in results if r['success'] == 0)

        return {
            'total_tests': total,
            'successful_tests': successful,
            'failed_tests': total - successful,
            'results': [
                {
                    'run_id': r['run_id'],
                    'testcase': r['testcase'],
                    'success': r['success'],
                    'result_file': r['result_file'],
                    'skipped': r.get('skipped', False)
                }
                for r in results
            ],
            'timestamp': datetime.now().isoformat()
        }

    def _save_summary(self, aggregated: Dict[str, Any]) -> None:
        """Save aggregated results summary to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dispatcher_summary_{timestamp}.json"

        # Get output directory from first result
        output_dir = Path('./summary_output')
        if aggregated['results']:
            result_file = aggregated['results'][0].get('result_file')
            if result_file:
                output_dir = Path(result_file).parent

        output_dir.mkdir(parents=True, exist_ok=True)

        import json
        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)

        logger.info(f"Summary saved to {output_dir / filename}")
