#!/usr/bin/env python3
"""
Dispatcher - Test Orchestration Framework
"""

import logging
from typing import Dict, Any, List, Dict
from pathlib import Path
from datetime import datetime

from adapter import BaseAdapter
from executor import Executor

logger = logging.getLogger(__name__)


class Dispatcher:
    """
    Test orchestration dispatcher for managing test executions.

    Routes test payloads to appropriate adapters based on testcase type.
    """

    def __init__(self, adapters: Dict[str, BaseAdapter]):
        """
        Initialize dispatcher.

        Args:
            adapters: Dict mapping test type to adapter instance
                      e.g., {'inference': InferenceAdapter(), 'operator': OperatorAdapter()}
        """
        self.adapters = adapters
        self.executors: List[Executor] = []
        self.results: List[Dict[str, Any]] = []

        logger.info(f"Dispatcher initialized with adapters: {list(adapters.keys())}")

    def dispatch(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point: Route payload to appropriate adapter and execute test.

        Args:
            payload: Complete test payload with testcase, config, etc.

        Returns:
            Aggregated results dictionary
        """
        testcase = payload.get('testcase', 'unknown')
        logger.info(f"Dispatcher: Routing testcase={testcase}")

        # Detect test type and get adapter
        test_type = self._detect_test_type(testcase)
        adapter = self.adapters.get(test_type)

        if not adapter:
            # Fallback to first available adapter
            logger.warning(f"Dispatcher: No adapter found for test_type={test_type}, using fallback")
            adapter = list(self.adapters.values())[0]

        logger.info(f"Dispatcher: Using adapter: {adapter.__class__.__name__}")

        # Create and run executor
        executor = Executor(payload, adapter)
        result = executor.run()

        # Aggregate results
        aggregated = self._aggregate_results([result])

        # Save summary
        self._save_summary(aggregated)

        logger.info("Dispatcher: Test orchestration completed")
        return aggregated

    def _detect_test_type(self, testcase: str) -> str:
        """Detect test type from testcase name."""
        testcase_lower = testcase.lower()
        if 'train' in testcase_lower or 'operator' in testcase_lower:
            return 'operator'
        elif 'infer' in testcase_lower:
            return 'inference'
        else:
            # Default to inference for unknown test types
            return 'inference'

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from executors."""
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.get('success') == 1)

        aggregated = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': total_tests - successful_tests,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Dispatcher: Aggregation completed - {successful_tests}/{total_tests} tests successful")

        return aggregated

    def _save_summary(self, aggregated: Dict[str, Any]) -> None:
        """Save aggregated results summary to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dispatcher_summary_{timestamp}.json"

        # Use output directory from first result if available
        output_dir = Path('./output')
        if aggregated.get('results') and len(aggregated['results']) > 0:
            result_file = aggregated['results'][0].get('result_file')
            if result_file:
                output_dir = Path(result_file).parent

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / filename

        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)

        logger.info(f"Dispatcher: Summary saved to {output_file}")
