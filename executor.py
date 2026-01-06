#!/usr/bin/env python3
"""
Executor - Universal Test Execution Framework
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from adapter import BaseAdapter

logger = logging.getLogger(__name__)


class Executor:
    """
    Universal test executor for inference, operator, and training tests.

    Processes complete test payloads with testcase, config, and optional metrics.
    """

    def __init__(self, payload: Dict[str, Any], adapter: BaseAdapter):
        """
        Initialize executor.

        Args:
            payload: Complete test payload with testcase, config, etc.
            adapter: Adapter instance implementing BaseAdapter
        """
        self.payload = payload
        self.adapter = adapter
        self.testcase = payload.get('testcase', 'unknown_test')
        self.run_id = payload.get('run_id', '')

        # Detect test type from testcase name
        self.test_type = self._detect_test_type(self.testcase)

        # Setup output directory
        output_dir = payload.get('config', {}).get('output_dir', './output')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Executor initialized: testcase={self.testcase}, test_type={self.test_type}")

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

    def run(self) -> Dict[str, Any]:
        """
        Run the complete test.

        Returns:
            Results dictionary with success, data, metrics, etc.
        """
        logger.info(f"Executor: Running test - {self.testcase}")
        start_ts = time.time()

        result = {
            'run_id': self.run_id,
            'testcase': self.testcase,
            'success': 0,
            'start_time': start_ts,
            'data': {},
            'metrics': []
        }

        try:
            # Setup adapter
            config = self.payload.get('config', {})
            self.adapter.setup(config)

            # Process payload through adapter
            logger.info(f"Executor: Calling adapter.process() for testcase={self.testcase}")
            response = self.adapter.process(self.payload)

            # Process response
            result['success'] = response.get('success', 0)
            result['data'] = response.get('data', {})

            if result.get('success') == 0 and 'error' in response:
                logger.warning(f"Executor: Adapter returned error: {response['error']}")
                result['error'] = response['error']

            # Collect metrics from payload and response
            duration = time.time() - start_ts
            result['duration'] = duration

            # Add execution metrics
            result['metrics'].append({
                'name': 'execution.duration',
                'value': duration,
                'unit': 's'
            })

            # Add adapter-provided metrics
            if 'metrics' in response:
                result['metrics'].extend(response['metrics'])

            # Save results
            self._save_results(result)

            logger.info(f"Executor: Test completed - {self.testcase} (success={result['success']}, duration={duration:.2f}s)")

        except Exception as e:
            logger.error(f"Executor: Test failed - {self.testcase}: {e}", exc_info=True)
            result['success'] = 0
            result['error'] = str(e)
            result['duration'] = time.time() - start_ts

        finally:
            try:
                self.adapter.teardown()
            except Exception as teardown_error:
                logger.warning(f"Executor: Teardown failed: {teardown_error}")

        return result

    def _save_results(self, result: Dict[str, Any]) -> None:
        """Save results to disk as JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = self.testcase.replace('.', '_').replace('/', '_')
        filename = f"{safe_name}_{timestamp}_results.json"
        output_file = self.output_dir / filename

        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        result['result_file'] = str(output_file)
        logger.info(f"Executor: Results saved to {output_file}")


class ExecutorFactory:
    """Factory for creating Executor instances."""

    @staticmethod
    def create(payload: Dict[str, Any], adapter: BaseAdapter) -> Executor:
        """
        Create an Executor instance.

        Args:
            payload: Complete test payload with testcase, config, etc.
            adapter: Adapter instance

        Returns:
            Configured Executor instance
        """
        return Executor(payload, adapter)
