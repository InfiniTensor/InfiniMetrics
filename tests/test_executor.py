#!/usr/bin/env python3
"""
Unit tests for executor.py
"""

import unittest
import sys
import os
import tempfile
import shutil
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infinimetrics.adapter import BaseAdapter
from infinimetrics.executor import Executor


class MockAdapter(BaseAdapter):
    """Mock adapter for testing."""

    def __init__(self, succeed=True):
        self.succeed = succeed
        self.setup_called = False
        self.teardown_called = False

    def process(self, payload):
        """Mock process implementation."""
        if not self.succeed:
            return {
                'success': 0,
                'error': 'Mock error'
            }

        return {
            'success': 1,
            'data': {
                'testcase': payload.get('testcase'),
                'result': 'test_data'
            },
            'metrics': [
                {'name': 'test_metric', 'value': 42, 'unit': 'units'}
            ]
        }

    def setup(self, config=None):
        self.setup_called = True

    def teardown(self):
        self.teardown_called = True


class TestExecutor(unittest.TestCase):
    """Test cases for Executor."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.payload = {
            'run_id': 'test_run_123',
            'testcase': 'infer.Test.Model',
            'config': {
                'output_dir': self.temp_dir,
                'model_path': '/fake/path'
            }
        }

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_executor_initialization(self):
        """Test executor initializes correctly."""
        adapter = MockAdapter()
        executor = Executor(self.payload, adapter)

        self.assertEqual(executor.payload, self.payload)
        self.assertEqual(executor.testcase, 'infer.Test.Model')
        self.assertEqual(executor.run_id, 'test_run_123')
        self.assertEqual(executor.test_type, 'inference')

    def test_detect_test_type_inference(self):
        """Test test type detection for inference."""
        adapter = MockAdapter()
        payload = {'testcase': 'infer.InfiniLM.Direct'}
        executor = Executor(payload, adapter)

        self.assertEqual(executor.test_type, 'inference')

    def test_detect_test_type_operator(self):
        """Test test type detection for operator."""
        adapter = MockAdapter()
        payload = {'testcase': 'train.InfiniTrain.SFT'}
        executor = Executor(payload, adapter)

        self.assertEqual(executor.test_type, 'operator')

    def test_detect_test_type_default(self):
        """Test test type detection defaults to inference."""
        adapter = MockAdapter()
        payload = {'testcase': 'unknown.test.case'}
        executor = Executor(payload, adapter)

        self.assertEqual(executor.test_type, 'inference')

    def test_run_success(self):
        """Test successful test execution."""
        adapter = MockAdapter(succeed=True)
        executor = Executor(self.payload, adapter)

        result = executor.run()

        # Check lightweight result (new format)
        self.assertEqual(result['success'], 1)
        self.assertEqual(result['run_id'], 'test_run_123')
        self.assertEqual(result['testcase'], 'infer.Test.Model')
        self.assertIn('duration', result)
        self.assertIn('result_file', result)
        self.assertTrue(adapter.setup_called)
        self.assertTrue(adapter.teardown_called)

        # Verify detailed result was saved to file
        import json
        from pathlib import Path
        result_file = Path(result['result_file'])
        self.assertTrue(result_file.exists())

        with open(result_file, 'r') as f:
            detailed_result = json.load(f)

        # Detailed file should contain full data
        self.assertIn('data', detailed_result)
        self.assertIn('metrics', detailed_result)

    def test_run_failure(self):
        """Test failed test execution."""
        adapter = MockAdapter(succeed=False)
        executor = Executor(self.payload, adapter)

        result = executor.run()

        self.assertEqual(result['success'], 0)
        self.assertIn('error', result)

    def test_run_with_metrics(self):
        """Test execution collects metrics."""
        adapter = MockAdapter(succeed=True)
        executor = Executor(self.payload, adapter)

        result = executor.run()

        # Lightweight result doesn't include metrics directly
        self.assertIn('result_file', result)

        # Metrics are in the detailed file
        import json
        from pathlib import Path
        with open(result['result_file'], 'r') as f:
            detailed_result = json.load(f)

        self.assertIn('metrics', detailed_result)
        self.assertGreater(len(detailed_result['metrics']), 0)

        # Check execution duration metric
        duration_metrics = [m for m in detailed_result['metrics'] if m['name'] == 'execution.duration']
        self.assertEqual(len(duration_metrics), 1)

    def test_result_file_creation(self):
        """Test results are saved to file."""
        adapter = MockAdapter(succeed=True)
        executor = Executor(self.payload, adapter)

        result = executor.run()

        self.assertIn('result_file', result)
        self.assertTrue(os.path.exists(result['result_file']))

        # Verify file contents
        with open(result['result_file'], 'r') as f:
            saved_data = json.load(f)

        self.assertEqual(saved_data['testcase'], 'infer.Test.Model')
        self.assertEqual(saved_data['success'], 1)

    def test_output_dir_creation(self):
        """Test output directory is created."""
        new_temp_dir = os.path.join(self.temp_dir, 'new_output')
        self.payload['config']['output_dir'] = new_temp_dir

        adapter = MockAdapter()
        executor = Executor(self.payload, adapter)
        executor.run()

        self.assertTrue(os.path.exists(new_temp_dir))

    def test_adapter_exception_handling(self):
        """Test executor handles adapter exceptions."""
        class FailingAdapter(BaseAdapter):
            def setup(self, config=None):
                raise RuntimeError("Setup failed")

            def process(self, payload):
                return {'success': 1}

            def teardown(self):
                pass

        adapter = FailingAdapter()
        executor = Executor(self.payload, adapter)

        result = executor.run()

        self.assertEqual(result['success'], 0)
        self.assertIn('error', result)


class TestExecutorFactory(unittest.TestCase):
    """Test cases for ExecutorFactory."""

    def test_create_executor(self):
        """Test ExecutorFactory.create() method."""
        adapter = MockAdapter()
        payload = {
            'run_id': 'test',
            'testcase': 'infer.Test',
            'config': {}
        }

        executor = ExecutorFactory.create(payload, adapter)

        self.assertIsInstance(executor, Executor)
        self.assertEqual(executor.payload, payload)
        self.assertEqual(executor.adapter, adapter)


class TestExecutorIntegration(unittest.TestCase):
    """Integration tests for Executor with real payloads."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_with_operator_payload(self):
        """Test executor with operator test payload."""
        payload = {
            "run_id": "train.infiniTrain.SFT.test",
            "testcase": "train.InfiniTrain.SFT",
            "config": {
                "model_source": "FM9G_70B",
                "operator": "Conv",
                "device": "cuda",
                "output_dir": self.temp_dir
            }
        }

        adapter = MockAdapter()
        executor = Executor(payload, adapter)
        result = executor.run()

        self.assertEqual(result['testcase'], 'train.InfiniTrain.SFT')
        self.assertEqual(executor.test_type, 'operator')

    def test_with_inference_payload(self):
        """Test executor with inference test payload."""
        payload = {
            "run_id": "infer.infinilm.direct.test",
            "testcase": "infer.InfiniLM.Direct",
            "config": {
                "model": "Qwen3-1.7B",
                "model_path": "/fake/model/path",
                "device": {"gpu_platform": "nvidia"},
                "output_dir": self.temp_dir
            }
        }

        adapter = MockAdapter()
        executor = Executor(payload, adapter)
        result = executor.run()

        self.assertEqual(result['testcase'], 'infer.InfiniLM.Direct')
        self.assertEqual(executor.test_type, 'inference')


if __name__ == '__main__':
    unittest.main()
