#!/usr/bin/env python3
"""
Unit tests for dispatcher.py
"""

import unittest
import sys
import os
import tempfile
import shutil
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapter import BaseAdapter
from dispatcher import Dispatcher


class MockInferenceAdapter(BaseAdapter):
    """Mock inference adapter for testing."""

    def process(self, payload):
        return {
            'success': 1,
            'data': {'type': 'inference', 'testcase': payload.get('testcase')},
            'metrics': []
        }


class MockOperatorAdapter(BaseAdapter):
    """Mock operator adapter for testing."""

    def process(self, payload):
        return {
            'success': 1,
            'data': {'type': 'operator', 'testcase': payload.get('testcase')},
            'metrics': []
        }


class TestDispatcher(unittest.TestCase):
    """Test cases for Dispatcher."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.adapters = {
            'inference': MockInferenceAdapter(),
            'operator': MockOperatorAdapter()
        }

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_dispatcher_initialization(self):
        """Test dispatcher initializes correctly."""
        dispatcher = Dispatcher(self.adapters)

        self.assertEqual(dispatcher.adapters, self.adapters)
        self.assertEqual(len(dispatcher.adapters), 2)

    def test_dispatch_inference_test(self):
        """Test dispatching inference test."""
        dispatcher = Dispatcher(self.adapters)

        payload = {
            'run_id': 'test_infer_123',
            'testcase': 'infer.InfiniLM.Direct',
            'config': {
                'model': 'Qwen3-1.7B',
                'output_dir': self.temp_dir
            }
        }

        result = dispatcher.dispatch(payload)

        self.assertEqual(result['total_tests'], 1)
        self.assertEqual(result['successful_tests'], 1)
        self.assertEqual(result['failed_tests'], 0)
        self.assertEqual(len(result['results']), 1)

        # Check the actual test result
        test_result = result['results'][0]
        self.assertEqual(test_result['success'], 1)
        self.assertEqual(test_result['testcase'], 'infer.InfiniLM.Direct')

    def test_dispatch_operator_test(self):
        """Test dispatching operator test."""
        dispatcher = Dispatcher(self.adapters)

        payload = {
            'run_id': 'test_operator_456',
            'testcase': 'train.InfiniTrain.SFT',
            'config': {
                'operator': 'Conv',
                'output_dir': self.temp_dir
            }
        }

        result = dispatcher.dispatch(payload)

        self.assertEqual(result['total_tests'], 1)
        self.assertEqual(result['successful_tests'], 1)

        test_result = result['results'][0]
        self.assertEqual(test_result['testcase'], 'train.InfiniTrain.SFT')

    def test_detect_test_type_inference(self):
        """Test test type detection for inference."""
        dispatcher = Dispatcher(self.adapters)

        test_type = dispatcher._detect_test_type('infer.InfiniLM.Direct')
        self.assertEqual(test_type, 'inference')

    def test_detect_test_type_operator(self):
        """Test test type detection for operator."""
        dispatcher = Dispatcher(self.adapters)

        test_type = dispatcher._detect_test_type('train.InfiniTrain.SFT')
        self.assertEqual(test_type, 'operator')

    def test_detect_test_type_default(self):
        """Test test type detection defaults to inference."""
        dispatcher = Dispatcher(self.adapters)

        test_type = dispatcher._detect_test_type('unknown.test.case')
        self.assertEqual(test_type, 'inference')

    def test_aggregate_results(self):
        """Test result aggregation."""
        dispatcher = Dispatcher(self.adapters)

        results = [
            {'success': 1, 'testcase': 'test1'},
            {'success': 1, 'testcase': 'test2'},
            {'success': 0, 'testcase': 'test3'}
        ]

        aggregated = dispatcher._aggregate_results(results)

        self.assertEqual(aggregated['total_tests'], 3)
        self.assertEqual(aggregated['successful_tests'], 2)
        self.assertEqual(aggregated['failed_tests'], 1)
        self.assertEqual(aggregated['results'], results)

    def test_summary_file_creation(self):
        """Test summary file is created."""
        dispatcher = Dispatcher(self.adapters)

        payload = {
            'run_id': 'test_summary',
            'testcase': 'infer.Test',
            'config': {'output_dir': self.temp_dir}
        }

        result = dispatcher.dispatch(payload)

        # Check that summary file was created in the output directory
        summary_files = [f for f in os.listdir(self.temp_dir) if f.startswith('dispatcher_summary_')]
        self.assertGreater(len(summary_files), 0)

    def test_adapter_fallback(self):
        """Test fallback when specific adapter not found."""
        # Create dispatcher with only inference adapter
        adapters = {'inference': MockInferenceAdapter()}
        dispatcher = Dispatcher(adapters)

        # Request operator test (should fallback to inference)
        payload = {
            'run_id': 'test_fallback',
            'testcase': 'train.Operator.Test',
            'config': {'output_dir': self.temp_dir}
        }

        # Should not crash, should use fallback
        result = dispatcher.dispatch(payload)
        self.assertIsNotNone(result)


class TestDispatcherIntegration(unittest.TestCase):
    """Integration tests for Dispatcher with real payloads."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.adapters = {
            'inference': MockInferenceAdapter(),
            'operator': MockOperatorAdapter()
        }

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_with_real_operator_payload(self):
        """Test dispatcher with real operator payload format."""
        payload = {
            "run_id": "train.infiniTrain.SFT.a8b4c9e1",
            "time": "2025-10-11 14:50:50",
            "testcase": "train.InfiniTrain.SFT",
            "success": 0,
            "config": {
                "model_source": "FM9G_70B",
                "operator": "Conv",
                "device": "cuda",
                "output_dir": self.temp_dir
            }
        }

        dispatcher = Dispatcher(self.adapters)
        result = dispatcher.dispatch(payload)

        self.assertEqual(result['total_tests'], 1)
        self.assertEqual(result['successful_tests'], 1)

    def test_with_real_inference_payload(self):
        """Test dispatcher with real inference payload format."""
        payload = {
            "run_id": "infer.infinilm.direct.test",
            "testcase": "infer.InfiniLM.Direct",
            "config": {
                "model": "Qwen3-1.7B",
                "model_path": "/home/fake/model",
                "device": {"gpu_platform": "nvidia"},
                "output_dir": self.temp_dir
            }
        }

        dispatcher = Dispatcher(self.adapters)
        result = dispatcher.dispatch(payload)

        self.assertEqual(result['total_tests'], 1)
        self.assertEqual(result['successful_tests'], 1)

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create multiple test payloads
        payloads = [
            {
                'run_id': 'test1',
                'testcase': 'infer.InfiniLM.Direct',
                'config': {'output_dir': self.temp_dir}
            },
            {
                'run_id': 'test2',
                'testcase': 'train.Operator.Conv',
                'config': {'output_dir': self.temp_dir}
            }
        ]

        dispatcher = Dispatcher(self.adapters)
        all_results = []

        for payload in payloads:
            result = dispatcher.dispatch(payload)
            all_results.append(result)

        # Verify all tests passed
        total_tests = sum(r['total_tests'] for r in all_results)
        successful_tests = sum(r['successful_tests'] for r in all_results)

        self.assertEqual(total_tests, 2)
        self.assertEqual(successful_tests, 2)


class TestDispatcherErrorHandling(unittest.TestCase):
    """Test error handling in Dispatcher."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_with_failing_adapter(self):
        """Test dispatcher handles adapter failures gracefully."""

        class FailingAdapter(BaseAdapter):
            def process(self, payload):
                return {
                    'success': 0,
                    'error': 'Adapter failed'
                }

        adapters = {'inference': FailingAdapter()}
        dispatcher = Dispatcher(adapters)

        payload = {
            'run_id': 'test_fail',
            'testcase': 'infer.Test',
            'config': {'output_dir': self.temp_dir}
        }

        result = dispatcher.dispatch(payload)

        # Should complete but show failure
        self.assertEqual(result['failed_tests'], 1)


if __name__ == '__main__':
    unittest.main()
