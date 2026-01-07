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

from infinimetrics.adapter import BaseAdapter
from infinimetrics.dispatcher import Dispatcher


class MockInferenceAdapter(BaseAdapter):
    """Mock inference adapter for testing."""

    def __init__(self):
        pass

    def process(self, test_input):
        # test_input is TestInput object (has testcase attribute)
        return {
            'success': 0,
            'data': {'type': 'inference', 'testcase': test_input.testcase},
            'metrics': []
        }


class MockOperatorAdapter(BaseAdapter):
    """Mock operator adapter for testing."""

    def __init__(self):
        pass

    def process(self, test_input):
        # test_input is TestInput object (has testcase attribute)
        return {
            'success': 0,
            'data': {'type': 'operator', 'testcase': test_input.testcase},
            'metrics': []
        }


class TestDispatcher(unittest.TestCase):
    """Test cases for Dispatcher."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_dispatcher_initialization(self):
        """Test dispatcher initializes correctly."""
        dispatcher = Dispatcher()

        self.assertIsNotNone(dispatcher)

    def test_dispatch_inference_test(self):
        """Test dispatching inference test."""
        dispatcher = Dispatcher()

        # Mock the _create_adapter method (now takes 2 params: test_type, framework)
        def mock_create_adapter(test_type, framework):
            if test_type == 'inference' and framework == 'infinilm':
                return MockInferenceAdapter()
            elif test_type == 'operator':
                return MockOperatorAdapter()
            raise ValueError(f"Unknown test type or framework: {test_type}, {framework}")

        dispatcher._create_adapter = mock_create_adapter

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
        self.assertEqual(test_result['success'], 0)  # 0 = success
        self.assertEqual(test_result['testcase'], 'infer.InfiniLM.Direct')

    def test_dispatch_operator_test(self):
        """Test dispatching operator test."""
        dispatcher = Dispatcher()

        # Mock the _create_adapter method
        def mock_create_adapter(test_type, framework):
            if test_type == 'inference':
                return MockInferenceAdapter()
            elif test_type == 'operator' and framework == 'infinitrain':
                return MockOperatorAdapter()
            raise ValueError(f"Unknown test type or framework: {test_type}, {framework}")

        dispatcher._create_adapter = mock_create_adapter

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

    def test_parse_testcase_inference(self):
        """Test testcase parsing for inference."""
        dispatcher = Dispatcher()

        test_type, framework = dispatcher._parse_testcase('infer.InfiniLM.Direct')
        self.assertEqual(test_type, 'inference')
        self.assertEqual(framework, 'infinilm')

    def test_parse_testcase_operator(self):
        """Test testcase parsing for operator."""
        dispatcher = Dispatcher()

        test_type, framework = dispatcher._parse_testcase('train.InfiniTrain.SFT')
        self.assertEqual(test_type, 'operator')
        self.assertEqual(framework, 'infinitrain')

    def test_parse_testcase_default(self):
        """Test testcase parsing defaults for invalid format."""
        dispatcher = Dispatcher()

        test_type, framework = dispatcher._parse_testcase('unknown')
        self.assertEqual(test_type, 'operator')
        self.assertEqual(framework, 'operator')

    def test_aggregate_results(self):
        """Test result aggregation."""
        dispatcher = Dispatcher()

        # Lightweight results from Executor (new format)
        # 0 = success, non-zero = failure
        results = [
            {'run_id': 'test1', 'testcase': 'test1', 'success': 0, 'result_file': '/path/to/test1.json'},
            {'run_id': 'test2', 'testcase': 'test2', 'success': 0, 'result_file': '/path/to/test2.json'},
            {'run_id': 'test3', 'testcase': 'test3', 'success': 1, 'result_file': '/path/to/test3.json'}
        ]

        aggregated = dispatcher._aggregate_results(results)

        self.assertEqual(aggregated['total_tests'], 3)
        self.assertEqual(aggregated['successful_tests'], 2)
        self.assertEqual(aggregated['failed_tests'], 1)

        # Check that results contain file references, not full data
        self.assertEqual(len(aggregated['results']), 3)
        self.assertIn('result_file', aggregated['results'][0])
        self.assertIn('success', aggregated['results'][0])
        self.assertIn('testcase', aggregated['results'][0])

    def test_summary_file_creation(self):
        """Test summary file is created."""
        dispatcher = Dispatcher()

        # Mock the _create_adapter method
        dispatcher._create_adapter = lambda test_type, framework: MockInferenceAdapter()

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
        """Test skip when adapter creation fails."""
        dispatcher = Dispatcher()

        # Mock _create_adapter to raise error
        def mock_create_error(test_type, framework):
            raise ValueError(f"Adapter not found for {test_type}/{framework}")

        dispatcher._create_adapter = mock_create_error

        # Request operator test (should be skipped)
        payload = {
            'run_id': 'test_fallback',
            'testcase': 'train.Operator.Test',
            'config': {'output_dir': self.temp_dir}
        }

        # Should skip the test and add to results
        result = dispatcher.dispatch(payload)
        self.assertIsNotNone(result)
        self.assertEqual(result['total_tests'], 1)
        self.assertEqual(result['failed_tests'], 1)

        # Check that the result has skipped flag
        test_result = result['results'][0]
        self.assertEqual(test_result['success'], 1)  # non-zero = failure
        self.assertTrue(test_result.get('skipped', False))


class TestDispatcherIntegration(unittest.TestCase):
    """Integration tests for Dispatcher with real payloads."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

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

        dispatcher = Dispatcher()

        # Mock _create_adapter
        dispatcher._create_adapter = lambda test_type, framework: (
            MockOperatorAdapter() if test_type == 'operator'
            else MockInferenceAdapter() if test_type == 'inference'
            else None
        )

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

        dispatcher = Dispatcher()

        # Mock _create_adapter
        dispatcher._create_adapter = lambda test_type, framework: MockInferenceAdapter()

        result = dispatcher.dispatch(payload)

        self.assertEqual(result['total_tests'], 1)
        self.assertEqual(result['successful_tests'], 1)

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create multiple test payloads (now pass as list to dispatch)
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

        dispatcher = Dispatcher()

        # Mock _create_adapter
        def mock_create(test_type, framework):
            if test_type == 'inference':
                return MockInferenceAdapter()
            elif test_type == 'operator':
                return MockOperatorAdapter()
            raise ValueError(f"Unknown test type: {test_type}, framework: {framework}")

        dispatcher._create_adapter = mock_create

        # Pass all payloads at once (batch mode)
        result = dispatcher.dispatch(payloads)

        # Verify all tests passed in single batch
        self.assertEqual(result['total_tests'], 2)
        self.assertEqual(result['successful_tests'], 2)


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
            def __init__(self):
                pass

            def process(self, test_input):
                return {
                    'success': 1,  # 1 = failure
                    'data': {'error': 'Adapter failed'},
                    'metrics': []
                }

        dispatcher = Dispatcher()

        # Mock _create_adapter to return failing adapter
        dispatcher._create_adapter = lambda test_type, framework: FailingAdapter()

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
