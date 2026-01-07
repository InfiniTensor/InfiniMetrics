#!/usr/bin/env python3
"""
Unit tests for adapter.py
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infinimetrics.adapter import BaseAdapter


class MockAdapter(BaseAdapter):
    """Mock adapter for testing."""

    def __init__(self):
        self.setup_called = False
        self.teardown_called = False

    def process(self, payload):
        """Mock process implementation."""
        return {
            'success': 1,
            'data': {'result': 'mock_result'},
            'metrics': []
        }

    def setup(self, config=None):
        """Mock setup."""
        self.setup_called = True

    def teardown(self):
        """Mock teardown."""
        self.teardown_called = True


class TestBaseAdapter(unittest.TestCase):
    """Test cases for BaseAdapter."""

    def test_adapter_is_abstract(self):
        """Test that BaseAdapter cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseAdapter()

    def test_mock_adapter_creation(self):
        """Test that mock adapter can be created."""
        adapter = MockAdapter()
        self.assertIsNotNone(adapter)

    def test_process_method(self):
        """Test process method returns correct format."""
        adapter = MockAdapter()
        payload = {
            'testcase': 'test.Mock.Test',
            'config': {}
        }

        result = adapter.process(payload)

        self.assertIn('success', result)
        self.assertIn('data', result)
        self.assertEqual(result['success'], 1)
        self.assertEqual(result['data']['result'], 'mock_result')

    def test_setup_method(self):
        """Test setup method is callable."""
        adapter = MockAdapter()
        adapter.setup()
        self.assertTrue(adapter.setup_called)

    def test_teardown_method(self):
        """Test teardown method is callable."""
        adapter = MockAdapter()
        adapter.teardown()
        self.assertTrue(adapter.teardown_called)

    def test_validate_method(self):
        """Test validate method returns empty list by default."""
        adapter = MockAdapter()
        errors = adapter.validate()
        self.assertEqual(errors, [])

    def test_get_info_method(self):
        """Test get_info returns correct structure."""
        adapter = MockAdapter()
        info = adapter.get_info()

        self.assertIn('name', info)
        self.assertIn('version', info)
        self.assertIn('supported_operations', info)
        self.assertIn('framework', info)
        self.assertEqual(info['name'], 'MockAdapter')
        self.assertEqual(info['version'], '1.0')

    def test_get_supported_operations(self):
        """Test get_supported_operations returns default list."""
        adapter = MockAdapter()
        ops = adapter._get_supported_operations()

        self.assertIsInstance(ops, list)
        self.assertIn('inference', ops)
        self.assertIn('operator', ops)
        self.assertIn('training', ops)

    def test_process_with_error_payload(self):
        """Test process with payload that causes error."""
        class ErrorAdapter(BaseAdapter):
            def process(self, payload):
                return {
                    'success': 0,
                    'error': 'Test error message'
                }

        adapter = ErrorAdapter()
        result = adapter.process({'testcase': 'test'})

        self.assertEqual(result['success'], 0)
        self.assertIn('error', result)


class TestAdapterIntegration(unittest.TestCase):
    """Integration tests for adapter usage patterns."""

    def test_full_lifecycle(self):
        """Test complete adapter lifecycle: setup -> process -> teardown."""
        adapter = MockAdapter()

        # Setup
        adapter.setup({'model_path': '/fake/path'})
        self.assertTrue(adapter.setup_called)

        # Process
        payload = {
            'testcase': 'test.integration',
            'config': {'param': 'value'}
        }
        result = adapter.process(payload)
        self.assertEqual(result['success'], 1)

        # Teardown
        adapter.teardown()
        self.assertTrue(adapter.teardown_called)

    def test_process_with_metrics(self):
        """Test process returns metrics."""
        class MetricsAdapter(BaseAdapter):
            def process(self, payload):
                return {
                    'success': 1,
                    'data': {},
                    'metrics': [
                        {'name': 'latency', 'value': 100, 'unit': 'ms'},
                        {'name': 'throughput', 'value': 50, 'unit': 'tokens/s'}
                    ]
                }

        adapter = MetricsAdapter()
        result = adapter.process({'testcase': 'test'})

        self.assertEqual(result['success'], 1)
        self.assertEqual(len(result['metrics']), 2)


if __name__ == '__main__':
    unittest.main()
