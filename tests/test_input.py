#!/usr/bin/env python3
"""
Unit tests for input.py
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infinimetrics.input import TestInput


class TestTestInput(unittest.TestCase):
    """Test cases for TestInput class."""

    def test_minimal_input(self):
        """Test creating TestInput with only required field."""
        test_input = TestInput(testcase="infer.InfiniLM.Direct")

        self.assertEqual(test_input.testcase, "infer.InfiniLM.Direct")
        self.assertIsNone(test_input.run_id)
        self.assertIsNotNone(test_input.time)
        self.assertEqual(test_input.config, {})
        self.assertEqual(test_input.metrics, [])

    def test_full_input(self):
        """Test creating TestInput with all fields."""
        test_input = TestInput(
            testcase="train.InfiniTrain.SFT",
            run_id="test.run.123",
            time="2025-10-11 14:50:50",
            success=0,
            config={"operator": "Conv", "device": "cuda"},
            metrics=[{"name": "latency", "unit": "ms"}]
        )

        self.assertEqual(test_input.testcase, "train.InfiniTrain.SFT")
        self.assertEqual(test_input.run_id, "test.run.123")
        self.assertEqual(test_input.time, "2025-10-11 14:50:50")
        self.assertEqual(test_input.success, 0)
        self.assertEqual(test_input.config["operator"], "Conv")
        self.assertEqual(len(test_input.metrics), 1)

    def test_to_dict_minimal(self):
        """Test converting minimal TestInput to dict."""
        test_input = TestInput(testcase="infer.Test")
        result = test_input.to_dict()

        self.assertEqual(result['testcase'], "infer.Test")
        self.assertIn('config', result)
        self.assertIn('time', result)
        self.assertNotIn('run_id', result)
        self.assertNotIn('success', result)
        self.assertNotIn('metrics', result)

    def test_to_dict_full(self):
        """Test converting full TestInput to dict."""
        test_input = TestInput(
            testcase="infer.Test",
            run_id="test.123",
            success=0,
            config={"key": "value"},
            metrics=[{"name": "metric1"}]
        )
        result = test_input.to_dict()

        self.assertEqual(result['testcase'], "infer.Test")
        self.assertEqual(result['run_id'], "test.123")
        self.assertEqual(result['success'], 0)
        self.assertEqual(result['config']['key'], "value")
        self.assertEqual(len(result['metrics']), 1)

    def test_from_dict_operator_input(self):
        """Test creating TestInput from operator input dict."""
        data = {
            "run_id": "train.infiniTrain.SFT.a8b4c9e1",
            "time": "2025-10-11 14:50:50",
            "testcase": "train.InfiniTrain.SFT",
            "success": 0,
            "config": {
                "model_source": "FM9G_70B",
                "operator": "Conv",
                "device": "cuda"
            },
            "metrics": [
                {"name": "operator.latency", "unit": "ms"}
            ]
        }

        test_input = TestInput.from_dict(data)

        self.assertEqual(test_input.testcase, "train.InfiniTrain.SFT")
        self.assertEqual(test_input.run_id, "train.infiniTrain.SFT.a8b4c9e1")
        self.assertEqual(test_input.config["operator"], "Conv")
        self.assertEqual(len(test_input.metrics), 1)

    def test_from_dict_inference_input(self):
        """Test creating TestInput from inference input dict."""
        data = {
            "run_id": "infer.infinilm.direct.test",
            "testcase": "infer.InfiniLM.Direct",
            "config": {
                "model": "Qwen3-1.7B",
                "model_path": "/home/model/Qwen3-1.7B",
                "device": {"gpu_platform": "nvidia"}
            }
        }

        test_input = TestInput.from_dict(data)

        self.assertEqual(test_input.testcase, "infer.InfiniLM.Direct")
        self.assertEqual(test_input.config["model"], "Qwen3-1.7B")
        self.assertIsNotNone(test_input.time)  # Auto-generated in __post_init__

    def test_invalid_testcase(self):
        """Test that empty testcase raises error."""
        with self.assertRaises(ValueError):
            TestInput(testcase="")

        with self.assertRaises(ValueError):
            TestInput(testcase=None)

    def test_config_helpers(self):
        """Test config getter/setter methods."""
        test_input = TestInput(
            testcase="test",
            config={"existing": "value"}
        )

        # Test get_config_value
        self.assertEqual(test_input.get_config_value("existing"), "value")
        self.assertEqual(test_input.get_config_value("missing", "default"), "default")
        self.assertIsNone(test_input.get_config_value("missing"))

        # Test set_config_value
        test_input.set_config_value("new_key", "new_value")
        self.assertEqual(test_input.get_config_value("new_key"), "new_value")

    def test_round_trip(self):
        """Test to_dict and from_dict round trip."""
        original = TestInput(
            testcase="infer.Test",
            run_id="test.123",
            config={"operator": "Conv"},
            metrics=[{"name": "latency"}]
        )

        # Convert to dict
        data_dict = original.to_dict()

        # Convert back to TestInput
        restored = TestInput.from_dict(data_dict)

        # Verify
        self.assertEqual(restored.testcase, original.testcase)
        self.assertEqual(restored.run_id, original.run_id)
        self.assertEqual(restored.config["operator"], "Conv")
        self.assertEqual(len(restored.metrics), 1)


if __name__ == '__main__':
    unittest.main()
