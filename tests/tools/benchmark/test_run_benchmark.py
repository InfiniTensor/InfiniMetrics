import unittest
import sys
import os

# --- Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from infinimetrics.tools.benchmark.run_benchmark import run_operator_tests

class TestRunBenchmark(unittest.TestCase):
    """
    Tests corresponding to run_benchmark.py (Integration Test)
    """

    def setUp(self):
        """Preparation data before tests"""
        self.sample_cases = [
            {
                "fully_qualified_name": "test.layer1",
                "op_type": "Linear",
                "input_shape": [1, 128],
                "original_data": "keep_me" # Verify if original data is preserved
            },
            {
                "fully_qualified_name": "test.layer2",
                "op_type": "Conv2d",
                "input_shape": [1, 3, 224, 224]
            }
        ]

    def test_run_operator_tests_success(self):
        """Test if main flow correctly merges results"""
        # Run the function under test
        results = run_operator_tests(self.sample_cases)

        # 1. Verify returned list length
        self.assertEqual(len(results), 2)

        # 2. Verify result merging logic
        first_result = results[0]
        
        # Check if original data still exists
        self.assertEqual(first_result["fully_qualified_name"], "test.layer1")
        self.assertEqual(first_result["original_data"], "keep_me")
        
        # Check if new data is added
        self.assertIn("actual_latency_ms", first_result)
        self.assertIn("status", first_result)
        self.assertEqual(first_result["status"], "SUCCESS")
        self.assertIsInstance(first_result["actual_latency_ms"], float)

if __name__ == '__main__':
    unittest.main()
