import unittest
import sys
import os

# --- Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import module to be tested
from infinimetrics.tools.benchmark.op_library_mock import benchmark_operator

class TestOpLibraryMock(unittest.TestCase):
    """
    Tests corresponding to op_library_mock.py
    """

    def test_benchmark_interface_contract(self):
        """
        Interface Contract Test: Verify that the returned data structure contains necessary fields.
        This is crucial for future replacement with a real library.
        """
        result = benchmark_operator(
            op_type="Conv2d", 
            fqn_name="layer1.0.conv1", 
            input_shape=[1, 64, 56, 56]
        )
        
        # Must be a dictionary
        self.assertIsInstance(result, dict)
        # Must contain these two keys
        self.assertIn("status", result)
        self.assertIn("actual_latency_ms", result)
        # Status must be SUCCESS (default mock library behavior)
        self.assertEqual(result["status"], "SUCCESS")
        # Latency must be a float
        self.assertIsInstance(result["actual_latency_ms"], float)

    def test_latency_simulation_logic(self):
        """Verify if simulated latency values are within a reasonable range"""
        for _ in range(5):
            result = benchmark_operator("Test", "test_op", [])
            ms = result["actual_latency_ms"]
            # We defined 0.01s~0.1s in code, i.e., 10ms~100ms
            self.assertTrue(10 <= ms <= 100, f"Latency {ms}ms is unexpectedly out of range")

if __name__ == '__main__':
    unittest.main()
