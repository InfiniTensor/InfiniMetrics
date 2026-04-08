import unittest
import sys
import os

# --- Path Configuration (Boilerplate) ---
# Allow running this script directly or via python -m unittest
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import module to be tested
from infinimetrics.tools.benchmark.parse_summary import parse_shape_string, parse_number_string

class TestModelParser(unittest.TestCase):
    """
    Tests corresponding to parse_summary.py
    """

    def test_parse_shape_string_valid(self):
        """Test valid shape string parsing"""
        self.assertEqual(parse_shape_string('[80, 64, 768]'), [80, 64, 768])
        self.assertEqual(parse_shape_string('[1, 128]'), [1, 128])

    def test_parse_shape_string_edge_cases(self):
        """Test edge cases for shape parsing"""
        self.assertIsNone(parse_shape_string('invalid'))
        self.assertIsNone(parse_shape_string('[1, 2')) # Missing closing bracket
        self.assertEqual(parse_shape_string('[]'), [])   # Empty list

    def test_parse_number_string(self):
        """Test number string parsing"""
        self.assertEqual(parse_number_string('3,087,790'), 3087790)
        self.assertEqual(parse_number_string('100'), 100)
        # Test special placeholders
        self.assertEqual(parse_number_string('--'), 0)
        self.assertEqual(parse_number_string(' -- '), 0)

if __name__ == '__main__':
    unittest.main()
