#!/usr/bin/env python3
"""
Test runner script for InfiniMetrics framework tests.
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_tests(verbosity=2):
    """Run all tests."""
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    # Return exit code based on success
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    verbosity = 2
    if len(sys.argv) > 1:
        if sys.argv[1] == '-v' or sys.argv[1] == '--verbose':
            verbosity = 2
        elif sys.argv[1] == '-q' or sys.argv[1] == '--quiet':
            verbosity = 0

    exit_code = run_tests(verbosity)
    sys.exit(exit_code)
