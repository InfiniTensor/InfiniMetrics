#!/usr/bin/env python3
"""Base Adapter - Unified Interface for All Test Types"""

import abc
from typing import Dict, Any


class BaseAdapter(abc.ABC):
    """
    Unified adapter base class for all test types (inference, operator, training).

    Lifecycle:
        1. __init__() - Create adapter (in Dispatcher)
        2. setup(config) - Initialize resources (in Executor)
        3. process(test_input) - Execute test
        4. teardown() - Cleanup resources
    """

    @abc.abstractmethod
    def process(self, test_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the test.

        Args:
            test_input: TestInput dict with testcase, config, etc.

        Returns:
            Dict with:
                - 'success': int (0 = success, non-zero = failure)
                - 'metrics': list (performance metrics)
        """
        pass

    def setup(self, config: Dict[str, Any]) -> None:
        """
        Initialize resources before running tests.

        Called by Executor before process().

        Args:
            config: Configuration dict from test_input.config
        """
        pass

    def teardown(self) -> None:
        """
        Cleanup resources after tests complete.

        Called by Executor after process() (even if process() fails).
        """
        pass
