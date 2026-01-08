#!/usr/bin/env python3
"""InfiniCore Operator Adapter"""

import logging
from datetime import datetime
from typing import Any, Dict

from infinimetrics.adapter import BaseAdapter

logger = logging.getLogger(__name__)


class InfiniCoreAdapter(BaseAdapter):
    """Adapter for InfiniCore operator tests (Conv, MatMul, etc.)."""

    def process(self, test_input: Any) -> Dict[str, Any]:
        """
        Execute the operator test.

        Args:
            test_input: TestInput object or dict with testcase, config, metrics, etc.

        Returns:
            Dict with success, metrics, time, and error_msg if failed
        """
        # Convert TestInput object to dict if needed
        if hasattr(test_input, "to_dict"):
            test_input = test_input.to_dict()
        elif not isinstance(test_input, dict):
            # Test fails directly
            return {
                "success": 1,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error_msg": f"Invalid test_input type: {type(test_input)}",
                "metrics": [],
            }

        testcase = test_input.get("testcase", "unknown")
        logger.info(f"InfiniCoreAdapter: Processing {testcase}")

        # Mock response - all data is fake
        return {
            "success": 0,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": [
                {
                    "name": "operator.latency",
                    "value": 19.2,
                    "type": "scalar",
                    "unit": "ms",
                },
                {"name": "operator.tensor_accuracy", "value": "PASS"},
            ],
        }
