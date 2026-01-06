#!/usr/bin/env python3
"""
Base Adapter - Unified Interface for All Test Types
"""

import abc
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class BaseAdapter(abc.ABC):
    """
    Unified adapter base class for all test types (inference, operator, training).

    The adapter receives the complete test payload and processes it based on testcase type.

    Payload Format:
        {
            'run_id': str,           # Unique run identifier
            'testcase': str,         # Test case name (e.g., 'train.InfiniTrain.SFT')
            'config': dict,          # Test configuration
            'metrics': list,         # Optional: list of metrics to collect
            'success': int           # Optional: success flag (0 or 1)
        }
    """

    @abc.abstractmethod
    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core processing method - handles the complete test payload.

        Args:
            payload: Complete test payload with testcase, config, metrics, etc.

        Returns:
            Dict with 'success' (0 or 1), 'data' (result data), and optional 'error'
        """
        pass

    def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Optional: Initialize resources (e.g., load models).

        Args:
            config: Configuration dict
        """
        pass

    def teardown(self) -> None:
        """Optional: Cleanup resources."""
        pass

    def validate(self) -> List[str]:
        """
        Validate adapter configuration and state.

        Returns:
            List of error messages (empty if validation passed)
        """
        return []

    def get_info(self) -> Dict[str, Any]:
        """
        Get adapter information.

        Returns:
            Dict with name, version, supported_operations, framework
        """
        return {
            'name': self.__class__.__name__,
            'version': '1.0',
            'supported_operations': self._get_supported_operations(),
            'framework': self.__class__.__name__.replace('Adapter', '').lower()
        }

    def _get_supported_operations(self) -> List[str]:
        """Get list of supported operations. Subclasses can override."""
        return ['inference', 'operator', 'training']
