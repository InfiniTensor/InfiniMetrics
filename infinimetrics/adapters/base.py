#!/usr/bin/env python3
"""
Unified Base Adapter for all framework adapters
Supports both stateless and stateful patterns
"""

import abc
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseAdapter(abc.ABC):
    """
    Unified base class for all adapters.

    Supports two patterns:
    1. Stateless: Only implement process()
    2. Stateful: Implement setup() + process() + teardown()

    Lifecycle:
        - setup(config): Optional initialization
        - process(request): MUST be implemented
        - teardown(): Optional cleanup
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize adapter.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config
        self._setup_complete = False
        logger.debug(f"{self.__class__.__name__} initialized")

    @abc.abstractmethod
    def process(self, request: Any) -> Dict[str, Any]:
        """
        Process a request - MUST be implemented by subclasses.

        Args:
            request: Request object (dict or custom type)

        Returns:
            Dict containing results with at least:
            - 'success': bool
            - 'data': Any (optional)
            - 'error': str (optional, if failed)
        """
        pass

    def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Optional setup hook for stateful adapters.
        Override this if your adapter needs initialization.

        Args:
            config: Optional config override
        """
        if config:
            self.config = config
        self._setup_complete = True
        logger.debug(f"{self.__class__.__name__} setup complete")

    def teardown(self) -> None:
        """
        Optional teardown hook for stateful adapters.
        Override this if your adapter needs cleanup.
        """
        self._setup_complete = False
        logger.debug(f"{self.__class__.__name__} teardown complete")

    def is_setup(self) -> bool:
        """
        Check if adapter has been setup.

        Returns:
            True if setup() has been called
        """
        return self._setup_complete

    def ensure_setup(self) -> None:
        """
        Ensure adapter is setup before processing.
        Raises RuntimeError if not setup.
        """
        if not self.is_setup():
            raise RuntimeError(
                f"{self.__class__.__name__} must be setup() before process(). "
                f"Call setup() or use setup() + process() + teardown() pattern."
            )

    def validate(self) -> List[str]:
        """
        Optional validation hook.
        Override to provide custom validation logic.

        Returns:
            List of error messages (empty if valid)
        """
        return []

    def process_with_validation(self, request: Any) -> Dict[str, Any]:
        """
        Process request with automatic validation.

        Args:
            request: Request to process

        Returns:
            Result dict with validation errors if any
        """
        errors = self.validate()
        if errors:
            return {
                'success': False,
                'error': f"Validation failed: {'; '.join(errors)}",
                'validation_errors': errors
            }

        return self.process(request)
