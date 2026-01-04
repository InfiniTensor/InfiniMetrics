#!/usr/bin/env python3
"""
Unified Validation Mixin for All Adapters

Provides common validation methods that can be mixed into any adapter class.
"""

import logging
from typing import List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationMixin:
    """
    Mixin class providing common validation methods for adapters.

    This class can be mixed into any adapter to gain standardized validation
    capabilities. All methods return a list of error messages (empty if valid).

    Examples:
        >>> class MyAdapter(ValidationMixin):
        ...     def validate_config(self):
        ...         errors = []
        ...         errors.extend(self.validate_file_exists("/path/to/file"))
        ...         errors.extend(self.validate_positive_number(10, "batch_size"))
        ...         return errors
    """

    def validate_file_exists(
        self,
        file_path: str,
        error_msg: Optional[str] = None,
        file_type: str = "File"
    ) -> List[str]:
        """
        Validate that a file or directory exists.

        Args:
            file_path: Path to the file or directory
            error_msg: Custom error message (optional)
            file_type: Type description for default error message

        Returns:
            List of error messages (empty if file exists)

        Examples:
            >>> adapter.validate_file_exists("/path/to/model")
            []
            >>> adapter.validate_file_exists("/nonexistent")
            ["File does not exist: /nonexistent"]
        """
        errors = []
        path = Path(file_path)

        if not path.exists():
            msg = error_msg or f"{file_type} does not exist: {file_path}"
            errors.append(msg)
            logger.error(msg)

        return errors

    def validate_positive_number(
        self,
        value: Union[int, float],
        name: str,
        allow_zero: bool = False
    ) -> List[str]:
        """
        Validate that a number is positive (or non-negative).

        Args:
            value: Number to validate
            name: Parameter name for error message
            allow_zero: Whether to allow zero as valid value

        Returns:
            List of error messages (empty if valid)

        Examples:
            >>> adapter.validate_positive_number(10, "batch_size")
            []
            >>> adapter.validate_positive_number(-1, "batch_size")
            ["batch_size must be positive, got: -1"]
        """
        errors = []

        if value < 0 or (not allow_zero and value == 0):
            requirement = "non-negative" if allow_zero else "positive"
            msg = f"{name} must be {requirement}, got: {value}"
            errors.append(msg)
            logger.error(msg)

        return errors

    def validate_in_range(
        self,
        value: float,
        min_val: float,
        max_val: float,
        name: str,
        inclusive: bool = True
    ) -> List[str]:
        """
        Validate that a value is within a specified range.

        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            name: Parameter name for error message
            inclusive: Whether to include boundaries in valid range

        Returns:
            List of error messages (empty if valid)

        Examples:
            >>> adapter.validate_in_range(0.5, 0.0, 1.0, "temperature")
            []
            >>> adapter.validate_in_range(1.5, 0.0, 1.0, "temperature")
            ["temperature must be between 0.0 and 1.0, got: 1.5"]
        """
        errors = []

        if inclusive:
            is_valid = min_val <= value <= max_val
        else:
            is_valid = min_val < value < max_val

        if not is_valid:
            range_str = f"[{min_val}, {max_val}]" if inclusive else f"({min_val}, {max_val})"
            msg = f"{name} must be in range {range_str}, got: {value}"
            errors.append(msg)
            logger.error(msg)

        return errors

    def validate_dependencies_available(
        self,
        available: bool,
        dependency_name: str,
        install_hint: Optional[str] = None
    ) -> List[str]:
        """
        Validate that required dependencies are available.

        Args:
            available: Whether dependencies are available
            dependency_name: Name of the dependency/module
            install_hint: Optional installation instructions

        Returns:
            List of error messages (empty if available)

        Examples:
            >>> adapter.validate_dependencies_available(
            ...     False,
            ...     "InfiniLM modules",
            ...     "Install from: https://github.com/InfiniTensor/InfiniLM"
            ... )
            ["InfiniLM modules are not available. Install from: ..."]
        """
        errors = []

        if not available:
            msg = f"{dependency_name} are not available"
            if install_hint:
                msg += f". {install_hint}"
            errors.append(msg)
            logger.error(msg)

        return errors

    def validate_not_empty(
        self,
        value: Any,
        name: str,
        allow_zero: bool = False
    ) -> List[str]:
        """
        Validate that a value is not empty.

        Args:
            value: Value to validate (can be string, list, dict, etc.)
            name: Parameter name for error message
            allow_zero: Whether zero is considered non-empty for numbers

        Returns:
            List of error messages (empty if not empty)
        """
        errors = []

        is_empty = False
        if value is None:
            is_empty = True
        elif isinstance(value, (str, list, dict, tuple, set)):
            is_empty = len(value) == 0
        elif isinstance(value, (int, float)):
            is_empty = value == 0 and not allow_zero

        if is_empty:
            msg = f"{name} cannot be empty"
            errors.append(msg)
            logger.error(msg)

        return errors

    def validate_one_of(
        self,
        value: Any,
        allowed_values: List[Any],
        name: str
    ) -> List[str]:
        """
        Validate that a value is one of the allowed values.

        Args:
            value: Value to validate
            allowed_values: List of allowed values
            name: Parameter name for error message

        Returns:
            List of error messages (empty if valid)

        Examples:
            >>> adapter.validate_one_of("nvidia", ["nvidia", "amd", "cpu"], "accelerator")
            []
            >>> adapter.validate_one_of("unknown", ["nvidia", "amd"], "accelerator")
            ["accelerator must be one of ['nvidia', 'amd', 'cpu'], got: unknown"]
        """
        errors = []

        if value not in allowed_values:
            msg = f"{name} must be one of {allowed_values}, got: {value}"
            errors.append(msg)
            logger.error(msg)

        return errors

    def validate_writable_directory(
        self,
        dir_path: str,
        create_if_missing: bool = False
    ) -> List[str]:
        """
        Validate that a directory is writable.

        Args:
            dir_path: Path to the directory
            create_if_missing: Whether to create directory if it doesn't exist

        Returns:
            List of error messages (empty if writable)

        Examples:
            >>> adapter.validate_writable_directory("/tmp/output", create_if_missing=True)
            []
        """
        errors = []
        path = Path(dir_path)

        # Create directory if requested
        if create_if_missing and not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            except Exception as e:
                msg = f"Failed to create directory {dir_path}: {e}"
                errors.append(msg)
                logger.error(msg)
                return errors

        # Check if directory exists
        if not path.exists():
            msg = f"Directory does not exist: {dir_path}"
            errors.append(msg)
            logger.error(msg)
            return errors

        # Check if it's writable
        try:
            test_file = path / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            msg = f"Directory is not writable: {dir_path} - {e}"
            errors.append(msg)
            logger.error(msg)

        return errors

    def validate_all(self, *validation_results: List[str]) -> List[str]:
        """
        Combine multiple validation results into a single list.

        Convenience method to combine validation results from multiple calls.

        Args:
            *validation_results: Variable number of validation result lists

        Returns:
            Combined list of all error messages

        Examples:
            >>> errors = adapter.validate_all(
            ...     adapter.validate_file_exists("/path"),
            ...     adapter.validate_positive_number(10, "batch_size")
            ... )
        """
        all_errors = []
        for errors in validation_results:
            all_errors.extend(errors)
        return all_errors
