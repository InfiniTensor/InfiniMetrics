#!/usr/bin/env python3
"""
Unified Data Type Handler for All Adapters

Provides standardized dtype handling, parsing, and size calculations.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DtypeHandler:
    """
    Unified data type handler for all adapters.

    Provides consistent dtype normalization, size calculation, and
    validation across different frameworks and backends.
    """

    # Standard dtype to byte size mapping
    DTYPE_SIZE: Dict[str, int] = {
        # Float types
        "float32": 4,
        "float64": 8,
        "float16": 2,
        "bfloat16": 2,
        "float": 4,  # Default to float32
        "double": 8,  # Alias for float64

        # Integer types
        "int32": 4,
        "int64": 8,
        "int16": 2,
        "int8": 1,
        "int": 4,  # Default to int32

        # Unsigned integer types
        "uint32": 4,
        "uint64": 8,
        "uint16": 2,
        "uint8": 1,

        # Boolean
        "bool": 1,

        # Complex types
        "complex64": 8,
        "complex128": 16,
    }

    # Common dtype name aliases
    DTYPE_ALIASES: Dict[str, str] = {
        # Float aliases
        "fp32": "float32",
        "fp64": "float64",
        "fp16": "float16",
        "bf16": "bfloat16",
        "half": "float16",

        # Integer aliases
        "i32": "int32",
        "i64": "int64",
        "i16": "int16",
        "i8": "int8",
        "u32": "uint32",
        "u64": "uint64",
        "u16": "uint16",
        "u8": "uint8",
        "long": "int64",
        "byte": "int8",
    }

    @classmethod
    def normalize_dtype(cls, dtype_str: str) -> str:
        """
        Normalize dtype string to standard form.

        Args:
            dtype_str: Dtype string (e.g., "fp32", "FP16", "Float32")

        Returns:
            Normalized dtype name in lowercase (e.g., "float32", "float16")

        Examples:
            >>> DtypeHandler.normalize_dtype("fp32")
            'float32'
            >>> DtypeHandler.normalize_dtype("BF16")
            'bfloat16'
            >>> DtypeHandler.normalize_dtype("i64")
            'int64'
        """
        if not dtype_str:
            return "float32"  # Default

        normalized = dtype_str.lower().strip()

        # Check against aliases
        if normalized in cls.DTYPE_ALIASES:
            return cls.DTYPE_ALIASES[normalized]

        # Return as-is if already standard
        return normalized

    @classmethod
    def get_dtype_bytes(cls, dtype_str: str) -> int:
        """
        Get the size in bytes of a dtype.

        Args:
            dtype_str: Dtype string

        Returns:
            Size in bytes (defaults to 4 for unknown types)

        Examples:
            >>> DtypeHandler.get_dtype_bytes("float32")
            4
            >>> DtypeHandler.get_dtype_bytes("int64")
            8
            >>> DtypeHandler.get_dtype_bytes("fp16")
            2
        """
        normalized = cls.normalize_dtype(dtype_str)

        # Handle common variations by checking if key is in dtype name
        for standard_name, size in cls.DTYPE_SIZE.items():
            if standard_name in normalized:
                return size

        # Default to 4 bytes (float32/int32)
        logger.warning(f"Unknown dtype: {dtype_str}, defaulting to 4 bytes")
        return 4

    @classmethod
    def is_float_dtype(cls, dtype_str: str) -> bool:
        """
        Check if dtype is a floating-point type.

        Args:
            dtype_str: Dtype string

        Returns:
            True if dtype is float/fp/bfloat type

        Examples:
            >>> DtypeHandler.is_float_dtype("float32")
            True
            >>> DtypeHandler.is_float_dtype("int32")
            False
        """
        normalized = cls.normalize_dtype(dtype_str)
        return "float" in normalized or "fp" in normalized or "bf" in normalized

    @classmethod
    def is_int_dtype(cls, dtype_str: str) -> bool:
        """
        Check if dtype is an integer type.

        Args:
            dtype_str: Dtype string

        Returns:
            True if dtype is int/uint type

        Examples:
            >>> DtypeHandler.is_int_dtype("int32")
            True
            >>> DtypeHandler.is_int_dtype("uint8")
            True
            >>> DtypeHandler.is_int_dtype("float32")
            False
        """
        normalized = cls.normalize_dtype(dtype_str)
        return "int" in normalized or "uint" in normalized or "i" == normalized[0] or "u" == normalized[0]

    @classmethod
    def is_bool_dtype(cls, dtype_str: str) -> bool:
        """
        Check if dtype is boolean type.

        Args:
            dtype_str: Dtype string

        Returns:
            True if dtype is bool
        """
        normalized = cls.normalize_dtype(dtype_str)
        return normalized == "bool"

    @classmethod
    def validate_dtype(cls, dtype_str: str) -> bool:
        """
        Validate if dtype string is recognized.

        Args:
            dtype_str: Dtype string

        Returns:
            True if dtype is known/valid

        Examples:
            >>> DtypeHandler.validate_dtype("float32")
            True
            >>> DtypeHandler.validate_dtype("unknown_type")
            False
        """
        try:
            # Try to get size - will return default for unknown
            size = cls.get_dtype_bytes(dtype_str)
            # If we found it in mapping (not default), it's valid
            normalized = cls.normalize_dtype(dtype_str)
            return normalized in cls.DTYPE_SIZE or any(name in normalized for name in cls.DTYPE_SIZE.keys())
        except Exception:
            return False

    @classmethod
    def get_common_dtypes(cls) -> list[str]:
        """
        Get list of commonly used dtypes.

        Returns:
            List of standard dtype names
        """
        return [
            "float32",
            "float16",
            "bfloat16",
            "int32",
            "int64",
            "int8",
            "uint8",
            "bool",
        ]

    @classmethod
    def calculate_tensor_size(
        cls,
        shape: tuple[int, ...],
        dtype_str: str
    ) -> int:
        """
        Calculate total size of a tensor in bytes.

        Args:
            shape: Tensor shape (e.g., (1024, 1024))
            dtype_str: Dtype string

        Returns:
            Size in bytes

        Examples:
            >>> DtypeHandler.calculate_tensor_size((1024, 1024), "float32")
            4194304
            >>> DtypeHandler.calculate_tensor_size((100,), "int64")
            800
        """
        import math
        dtype_bytes = cls.get_dtype_bytes(dtype_str)

        if not shape:
            return 0

        num_elements = math.prod(shape) if hasattr(math, 'prod') else 1
        # Fallback for older Python versions
        if not hasattr(math, 'prod'):
            num_elements = 1
            for dim in shape:
                num_elements *= dim

        return num_elements * dtype_bytes

    @classmethod
    def format_dtype(cls, dtype_str: str, framework: str = "numpy") -> str:
        """
        Format dtype string for specific framework.

        Args:
            dtype_str: Generic dtype string
            framework: Target framework ("numpy", "torch", "tensorflow")

        Returns:
            Framework-specific dtype string

        Examples:
            >>> DtypeHandler.format_dtype("float32", "torch")
            'torch.float32'
            >>> DtypeHandler.format_dtype("int64", "numpy")
            'int64'
        """
        normalized = cls.normalize_dtype(dtype_str)

        if framework == "torch":
            # PyTorch style
            torch_dtypes = {
                "float32": "torch.float32",
                "float64": "torch.float64",
                "float16": "torch.float16",
                "bfloat16": "torch.bfloat16",
                "int32": "torch.int32",
                "int64": "torch.int64",
                "int16": "torch.int16",
                "int8": "torch.int8",
                "uint8": "torch.uint8",
                "bool": "torch.bool",
            }
            return torch_dtypes.get(normalized, f"torch.{normalized}")

        elif framework == "tensorflow":
            # TensorFlow style
            tf_dtypes = {
                "float32": "tf.float32",
                "float64": "tf.float64",
                "float16": "tf.float16",
                "int32": "tf.int32",
                "int64": "tf.int64",
                "int16": "tf.int16",
                "int8": "tf.int8",
                "uint8": "tf.uint8",
                "bool": "tf.bool",
            }
            return tf_dtypes.get(normalized, f"tf.{normalized}")

        else:
            # Default (numpy style)
            return normalized
