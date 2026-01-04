#!/usr/bin/env python3
"""
Unified Device Handler for All Adapters

Provides standardized device type handling across InfiniCore and InfiniLM adapters.
"""

import logging
from typing import Dict, Optional, Union, Type
from enum import Enum

logger = logging.getLogger(__name__)


class StandardDeviceType(Enum):
    """Standardized device type enumeration"""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    CPU = "cpu"
    AUTO = "auto"


class DeviceHandler:
    """
    Unified device type handler for all adapters.

    Provides consistent device name parsing, normalization, and conversion
    to framework-specific device types.
    """

    # Standard device name aliases for flexible matching
    DEVICE_ALIASES: Dict[str, list[str]] = {
        "nvidia": ["nvidia", "cuda", "gpu", "nvidia_gpu"],
        "amd": ["amd", "rocm", "amd_gpu"],
        "intel": ["intel", "xpu", "intel_gpu"],
        "cpu": ["cpu"],
    }

    @classmethod
    def normalize_device_name(cls, device_str: str) -> str:
        """
        Normalize device name to standard form.

        Args:
            device_str: Device string (e.g., "NVIDIA", "cuda", "GPU")

        Returns:
            Normalized device name in lowercase (e.g., "nvidia", "cpu")

        Examples:
            >>> DeviceHandler.normalize_device_name("CUDA")
            'nvidia'
            >>> DeviceHandler.normalize_device_name("AMD")
            'amd'
            >>> DeviceHandler.normalize_device_name("cpu")
            'cpu'
        """
        if not device_str:
            return "cpu"

        device_lower = device_str.lower().strip()

        # Check against aliases
        for standard_name, aliases in cls.DEVICE_ALIASES.items():
            if device_lower in [a.lower() for a in aliases]:
                return standard_name

        logger.warning(f"Unknown device: {device_str}, defaulting to CPU")
        return "cpu"

    @classmethod
    def parse_device(cls, device_str: str) -> StandardDeviceType:
        """
        Parse device string to StandardDeviceType enum.

        Args:
            device_str: Device string

        Returns:
            StandardDeviceType enum value
        """
        normalized = cls.normalize_device_name(device_str)
        return StandardDeviceType(normalized)

    @classmethod
    def to_framework_device(cls, device_str: str, framework_device_enum: Type[Enum]) -> Enum:
        """
        Convert to framework-specific device type.

        Args:
            device_str: Device string (e.g., "nvidia", "cpu")
            framework_device_enum: Framework's device enum (e.g., InfiniLM's DeviceType)

        Returns:
            Framework-specific device enum value

        Examples:
            >>> class SomeDeviceType(Enum):
            ...     DEVICE_TYPE_NVIDIA = 1
            ...     DEVICE_TYPE_CPU = 0
            >>> DeviceHandler.to_framework_device("nvidia", SomeDeviceType)
            <SomeDeviceType.DEVICE_TYPE_NVIDIA: 1>
        """
        normalized = cls.normalize_device_name(device_str)

        # Build attribute name (e.g., "nvidia" -> "DEVICE_TYPE_NVIDIA")
        attr_name = f"DEVICE_TYPE_{normalized.upper()}"

        # Try to get from framework enum
        if hasattr(framework_device_enum, attr_name):
            return getattr(framework_device_enum, attr_name)

        # Try common fallbacks
        fallback_attrs = [
            "DEVICE_TYPE_NVIDIA",
            "DEVICE_TYPE_DEFAULT",
            "DEVICE_TYPE_GPU",
        ]

        for fallback in fallback_attrs:
            if hasattr(framework_device_enum, fallback):
                logger.warning(
                    f"Framework doesn't support {normalized}, using {fallback}"
                )
                return getattr(framework_device_enum, fallback)

        # Last resort: return first enum value
        logger.error(f"Cannot find suitable device type for {normalized}")
        return list(framework_device_enum)[0]

    @classmethod
    def to_uppercase_device(cls, device_str: str) -> str:
        """
        Convert device string to uppercase standard form.

        Useful for adapters that expect uppercase device names.

        Args:
            device_str: Device string

        Returns:
            Uppercase device name (e.g., "NVIDIA", "CPU")

        Examples:
            >>> DeviceHandler.to_uppercase_device("nvidia")
            'NVIDIA'
            >>> DeviceHandler.to_uppercase_device("cuda")
            'NVIDIA'
        """
        return cls.normalize_device_name(device_str).upper()

    @classmethod
    def is_cpu(cls, device_str: str) -> bool:
        """Check if device string represents CPU."""
        return cls.normalize_device_name(device_str) == "cpu"

    @classmethod
    def is_gpu(cls, device_str: str) -> bool:
        """Check if device string represents any GPU."""
        normalized = cls.normalize_device_name(device_str)
        return normalized in ["nvidia", "amd", "intel"]
