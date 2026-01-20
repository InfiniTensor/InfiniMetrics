"""Device management utilities for test adapters."""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class DeviceManager:
    """Utility class for device detection and configuration validation."""

    SUPPORTED_DEVICE_TYPES = ["cpu", "cuda", "gpu", "rocm", "accelerator"]

    @staticmethod
    def get_device_type(config: Dict[str, Any]) -> str:
        """
        Get the device type from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Device type as lowercase string (default: "cuda")
        """
        device = config.get("device", "cuda")
        return str(device).lower()

    @staticmethod
    def validate_device_config(config: Dict[str, Any]) -> List[str]:
        """
        Validate device configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        device = DeviceManager.get_device_type(config)

        # Check if device type is supported
        if device not in DeviceManager.SUPPORTED_DEVICE_TYPES:
            errors.append(
                f"Unsupported device type '{device}'. "
                f"Supported types: {', '.join(DeviceManager.SUPPORTED_DEVICE_TYPES)}"
            )

        # Device-specific validation
        if device == "cuda":
            device_id = config.get("device_id")
            if device_id is not None:
                if not isinstance(device_id, int) or device_id < 0:
                    errors.append(
                        f"Invalid device_id '{device_id}'. Must be a non-negative integer."
                    )

        return errors

    @staticmethod
    def is_cpu_mode(config: Dict[str, Any]) -> bool:
        """
        Check if configuration specifies CPU mode.

        Args:
            config: Configuration dictionary

        Returns:
            True if CPU mode is enabled
        """
        return DeviceManager.get_device_type(config) == "cpu"

    @staticmethod
    def should_skip_build(config: Dict[str, Any]) -> bool:
        """
        Determine if build should be skipped based on config.

        Args:
            config: Configuration dictionary

        Returns:
            True if build should be skipped
        """
        return DeviceManager.is_cpu_mode(config) or config.get("skip_build", False)
