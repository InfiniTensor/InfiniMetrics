#!/usr/bin/env python3
"""
Common Constants
Shared constants for both inference and training
"""

from enum import Enum

class ProcessorType(str, Enum):
    CPU = "cpu"                    
    ACCELERATOR = "accelerator"

class AcceleratorType(str, Enum):
    """Supported accelerator platforms"""
    NVIDIA = "nvidia"
    AMD = "amd"        # ROCm
    ASCEND = "ascend"  # Huawei NPU
    CAMBRICON = "cambricon"  # Cambricon MLU
    GENERIC = "generic"


# Common execution defaults
DEFAULT_WARMUP_ITERATIONS = 10
DEFAULT_MEASURED_ITERATIONS = 100
DEFAULT_TIMEOUT_MS = 30000

# Common paths
DEFAULT_OUTPUT_DIR = "./test_output"
DEFAULT_LOG_DIR = "./logs"
