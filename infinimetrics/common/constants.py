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

# Inference-specific defaults
DEFAULT_MAX_SEQ_LEN = 4096
DEFAULT_PROMPT_TOKEN_NUM = 1024
DEFAULT_OUTPUT_TOKEN_NUM = 128
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50
DEFAULT_CONCURRENCY = 32
DEFAULT_STATIC_BATCH_SIZE = 1
DEFAULT_STREAM = True
DEFAULT_TIMEOUT_MS_SERVICE = 30000
