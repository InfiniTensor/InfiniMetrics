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
    AMD = "amd"  # ROCm
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


# vLLM-specific defaults
DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_VLLM_DTYPE = "auto"
DEFAULT_VLLM_TRUST_REMOTE_CODE = True
DEFAULT_VLLM_DISABLE_LOG_STATS = True
DEFAULT_VLLM_SEED = 0
DEFAULT_VLLM_SWAP_SPACE = 4  # GB
DEFAULT_VLLM_MAX_NUM_BATCHED_TOKENS = None
DEFAULT_VLLM_MAX_NUM_SEQS = 256
DEFAULT_VLLM_ENFORCE_EAGER = False

# Data type byte sizes for bandwidth calculation
DTYPE_BYTES_MAP = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
}


# Field name constants organized by context
class InfiniMetricsJson:
    """Top-level JSON field names for InfiniMetrics"""

    CONFIG = "config"
    METRICS = "metrics"
    TESTCASE = "testcase"
    RUN_ID = "run_id"
    TIME = "time"
    RESULT_CODE = "result_code"
    ERROR_MSG = "error_msg"


# ============================================================
# InfiniCore Adapter Constants
# Only for strings used 2+ times
# ============================================================
class InfiniCoreRequest:
    """InfiniCore API request field names"""

    OPERATOR = "operator"
    DEVICE = "device"
    TORCH_OP = "torch_op"
    INFINICORE_OP = "infinicore_op"
    ARGS = "args"
    TESTCASES = "testcases"
    DESCRIPTION = "description"
    INPUTS = "inputs"
    KWARGS = "kwargs"
    RESULT = "result"
    TOLERANCE = "tolerance"


class OperatorConfig:
    """Operator configuration field names"""

    OPERATOR = "operator"
    DEVICE = "device"
    INPUTS = "inputs"
    OUTPUTS = "outputs"
    ATTRIBUTES = "attributes"
    TOLERANCE = "tolerance"
    INFINICORE_OP = "infinicore_op"
    TORCH_OP = "torch_op"


class TensorSpec:
    """Tensor specification field names"""

    NAME = "name"
    SHAPE = "shape"
    DTYPE = "dtype"
    STRIDES = "strides"
    VALUE = "value"
    INPLACE = "inplace"


class InfiniCoreResult:
    """InfiniCore test result field names"""

    TESTCASES = "testcases"
    RESULT = "result"
    STATUS = "status"
    SUCCESS = "success"
    ERROR = "error"
    PERF_MS = "perf_ms"
    METRICS = "metrics"


# Device and performance constants
DEVICE_CPU = "CPU"
DEVICE_NVIDIA = "NVIDIA"
PERF_HOST = "host"
PERF_DEVICE = "device"
PLATFORM_INFINICORE = "infinicore"

# Default values
DEFAULT_TOLERANCE = {"atol": 1e-3, "rtol": 1e-3}
