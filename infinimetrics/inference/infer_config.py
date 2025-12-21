#!/usr/bin/env python3
"""
Inference Configuration Manager
Parses config.json, identifies direct/service mode, identifies infinilm/vllm framework
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import random
import string

from common.testcase_utils import (
    parse_testcase, 
    generate_run_id,
    validate_testcase_format
)
from common.constants import (
    ProcessorType,
    AcceleratorType,
    DEFAULT_WARMUP_ITERATIONS,
    DEFAULT_MEASURED_ITERATIONS,
    DEFAULT_TIMEOUT_MS,
    DEFAULT_OUTPUT_DIR
)

logger = logging.getLogger(__name__)

class InferMode(Enum):
    """Inference mode enumeration"""
    DIRECT = "direct"
    SERVICE = "service"

class FrameworkType(Enum):
    """Framework type enumeration"""
    INFINILM = "infinilm"
    VLLM = "vllm"
#inference default
DEFAULT_TIMEOUT_MS_SERVICE = 30000
DEFAULT_MAX_SEQ_LEN = 4096
DEFAULT_PROMPT_TOKEN_NUM = 1024
DEFAULT_OUTPUT_TOKEN_NUM = 128
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50
DEFAULT_CONCURRENCY = 32
DEFAULT_STATIC_BATCH_SIZE = 1
DEFAULT_STREAM = True

@dataclass
class ParallelConfig:
    """Parallel configuration"""
    dp: int = 1
    tp: int = 1
    pp: int = 1
    sp: int = 1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParallelConfig':
        """Create parallel configuration from dictionary"""
        if not data:
            return cls()
        return cls(
            dp=data.get("dp", 1),
            tp=data.get("tp", 1),
            pp=data.get("pp", 1),
            sp=data.get("sp", 1)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "dp": self.dp,
            "tp": self.tp,
            "pp": self.pp,
            "sp": self.sp
        }

@dataclass
class DeviceConfig:
    """Device configuration"""
    processor_type: ProcessorType = ProcessorType.ACCELERATOR
    accelerator: Optional[AcceleratorType] = AcceleratorType.NVIDIA
    device_ids: List[int] = None
    cpu_only: bool = False

    def __post_init__(self):
        if self.device_ids is None:
            self.device_ids = [0]
        
        if self.cpu_only:
            self.processor_type = ProcessorType.CPU
            self.accelerator = None
        
        if self.processor_type == ProcessorType.ACCELERATOR and self.accelerator is None:
            raise ValueError("accelerator must be specified when processor_type is ACCELERATOR")
        
        if self.processor_type == ProcessorType.CPU and self.accelerator is not None:
            self.accelerator = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeviceConfig':
        """Create device configuration from dictionary"""
        if not data:
            return cls()
        
        cpu_only = data.get("cpu_only", False)
        
        if cpu_only:
            # CPU mode
            return cls(
                processor_type=ProcessorType.CPU,
                accelerator=None,
                device_ids=data.get("device_ids", [0]),
                cpu_only=True
            )
        else:
            # accelerator mode
            accelerator_str = data.get("accelerator")
            if not accelerator_str:
                accelerator_str = data.get("gpu_platform", "nvidia")
            
            # Mapping to the correct acceleratorType
            accelerator_map = {
                "nvidia": AcceleratorType.NVIDIA,
                "cambricon": AcceleratorType.CAMBRICON,
                "ascend": AcceleratorType.ASCEND,
                "intel": AcceleratorType.INTEL,
                "xpu": AcceleratorType.INTEL,    
                "amd": AcceleratorType.AMD,
                "rocm": AcceleratorType.AMD,         
            }
            
            accelerator = accelerator_map.get(accelerator_str.lower())
            if accelerator is None:
                logger.warning(f"Unknown accelerator: {accelerator_str}, using NVIDIA as default")
                accelerator = AcceleratorType.NVIDIA
            
            return cls(
                processor_type=ProcessorType.ACCELERATOR,
                accelerator=accelerator,
                device_ids=data.get("device_ids", [0]),
                cpu_only=False
            )

    def to_dict(self) -> Dict[str, Any]:
        """Into a dictionary"""
        result = {
            "processor_type": self.processor_type.value,
            "device_ids": self.device_ids,
            "cpu_only": self.cpu_only
        }
        
        if self.accelerator:
            result["accelerator"] = self.accelerator.value
        
        return result
    
    @property
    def is_cpu(self) -> bool:
        """If it is in CPU mode"""
        return self.processor_type == ProcessorType.CPU
    
    @property
    def accelerator_type(self) -> Optional[AcceleratorType]:
        """Gets the accelerator type"""
        return self.accelerator if self.processor_type == ProcessorType.ACCELERATOR else None

@dataclass
class DirectInferArgs:
    """Direct inference arguments"""
    parallel: ParallelConfig
    static_batch_size: int
    prompt_token_num: int
    output_token_num: int = DEFAULT_OUTPUT_TOKEN_NUM
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DirectInferArgs':
        """Create direct inference arguments from dictionary"""
        return cls(
            parallel=ParallelConfig.from_dict(data.get("parallel", {})),
            static_batch_size=data.get("static_batch_size", DEFAULT_STATIC_BATCH_SIZE),
            prompt_token_num=data.get("prompt_token_num", DEFAULT_PROMPT_TOKEN_NUM),
            output_token_num=data.get("output_token_num", DEFAULT_OUTPUT_TOKEN_NUM),
            max_seq_len=data.get("max_seq_len", DEFAULT_MAX_SEQ_LEN),
            temperature=data.get("temperature", DEFAULT_TEMPERATURE),
            top_p=data.get("top_p", DEFAULT_TOP_P),
            top_k=data.get("top_k", DEFAULT_TOP_K)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "parallel": self.parallel.to_dict(),
            "static_batch_size": self.static_batch_size,
            "prompt_token_num": self.prompt_token_num,
            "output_token_num": self.output_token_num,
            "max_seq_len": self.max_seq_len,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k
        }

@dataclass
class ServiceInferArgs:
    """Service inference arguments"""
    parallel: ParallelConfig
    request_trace: str
    concurrency: int = DEFAULT_CONCURRENCY
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    stream: bool = DEFAULT_STREAM
    timeout_ms: int = DEFAULT_TIMEOUT_MS_SERVICE

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ServiceInferArgs':
        """Create service inference arguments from dictionary"""
        return cls(
            parallel=ParallelConfig.from_dict(data.get("parallel", {})),
            request_trace=data.get("request_trace", ""),
            concurrency=data.get("concurrency", DEFAULT_CONCURRENCY),
            max_seq_len=data.get("max_seq_len", DEFAULT_MAX_SEQ_LEN),
            stream=data.get("stream", DEFAULT_STREAM),
            timeout_ms=data.get("timeout_ms", DEFAULT_TIMEOUT_MS_SERVICE)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "parallel": self.parallel.to_dict(),
            "request_trace": self.request_trace,
            "concurrency": self.concurrency,
            "max_seq_len": self.max_seq_len,
            "stream": self.stream,
            "timeout_ms": self.timeout_ms
        }

@dataclass
class InferConfig:
    """Main inference configuration class"""
    # Basic information
    run_id: str
    testcase: str
    model: str
    model_path: str
    model_config: Optional[str]

    # Dataset
    train_dataset: Optional[str]
    validation_dataset: Optional[str]
    test_dataset: Optional[str]

    # Output
    output_dir: str

    # Execution mode
    mode: InferMode
    framework: FrameworkType

    # Device configuration
    device: DeviceConfig

    # Inference arguments (varies by mode)
    infer_args: Any  # DirectInferArgs or ServiceInferArgs

    # Execution parameters
    timeout_ms: int
    warmup_iterations: int
    measured_iterations: int

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'InferConfig':
        """Create configuration object from dictionary"""
        # 1. Read outer level
        outer_run_id = config_dict.get("run_id")
        outer_testcase = config_dict.get("testcase")
        config_data = config_dict.get("config", {})

        # 2. verify testcase position
        inner_testcase = config_data.get("testcase")
        if outer_testcase:
            testcase = outer_testcase
            if not validate_testcase_format(testcase):
                raise ValueError(f"Invalid testcase format: {testcase}")
            logger.info(f"Using outer testcase: {testcase}")
        elif inner_testcase:
            raise ValueError(
                "testcase must be at the outer level, not inside 'config'. "
                f"Found: '{inner_testcase}' inside 'config'."
            )
        else:
            raise ValueError(
                "testcase is required at the outer level of the config."
            )

        # 3. verify run_id position
        inner_run_id = config_data.get("run_id")
        if inner_run_id:
            raise ValueError(
                "run_id must be at the outer level, not inside 'config'. "
                f"Found: '{inner_run_id}' inside 'config'."
            )

        # 4.  generate run_id
        run_id = generate_run_id(testcase, outer_run_id)

        # 5. Analytical reasoning models and frameworks
        mode_str, framework_str = parse_testcase(testcase)
        
        mode = InferMode.DIRECT if mode_str == "direct" else InferMode.SERVICE
        if framework_str == "infinilm":
            framework = FrameworkType.INFINILM
        elif framework_str == "vllm":
            framework = FrameworkType.VLLM
        else:
            raise ValueError(f"Unsupported inference framework: {framework_str}")

        # 6. Parse model_path
        model_path = config_data.get("model_path")
        model_config = config_data.get("model_config")

        if not model_path and model_config:
            model_path = str(Path(model_config).parent)
            logger.info(f"Inferred model_path from model_config: {model_path}")
        elif not model_path:
            raise ValueError("Either model_path or model_config must be provided in config")

        # 7. Parse inference arguments
        infer_args_dict = config_data.get("infer_args", {})
        if mode == InferMode.DIRECT:
            infer_args = DirectInferArgs.from_dict(infer_args_dict)
        else:
            infer_args = ServiceInferArgs.from_dict(infer_args_dict)

        # 8. Parse device configuration
        device_config = DeviceConfig.from_dict(config_data.get("device", {}))

        return cls(
            run_id=run_id,
            testcase=testcase,
            model=config_data.get("model", "unknown"),
            model_path=model_path,
            model_config=model_config,

            train_dataset=config_data.get("train_dataset"),
            validation_dataset=config_data.get("validation_dataset"),
            test_dataset=config_data.get("test_dataset"),

            output_dir=config_data.get("output_dir", "./test_output"),

            mode=mode,
            framework=framework,
            device=device_config,
            infer_args=infer_args,

            timeout_ms=config_data.get("timeout_ms", 30000)if mode == InferMode.SERVICE else None,
            warmup_iterations=config_data.get("warmup_iterations", 10),
            measured_iterations=config_data.get("measured_iterations", 100)
        )
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON output)"""
        return {
            "run_id": self.run_id,
            "testcase": self.testcase,
            "mode": self.mode.value,
            "framework": self.framework.value,
            "model": self.model,
            "model_path": self.model_path,
            "model_config": self.model_config,
            "device": self.device.to_dict(),
            "infer_args": self.infer_args.to_dict() if hasattr(self.infer_args, 'to_dict') else {},
            "warmup_iterations": self.warmup_iterations,
            "measured_iterations": self.measured_iterations
        }
        if self.mode == InferMode.SERVICE and self.timeout_ms is not None:
            result["timeout_ms"] = self.timeout_ms

        return result

class InferConfigManager:
    """Inference Configuration Manager"""

    @staticmethod
    def load_config(config_file: str) -> InferConfig:
        """Load configuration from config file"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # Support single config or config list
            if isinstance(config_data, dict):
                config_dict = config_data
            elif isinstance(config_data, list) and len(config_data) > 0:
                config_dict = config_data[0]
            else:
                raise ValueError(f"Invalid config format in {config_file}")

            # Create configuration object
            config = InferConfig.from_dict(config_dict)

            # Validate configuration
            errors = InferConfigManager.validate_config(config)
            if errors:
                error_msg = "Configuration validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
                logger.error(error_msg)
                raise ValueError(error_msg)

            return config

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON config file: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load config from {config_file}: {e}")
            raise

    @staticmethod
    def validate_config(config: InferConfig) -> List[str]:
        """Validate configuration effectiveness"""
        errors = []

        # Basic validation
        if not config.run_id:
            errors.append("run_id cannot be empty")

        if not config.testcase:
            errors.append("testcase cannot be empty")

        if not config.model or config.model == "unknown":
            errors.append("model name is required")

        # Model path validation
        model_dir = Path(config.model_path)
        if not model_dir.exists():
            errors.append(f"Model directory does not exist: {model_dir}")
        else:
            # Check config.json
            config_file = model_dir / "config.json"
            if not config_file.exists():
                errors.append(f"config.json not found in model directory: {model_dir}")

        # Mode-specific validation
        if config.mode == InferMode.DIRECT:
            if not isinstance(config.infer_args, DirectInferArgs):
                errors.append("Direct mode requires DirectInferArgs")
            else:
                if config.infer_args.static_batch_size <= 0:
                    errors.append("Batch size must be positive")
                if config.infer_args.prompt_token_num <= 0:
                    errors.append("Prompt token number must be positive")

        elif config.mode == InferMode.SERVICE:
            if not isinstance(config.infer_args, ServiceInferArgs):
                errors.append("Service mode requires ServiceInferArgs")
            else:
                if not config.infer_args.request_trace:
                    errors.append("Request trace is required for service mode")
                if config.infer_args.concurrency <= 0:
                    errors.append("Concurrency must be positive")

                # Verify trace file exists
                trace_path = Path(config.infer_args.request_trace)
                if not trace_path.exists():
                    errors.append(f"Trace file not found: {trace_path}")

        # Output directory validation
        output_dir = Path(config.output_dir)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            test_file = output_dir / ".test_write"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            errors.append(f"Output directory is not writable: {e}")

        return errors

    @staticmethod
    def generate_auto_run_id(testcase: str) -> str:
        """
        Auto-generate run_id (public method)
    
        Args:
            testcase: testcase string
        
        Returns:
            Generated run_id
        """
        from common.testcase_utils import generate_auto_run_id as common_generate_run_id
        return common_generate_run_id(testcase)

    # Original private method calls public method
    @staticmethod
    def _generate_auto_run_id(testcase: str) -> str:
        """Private method, calls public method (maintain backward compatibility)"""
        return InferConfigManager.generate_auto_run_id(testcase)