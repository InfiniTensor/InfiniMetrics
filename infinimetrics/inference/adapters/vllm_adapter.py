#!/usr/bin/env python3
"""
vLLM Adapter Implementation
"""

import logging
import time
import os
from pathlib import Path
from typing import List, Tuple, Optional, Set, Dict, Any

import vllm
from vllm import LLM, SamplingParams

from adapter_base import InferAdapter
from infer_config import InferConfig, DirectInferArgs, ServiceInferArgs
from common.constants import (
    DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K,
    DEFAULT_MAX_SEQ_LEN, DEFAULT_OUTPUT_TOKEN_NUM,
    DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
    DEFAULT_VLLM_DTYPE,
    DEFAULT_VLLM_TRUST_REMOTE_CODE,
    DEFAULT_VLLM_DISABLE_LOG_STATS,
    DEFAULT_VLLM_SEED,
    DEFAULT_VLLM_SWAP_SPACE,
)

logger = logging.getLogger(__name__)


class VLLMAdapter(InferAdapter):
    """vLLM Adapter """
    
    def __init__(self, config: InferConfig):
        super().__init__(config)
        self.model = None
        self.vllm_version = getattr(vllm, '__version__', 'unknown')
        
        # Initialize according to mode
        if config.mode.value == "direct":
            self.infer_args: DirectInferArgs = config.infer_args
        else:
            self.infer_args: ServiceInferArgs = config.infer_args
        
        logger.info(f"vLLM version: {self.vllm_version}")
    
    def _setup_device_visibility(self):
        """Configure device visibility"""
        if self.config.device.cpu_only:
            logger.info("CPU-only mode")
            return
            
        if self.config.device.accelerator.value == "nvidia" and self.config.device.device_ids:
            device_str = ",".join(str(d) for d in self.config.device.device_ids)
            if not os.environ.get("CUDA_VISIBLE_DEVICES"):
                os.environ["CUDA_VISIBLE_DEVICES"] = device_str
                logger.info(f"Set CUDA_VISIBLE_DEVICES={device_str}")
    
    def load_model(self) -> None:
        """Load vLLM model"""
        logger.info(f"Loading vLLM model from {self.config.model_path}")
        
        # Set device visibility
        self._setup_device_visibility()
        
        # Build vLLM arguments
        vllm_kwargs = {
            "model": self.config.model_path,
            "max_model_len": self.config.infer_args.max_seq_len,
            "tensor_parallel_size": self.config.infer_args.parallel.tp,
            "dtype": DEFAULT_VLLM_DTYPE,
            "trust_remote_code": DEFAULT_VLLM_TRUST_REMOTE_CODE,
            "disable_log_stats": DEFAULT_VLLM_DISABLE_LOG_STATS,
            "seed": DEFAULT_VLLM_SEED,
        }
        
        # NVIDIA GPU-specific parameters
        if (not self.config.device.cpu_only and 
            self.config.device.accelerator.value == "nvidia"):
            vllm_kwargs.update({
                "gpu_memory_utilization": DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
                "swap_space": DEFAULT_VLLM_SWAP_SPACE,
            })
        
        # Override values from framework_kwargs
        framework_kwargs = self._parse_framework_kwargs()
        for key in vllm_kwargs:
            if key in framework_kwargs:
                vllm_kwargs[key] = framework_kwargs[key]
        
        # Load model
        self.model = LLM(**vllm_kwargs)
        self.model_loaded = True
        self.tokenizer = self.model.get_tokenizer()
        logger.info(f"vLLM model loaded, vocab size: {len(self.tokenizer)}")
    
    def _parse_framework_kwargs(self) -> Dict[str, Any]:
        """Parse framework-specific parameters"""
        if hasattr(self.config.infer_args, 'framework_kwargs'):
            kwargs = self.config.infer_args.framework_kwargs
            if kwargs and isinstance(kwargs, dict):
                return kwargs
        return {}
    
    def unload_model(self) -> None:
        """Unload model"""
        if self.model:
            self.model = None
            self.model_loaded = False
            self.tokenizer = None
            logger.info("vLLM model unloaded")
    
    def generate(
        self, 
        prompts: List[str], 
        max_tokens: int,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K
    ) -> Tuple[List[str], List[float], List[float]]:
        """Generate text"""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        if not prompts:
            return [], [], []
        
        # Sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
        )
        
        generated_texts = []
        latencies_ms = []
        ttfts_ms = []
        
        # Record start time
        start_time = time.perf_counter()
        
        # Batch generation
        outputs = self.model.generate(prompts, sampling_params)
        
        # Calculate total latency
        total_latency = (time.perf_counter() - start_time) * 1000
        
        # Process outputs
        for i, output in enumerate(outputs):
            # Extract generated text
            if hasattr(output, 'outputs') and output.outputs:
                generated_text = output.outputs[0].text
            elif hasattr(output, 'text'):
                generated_text = output.text
            else:
                generated_text = str(output)
            
            generated_texts.append(generated_text)
            
            # Compute latency
            avg_latency = total_latency / len(outputs) if outputs else 0
            latencies_ms.append(avg_latency)
            
            # Estimate TTFT
            estimated_ttft = avg_latency * 0.15
            ttfts_ms.append(estimated_ttft)
        
        if outputs:
            logger.info(f"Generated {len(generated_texts)} responses, avg latency: {total_latency/len(outputs):.1f}ms")
        
        return generated_texts, latencies_ms, ttfts_ms
    
    def calculate_perplexity(self, test_data: List[str]) -> float:
        """Calculate perplexity (placeholder implementation)"""
        if not test_data:
            return 0.0
        
        logger.info(f"Calculating perplexity for {len(test_data)} samples")
        
        # vLLM does not directly support perplexity calculation
        # Return a placeholder value
        return 0.0
    
    def _validate_framework_config(self) -> List[str]:
        """Validate configuration"""
        errors = []
        
        # Check model path
        if not Path(self.config.model_path).exists():
            errors.append(f"Model path does not exist: {self.config.model_path}")
        
        # Check parallelism parameters
        tp = self.config.infer_args.parallel.tp
        if tp <= 0:
            errors.append(f"Tensor parallel size must be positive, got: {tp}")
        
        # Check sequence length
        max_seq_len = self.config.infer_args.max_seq_len
        if max_seq_len <= 0:
            errors.append(f"Max sequence length must be positive, got: {max_seq_len}")
        
        return errors
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")
        return len(self.tokenizer)
    
    def get_special_token_ids(self) -> Set[int]:
        """Get special token IDs"""
        if self.tokenizer is None:
            return set()
        
        special_ids = set()
        
        # Collect special tokens
        special_tokens = []
        
        # Common special token attributes
        token_attrs = ['bos_token', 'eos_token', 'pad_token', 'unk_token']
        
        for attr in token_attrs:
            token = getattr(self.tokenizer, attr, None)
            if token and isinstance(token, str):
                special_tokens.append(token)
        
        # Convert tokens to IDs
        for token in special_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is not None:
                special_ids.add(token_id)
        
        logger.debug(f"Found {len(special_ids)} special token IDs")
        return special_ids
        
