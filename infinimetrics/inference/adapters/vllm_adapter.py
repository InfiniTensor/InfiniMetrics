#!/usr/bin/env python3
"""
vLLM Adapter Implementation
"""

import logging
import time
import os
import torch
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
    
    def _cleanup_framework_resources(self) -> None:
        """vLLM-specific resource cleanup"""
        try:
            if hasattr(torch.cuda, 'empty_cache') and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared")
        except ImportError:
            pass
    
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

        # Only collect first few texts for debugging        
        debug_texts = []
        latencies_ms = []
        ttfts_ms = []

        # Process each prompt individually to get accurate TTFT
        for i, prompt in enumerate(prompts):

            try:
                #  start time
                start_time = time.perf_counter()
                outputs = self.model.generate([prompt], sampling_params)
                output = outputs[0]
                
                # latency
                latency_ms = (time.perf_counter() - start_time) * 1000
                latencies_ms.append(latency_ms)

                # TTFT from vLLM metrics
                if hasattr(output, "metrics") and hasattr(
                    output.metrics, "first_token_time"
                ):
                    ttft_ms = output.metrics.first_token_time * 1000
                else:
                    # fallback (very rare)
                    ttft_ms = latency_ms * 0.1

                ttfts_ms.append(ttft_ms)

                if i < 3:  # store a few for debugging
                    if hasattr(output, "outputs") and output.outputs:
                        debug_texts.append(output.outputs[0].text)
                    elif hasattr(output, "text"):
                        debug_texts.append(output.text)
                    else:
                        debug_texts.append(str(output))                    
                
            except Exception as e:
                logger.error(f"vLLM generation failed for sample {i}: {e}")
                latencies_ms.append(0.0)
                ttfts_ms.append(0.0)
        
        if latencies_ms:
            avg_latency = sum(latencies_ms) / len(latencies_ms)
            logger.info(f"vLLM generation complete — avg latency {avg_latency:.1f} ms")

        return debug_texts, latencies_ms, ttfts_ms
    
    def calculate_perplexity(self, test_data: List[str]) -> float:
        """Use transformers to compute perplexity (shared with InfiniLM idea)"""

        if not test_data:
            return 0.0

        logger.info(f"Calculating perplexity for {len(test_data)} samples")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )

            total_loss = 0.0
            total_tokens = 0

            for text in test_data[:5]:  # small sample
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model(**inputs, labels=inputs["input_ids"])
                tokens = len(inputs["input_ids"][0])
                total_loss += out.loss.item() * tokens
                total_tokens += tokens

            if total_tokens > 0:
                avg_loss = total_loss / total_tokens
                return float(torch.exp(torch.tensor(avg_loss)))

        except Exception as e:
            logger.warning(f"Perplexity calculation failed: {e}")

        return super().calculate_perplexity(test_data)
    
    def _validate_framework_config(self) -> List[str]:
        """Validate vLLM-specific configuration"""
        errors = []
        
        # Check model path
        if not Path(self.config.model_path).exists():
            errors.append(f"Model path does not exist: {self.config.model_path}")
        
        # Simplified validation logic
        errors.extend(self._validate_parallel_config())
        errors.extend(self._validate_sampling_params())
        
        return errors
    
    def _validate_parallel_config(self) -> List[str]:
        """Validate parallel configuration"""
        errors = []
        
        if self.config.infer_args.parallel.tp <= 0:
            errors.append("Tensor parallel size must be > 0")
        if self.config.infer_args.max_seq_len <= 0:
            errors.append("max_seq_len must be > 0")
        return errors
    
    def _validate_sampling_params(self) -> List[str]:
        errors = []
        if self.config.infer_args.temperature < 0:
            errors.append("Temperature must be >= 0")
        if not (0 <= self.config.infer_args.top_p <= 1):
            errors.append("top_p must be in [0,1]")
        if self.config.infer_args.top_k < -1:
            errors.append("top_k must be >= -1")
        return errors
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")
        return len(self.tokenizer)
