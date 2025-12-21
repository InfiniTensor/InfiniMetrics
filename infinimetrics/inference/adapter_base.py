#!/usr/bin/env python3
"""
Inference Adapter Base Class
Defines a unified interface for different frameworks (InfiniLM/vLLM)
"""

import abc
import logging
import random
from typing import List, Tuple, Optional, Dict, Any, Set
from infer_config import InferConfig
from utils.gpu_monitor import create_accelerator_monitor

logger = logging.getLogger(__name__)

class InferAdapter(abc.ABC):
    """Base class for inference adapters"""

    def __init__(self, config: InferConfig):
        self.config = config
        self.model_loaded = False
        self.service_started = False
        self.tokenizer = None

    @abc.abstractmethod
    def load_model(self) -> None:
        """
        Load model
        Subclasses must implement the actual model loading logic
        """
        pass

    @abc.abstractmethod
    def unload_model(self) -> None:
        """
        Unload model
        Subclasses must implement model unloading and resource cleanup
        """
        pass

    @abc.abstractmethod
    def generate(
        self, 
        prompts: List[str], 
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> Tuple[List[str], List[float], List[float]]:
        """
        Generate text
        Returns: (list of generated texts, list of latencies (ms), list of TTFT values (ms))
        """
        pass

    @abc.abstractmethod
    def batch_generate(
        self,
        batch_prompts: List[List[str]],
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> Tuple[List[List[str]], List[List[float]], List[List[float]]]:
        """
        Batch text generation
        Returns: (batch of generated texts, batch of latencies, batch of TTFT values)
        """
        pass

    @abc.abstractmethod
    def calculate_perplexity(self, test_data: List[str]) -> float:
        """
        Calculate perplexity
        """
        pass

    @abc.abstractmethod
    def launch_service(self, port: int = 8000) -> None:
        """
        Launch inference service
        """
        pass

    @abc.abstractmethod
    def stop_service(self) -> None:
        """
        Stop inference service
        """
        pass

    @abc.abstractmethod
    def is_service_ready(self, port: int = 8000) -> bool:
        """
        Check whether the service is ready
        """
        pass

    @abc.abstractmethod
    def get_service_url(self) -> str:
        """
        Get service URL
        """
        pass

    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")
        return len(self.tokenizer)

    def get_special_token_ids(self) -> Set[int]:
        """Get a set of special token IDs that should be excluded"""
        if self.tokenizer is None:
            return set()

        special_ids = set()

        # Retrieve tokenizer's special token map
        special_tokens_map = getattr(self.tokenizer, 'special_tokens_map', {})

        # Collect IDs of all special tokens
        for key, token in special_tokens_map.items():
            if isinstance(token, int):
                special_ids.add(token)
            elif isinstance(token, str):
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id is not None:
                    special_ids.add(token_id)

        # Add commonly used special tokens
        common_special_tokens = [
            "bos_token", "eos_token", "pad_token", "unk_token",
            "sep_token", "cls_token", "mask_token"
        ]

        for token_name in common_special_tokens:
            token = getattr(self.tokenizer, token_name, None)
            if token is not None:
                if isinstance(token, str):
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    if token_id is not None:
                        special_ids.add(token_id)
                elif hasattr(token, 'content'):
                    # Handle special token objects
                    token_id = self.tokenizer.convert_tokens_to_ids(token.content)
                    if token_id is not None:
                        special_ids.add(token_id)

        logger.debug(f"Found {len(special_ids)} special token IDs: {sorted(list(special_ids))}")
        return special_ids

    def generate_random_tokens(self, num_tokens: int, exclude_special: bool = True) -> List[int]:
        """
        Generate a sequence of random token IDs
        Args:
            num_tokens: Number of tokens to generate
            exclude_special: Whether to exclude special tokens
        Returns:
            List of random token IDs
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")

        vocab_size = self.get_vocab_size()

        if exclude_special:
            special_ids = self.get_special_token_ids()
            # Create valid token range (excluding special tokens)
            all_ids = set(range(vocab_size))
            valid_ids = sorted(list(all_ids - special_ids))

            if not valid_ids:
                logger.warning("No valid tokens after excluding special tokens, using all tokens")
                valid_ids = list(range(vocab_size))
        else:
            valid_ids = list(range(vocab_size))

        # Random sampling
        tokens = random.choices(valid_ids, k=num_tokens)

        logger.debug(f"Generated {num_tokens} random tokens (vocab_size={vocab_size}, "
                     f"exclude_special={exclude_special})")

        return tokens

    def tokens_to_text(self, tokens: List[int]) -> str:
        """Convert token IDs to text (for debugging)"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def generate_random_prompt(self, token_count: int, exclude_special: bool = True) -> str:
        """
        Generate a random prompt text
        Args:
            token_count: Number of tokens in the prompt
            exclude_special: Whether to exclude special tokens
        Returns:
            A random prompt string
        """
        tokens = self.generate_random_tokens(token_count, exclude_special)
        return self.tokens_to_text(tokens)

    def generate_random_prompts(self, num_prompts: int, token_count: int,
                               exclude_special: bool = True) -> List[str]:
        """
        Generate multiple random prompts
        Args:
            num_prompts: Number of prompts to generate
            token_count: Number of tokens in each prompt
            exclude_special: Whether to exclude special tokens
        Returns:
            List of random prompt strings
        """
        prompts = []
        for i in range(num_prompts):
            prompt = self.generate_random_prompt(token_count, exclude_special)
            prompts.append(prompt)

        logger.info(f"Generated {num_prompts} random prompts, {token_count} tokens each")
        return prompts

    def validate_config(self) -> List[str]:
        """
        Validate adapter configuration
        Returns: List of error messages
        """
        errors = []

        # Validate model path
        import os
        if not os.path.exists(self.config.model_path):
            errors.append(f"Model path does not exist: {self.config.model_path}")

        # Validate framework-specific configuration
        errors.extend(self._validate_framework_config())

        return errors

    @abc.abstractmethod
    def _validate_framework_config(self) -> List[str]:
        """
        Validate framework-specific configuration
        Subclasses must implement this
        """
        pass

    def get_peak_memory_usage(self) -> Optional[float]:
        """
        Get peak accelerator memory usage (GB)
        Uses the unified accelerator monitoring system
        """
        try:
            # First try to use PyTorch APIs for accurate peak memory
            import torch
        
            # Check various PyTorch accelerator backends
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                # NVIDIA CUDA
                max_memory = 0
                for i in range(torch.cuda.device_count()):
                    max_memory = max(max_memory, torch.cuda.max_memory_allocated(i))
                if max_memory > 0:
                    return max_memory / (1024 ** 3)  # Convert to GB
        
            elif hasattr(torch, 'mlu') and torch.mlu.is_available():
                # Cambricon MLU
                if hasattr(torch.mlu, 'max_memory_allocated'):
                    max_memory = 0
                    for i in range(torch.mlu.device_count()):
                        max_memory = max(max_memory, torch.mlu.max_memory_allocated(i))
                    if max_memory > 0:
                        return max_memory / (1024 ** 3)
        
            elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                # Intel XPU
                if hasattr(torch.xpu, 'max_memory_allocated'):
                    max_memory = 0
                    for i in range(torch.xpu.device_count()):
                        max_memory = max(max_memory, torch.xpu.max_memory_allocated(i))
                    if max_memory > 0:
                        return max_memory / (1024 ** 3)
        
            # Other accelerators could be added here
            # elif hasattr(torch, 'hip') and torch.hip.is_available():  # AMD ROCm
            #     pass
        
            logger.debug("No PyTorch accelerator API available for peak memory")
        
            # Fallback: Try to import and use the accelerator monitor
            try:            
                # Get accelerator type from config (assuming it's stored somewhere)
                accelerator_type = getattr(self.config.device, 'accelerator', 'nvidia')
                device_ids = getattr(self.config.device, 'device_ids', [0])
            
                # Create monitor and get peak memory
                monitor = create_accelerator_monitor(accelerator_type, device_ids)
                monitor.start_monitoring()
                # Need to actually run some monitoring here
                time.sleep(0.1)  # Brief monitoring
                monitor.stop_monitoring()
            
                peak_gb = monitor.get_peak_memory_gb()
                if peak_gb > 0:
                    return peak_gb
                
            except ImportError:
                logger.debug("Accelerator monitor not available")
            
        except ImportError:
            logger.warning("PyTorch not available, cannot get accelerator memory usage")
        except Exception as e:
            logger.warning(f"Failed to get peak memory usage: {e}")
    
        return None
