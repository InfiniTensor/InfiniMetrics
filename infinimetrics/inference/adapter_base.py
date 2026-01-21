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
from utils.accelerator_monitor import create_accelerator_monitor

logger = logging.getLogger(__name__)

class InferAdapter(abc.ABC):
    """Base class for inference adapters"""

    def __init__(self, config: InferConfig):
        self.config = config
        self.model_loaded = False
        self.tokenizer = None
        self.model = None

    @abc.abstractmethod
    def load_model(self) -> None:
        """
        Load model
        """
        pass

    def unload_model(self) -> None:
        """Unload model"""
        if self.model_loaded:
            try:
                # Framework-specific cleanup logic is implemented in subclasses
                self._cleanup_framework_resources()
            except Exception as e:
                logger.warning(f"Error during framework-specific cleanup: {e}")
            
            self.model = None
            self.model_loaded = False
            self.tokenizer = None
            logger.info("Model unloaded")
    
    @abc.abstractmethod
    def _cleanup_framework_resources(self) -> None:
        """Framework-specific resource cleanup"""
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
        """
        pass
    
    def calculate_perplexity(self, test_data: List[str]) -> float:
        """Calculate perplexity"""
        if not test_data:
            logger.warning("No test data provided for perplexity calculation")
            return 0.0
        
        logger.info(f"Calculating perplexity for {len(test_data)} samples")
        
        # The default implementation, which returns a placeholder value
        logger.warning("Perplexity calculation not implemented, returning placeholder value")
        return 0.0
    
    @abc.abstractmethod
    def _validate_framework_config(self) -> List[str]:
        """Validate framework-specific configuration"""
        pass

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
        """
        all_results = []
        for prompts in batch_prompts:
            texts, latencies, ttfts = self.generate(
                prompts, max_tokens, temperature, top_p, top_k
            )
            all_results.append((texts, latencies, ttfts))
        
        # Results of reorganization
        all_texts, all_latencies, all_ttfts = [], [], []
        for texts, latencies, ttfts in all_results:
            all_texts.append(texts)
            all_latencies.append(latencies)
            all_ttfts.append(ttfts)
        
        return all_texts, all_latencies, all_ttfts

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

        # Generic special token fetching logic
        token_attrs = ['bos_token', 'eos_token', 'pad_token', 'unk_token']
        
        for attr in token_attrs:
            token = getattr(self.tokenizer, attr, None)
            if token and isinstance(token, str):
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id is not None:
                    special_ids.add(token_id)
        
        logger.debug(f"Found {len(special_ids)} special token IDs")
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
        """Validate adapter configuration"""
        errors = []
        import os
        if not os.path.exists(self.config.model_path):
            errors.append(f"Model path does not exist: {self.config.model_path}")
        errors.extend(self._validate_framework_config())
        return errors

    def get_peak_memory_usage(self) -> Optional[float]:
        """Get peak accelerator memory usage (GB)"""
        try:
            import torch
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                max_memory = 0
                for i in range(torch.cuda.device_count()):
                    max_memory = max(max_memory, torch.cuda.max_memory_allocated(i))
                return max_memory / (1024 ** 3)
        except ImportError:
            logger.warning("PyTorch not available, cannot get GPU memory usage")
        return None
        
