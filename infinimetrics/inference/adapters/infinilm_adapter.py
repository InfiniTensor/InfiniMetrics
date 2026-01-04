#!/usr/bin/env python3
"""
InfiniLM Adapter - Refactored and Simplified

This is a MUCH simpler version using common utilities.
Original: 497 lines
Refactored: ~180 lines
Reduction: ~65%
"""

import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional

from adapter_base import InferAdapter
from infer_config import InferConfig, DirectInferArgs
from common.constants import DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K
from common.device_utils import DeviceHandler
from common.validation_utils import ValidationMixin
from common.timing_utils import Timer
from common.prompt_utils import PromptGenerator
from common.inference_executor import SimpleInferenceExecutor

logger = logging.getLogger(__name__)

# Try to import InfiniLM modules
try:
    from infinilm.jiuge import JiugeForCauslLM
    from infinilm.libinfinicore_infer import DeviceType as InfiniDeviceType
    from infinilm.infer_task import InferTask, KVCache
    INFINILM_AVAILABLE = True
    DeviceType = InfiniDeviceType
except ImportError:
    INFINILM_AVAILABLE = False
    JiugeForCauslLM = None
    DeviceType = None
    InferTask = None
    KVCache = None


class InfiniLMAdapter(InferAdapter, ValidationMixin):
    """Simplified InfiniLM adapter using common utilities."""

    def __init__(self, config: InferConfig):
        super().__init__(config)

        self.model_instance = None
        self.executor = None  # Will be created on load

        logger.info(f"InfiniLMAdapter created for model: {config.model}")

    def load_model(self) -> None:
        """Load InfiniLM model."""
        if not INFINILM_AVAILABLE:
            raise ImportError("InfiniLM modules not available")

        logger.info(f"Loading model from: {self.config.model_path}")

        device_type = self._get_device_type()
        tp_size = self.config.infer_args.parallel.tp

        self.model_instance = JiugeForCauslLM(
            self.config.model_path,
            device_type,
            tp_size,
            max_tokens=self.config.infer_args.max_seq_len
        )

        self.tokenizer = self.model_instance.tokenizer

        # Create inference executor
        eos_tokens = [self.tokenizer.eos_token_id] if hasattr(self.tokenizer, 'eos_token_id') else []
        self.executor = SimpleInferenceExecutor(
            self.model_instance,
            self.tokenizer,
            eos_tokens
        )

        self.model_loaded = True
        logger.info("Model loaded successfully")

    def unload_model(self) -> None:
        """Unload model."""
        if self.model_instance:
            try:
                if hasattr(self.model_instance, 'destroy_model_instance'):
                    self.model_instance.destroy_model_instance()
            except Exception as e:
                logger.warning(f"Error unloading: {e}")

        self.model_instance = None
        self.executor = None
        self.model_loaded = False
        self.tokenizer = None

    def generate(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K
    ) -> Tuple[List[str], List[float], List[float]]:
        """Generate text using simplified executor."""
        if not self.model_loaded or not self.executor:
            raise RuntimeError("Model not loaded")

        logger.info(f"Generating for {len(prompts)} prompts, max_tokens={max_tokens}")

        # Create inference tasks
        tasks = self._create_tasks(prompts, temperature, top_p, top_k)

        # Run inference using executor
        texts, latencies, ttfts = self.executor.run_batch_inference(
            tasks,
            max_tokens,
            batch_infer_fn=self.model_instance.batch_infer_one_round
        )

        logger.info(f"Generated {len(texts)} texts")
        return texts, latencies, ttfts

    def get_peak_memory_usage(self) -> Optional[float]:
        """Get peak memory usage."""
        try:
            import torch
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.synchronize()
                max_memory = max(
                    [torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())],
                    default=0
                )
                return max_memory / (1024 ** 3)
        except Exception as e:
            logger.warning(f"Failed to get memory: {e}")
        return None

    def calculate_perplexity(self, test_data: List[str]) -> float:
        """Calculate perplexity."""
        if not self.model_loaded or not self.model_instance:
            raise RuntimeError("Model not loaded")

        if not hasattr(self.model_instance, 'perplexity'):
            logger.warning("Model doesn't support perplexity")
            return 0.0

        try:
            test_sequences = [
                self.tokenizer.encode(text)[:self.config.infer_args.max_seq_len]
                for text in test_data
            ]
            return self.model_instance.perplexity(test_sequences, batch_size=4)
        except Exception as e:
            logger.error(f"Perplexity failed: {e}")
            return 0.0

    def _get_device_type(self):
        """Get device type using common DeviceHandler."""
        if DeviceType is None:
            return None
        accelerator = self.config.device.accelerator.value
        return DeviceHandler.to_framework_device(accelerator, DeviceType)

    def _create_tasks(self, prompts: List[str], temperature: float, top_p: float, top_k: int):
        """Create inference tasks."""
        tasks = []

        for i, prompt in enumerate(prompts):
            tokens = self.tokenizer.encode(prompt)

            task = InferTask(
                id=i,
                tokens=tokens,
                max_tokens=self.config.infer_args.max_seq_len,
                temperature=temperature,
                topk=top_k,
                topp=top_p,
                end_tokens=[]
            )

            # Bind KV cache
            kv_cache = KVCache(self.model_instance)
            task.bind_kvcache(kv_cache)

            tasks.append((task, kv_cache))

        logger.info(f"Created {len(tasks)} tasks")
        return tasks

    def _validate_framework_config(self) -> List[str]:
        """Validate using ValidationMixin."""
        errors = []

        errors.extend(self.validate_dependencies_available(
            INFINILM_AVAILABLE, "InfiniLM modules"
        ))

        errors.extend(self.validate_file_exists(
            self.config.model_path,
            f"Model directory does not exist: {self.config.model_path}"
        ))

        errors.extend(self.validate_positive_number(
            self.config.infer_args.parallel.tp, "TP size"
        ))

        return errors

    # ========================================================================
    # Simplified Prompt Generation (using PromptGenerator)
    # ========================================================================

    def _generate_test_prompts(self) -> List[str]:
        """Generate test prompts using common PromptGenerator."""
        total_needed = (
            (self.config.warmup_iterations + self.config.measured_iterations)
            * self.config.infer_args.static_batch_size
        )

        # Check if using custom prompt config
        if hasattr(self.config, 'prompt_config') and self.config.prompt_config:
            return self._generate_custom_prompts(total_needed)

        # Use simple default generation
        generator = PromptGenerator()
        return generator.generate_prompts(
            total_needed,
            self.config.infer_args.prompt_token_num
        )

    def _generate_custom_prompts(self, count: int) -> List[str]:
        """Generate prompts from custom config."""
        prompt_config = self.config.prompt_config

        try:
            from utils.prompt_generator import PromptGenerator as CustomGenerator

            generator = CustomGenerator(
                method=prompt_config.get("method", "template"),
                template_name=prompt_config.get("template_name", "ai_qa"),
                topic_name=prompt_config.get("topic_name", "ai_ml"),
                prompt_file=prompt_config.get("prompt_file"),
                fixed_prompt=prompt_config.get("fixed_prompt"),
                tokenizer=self.tokenizer if hasattr(self, 'tokenizer') else None
            )

            return generator.generate_prompts(count, self.config.infer_args.prompt_token_num)

        except Exception as e:
            logger.warning(f"Custom generator failed: {e}, using default")
            generator = PromptGenerator()
            return generator.generate_prompts(count, self.config.infer_args.prompt_token_num)
