#!/usr/bin/env python3
"""
Unified Inference Executor

Handles the complex logic of batch inference, token generation,
and performance tracking. This can be used by both InfiniLM and
other model adapters.
"""

import logging
from typing import List, Tuple, Optional, Callable, Any
from abc import ABC, abstractmethod
from common.timing_utils import Timer

logger = logging.getLogger(__name__)


class InferenceExecutor(ABC):
    """
    Abstract base for inference execution strategies.

    Encapsulates the complex logic of running batch inference with
    token-by-token generation, EOS handling, and performance tracking.
    """

    def __init__(self, model_instance, tokenizer, eos_token_ids: Optional[list] = None):
        """
        Initialize executor.

        Args:
            model_instance: The model to use for inference
            tokenizer: Tokenizer for encoding/decoding
            eos_token_ids: List of end-of-sequence token IDs
        """
        self.model = model_instance
        self.tokenizer = tokenizer
        self.eos_token_ids = eos_token_ids or []

    @abstractmethod
    def run_batch_inference(self, tasks: List, max_tokens: int) -> Tuple[List[str], List[float], List[float]]:
        """
        Run batch inference - implemented by subclasses.

        Returns:
            (generated_texts, latencies, ttfts)
        """
        pass


class SimpleInferenceExecutor(InferenceExecutor):
    """
    Simplified inference executor for models with batch inference support.

    Handles the complex token-by-token generation loop with proper
    EOS detection and performance tracking.
    """

    def run_batch_inference(
        self,
        tasks: List,
        max_tokens: int,
        batch_infer_fn: Callable
    ) -> Tuple[List[str], List[float], List[float]]:
        """
        Execute batch inference with automatic token generation.

        Args:
            tasks: List of inference tasks
            max_tokens: Maximum tokens to generate
            batch_infer_fn: Function that runs single-round batch inference

        Returns:
            (generated_texts, latencies, ttfts)
        """
        if not tasks:
            return [], [], []

        all_tokens = [[] for _ in range(len(tasks))]
        ttfts = [0.0] * len(tasks)

        # First round inference (Time To First Token)
        with Timer("ttft", auto_logger=False) as ttft_timer:
            outputs = batch_infer_fn(tasks)

        self._process_first_output(outputs, all_tokens, ttfts, ttft_timer.elapsed_ms)

        # Continue generating tokens
        generated_count = 1
        with Timer("generation", auto_logger=False) as gen_timer:

            while generated_count < max_tokens:
                active_tasks, active_indices = self._get_active_tasks(
                    tasks, all_tokens, max_tokens
                )

                if not active_tasks:
                    break

                # Run inference on active tasks
                active_outputs = batch_infer_fn(active_tasks)

                # Process outputs
                self._process_active_outputs(
                    active_outputs, active_indices, all_tokens
                )

                generated_count += 1

                if generated_count % 10 == 0:
                    logger.debug(f"Generated {generated_count}/{max_tokens} tokens")

        total_latency = gen_timer.elapsed_ms

        # Decode all tokens to text
        generated_texts = [
            self.tokenizer.decode(tokens, skip_special_tokens=True) if tokens else ""
            for tokens in all_tokens
        ]

        latencies = [total_latency] * len(tasks)

        logger.info(f"Completed inference for {len(tasks)} tasks")
        return generated_texts, latencies, ttfts

    def _process_first_output(self, outputs, all_tokens, ttfts, ttft_ms: float):
        """Process output from first inference round."""
        for i, output in enumerate(outputs):
            if output:
                token = output[0] if isinstance(output, list) else output
                all_tokens[i].append(token)
                ttfts[i] = ttft_ms
            else:
                all_tokens[i].append(0)
                ttfts[i] = 0.0

    def _get_active_tasks(self, tasks: List, all_tokens: List[List[int]], max_tokens: int):
        """Get tasks that should continue generating."""
        active_tasks = []
        active_indices = []

        for i, (task, tokens) in enumerate(zip(tasks, all_tokens)):
            if not tokens:
                continue

            last_token = tokens[-1]

            # Check if not EOS and not at max length
            if not self._is_eos(last_token) and len(tokens) < max_tokens:
                task.next(last_token)
                active_tasks.append(task)
                active_indices.append(i)

        return active_tasks, active_indices

    def _process_active_outputs(self, outputs, active_indices, all_tokens):
        """Process outputs from active task inference."""
        for idx, task_idx in enumerate(active_indices):
            if idx < len(outputs) and outputs[idx]:
                token = outputs[idx][0] if isinstance(outputs[idx], list) else outputs[idx]
                all_tokens[task_idx].append(token)

    def _is_eos(self, token: int) -> bool:
        """Check if token is an end-of-sequence token."""
        if not self.eos_token_ids:
            return False

        if isinstance(self.eos_token_ids, list):
            return token in self.eos_token_ids
        else:
            return token == self.eos_token_ids
