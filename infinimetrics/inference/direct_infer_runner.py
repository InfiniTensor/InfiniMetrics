#!/usr/bin/env python3
"""
Direct Inference Runner Implementation
Launch real model for batch inference testing
"""
import random
import string
import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from utils.accelerator_monitor import create_accelerator_monitor

from infer_runner_base import InferRunnerBase, ScalarMetric, TimeseriesMetric
from infer_config import InferConfig, DirectInferArgs
from common.metrics import ScalarMetric

logger = logging.getLogger(__name__)

class DirectInferRunner(InferRunnerBase):
    """Direct Inference Runner"""

    def __init__(self, config: InferConfig, adapter):
        super().__init__(config, adapter)
        self.infer_args: DirectInferArgs = config.infer_args

        # accelerator monitor
        self.accelerator_monitor = None

        logger.info(f"DirectInferRunner created for batch_size={self.infer_args.static_batch_size}")

    def setup(self) -> None:
        """Set up direct inference environment"""
        logger.info("Setting up direct inference environment")

        # Create accelerator monitor
        accelerator_type = self.config.device.accelerator  
        device_ids = self.config.device.device_ids         
        cpu_only = self.config.device.cpu_only            
        
        if self.config.device.cpu_only:
            logger.info("CPU-only mode, accelerator monitoring disabled")
            self.accelerator_monitor = None
        else:
            self.accelerator_monitor = create_accelerator_monitor(
                accelerator_type=self.config.device.accelerator.value, 
                device_ids=device_ids
            )

        # Start accelerator monitoring
        if self.accelerator_monitor:
            self.accelerator_monitor.start_monitoring()
            logger.info(f"accelerator monitoring started for devices: {device_ids}")

        # Validate configuration
        if self.infer_args.static_batch_size <= 0:
            raise ValueError(f"Invalid batch size: {self.infer_args.static_batch_size}")

        # Load model
        logger.info("Loading model via adapter")
        self.adapter.load_model()

    def execute(self) -> None:
        """Execute direct inference test"""
        logger.info("Executing direct inference benchmark")

        # Generate test prompts
        prompts = self._generate_test_prompts()
        total_prompts = len(prompts)

        logger.info(f"Generated {total_prompts} test prompts")
        logger.info(f"Input tokens per prompt: {self.infer_args.prompt_token_num}")
        logger.info(f"Output tokens per prompt: {self.infer_args.output_token_num}")

        # Warmup phase
        if self.config.warmup_iterations > 0:
            logger.info(f"Warmup phase ({self.config.warmup_iterations} iterations)")

            warmup_prompts = prompts[:self.config.warmup_iterations * self.infer_args.static_batch_size]

            # Perform warmup in batches
            for i in range(0, len(warmup_prompts), self.infer_args.static_batch_size):
                batch_end = min(i + self.infer_args.static_batch_size, len(warmup_prompts))
                batch_prompts = warmup_prompts[i:batch_end]

                if not batch_prompts:
                    break

                logger.debug(f"Warmup batch {i//self.infer_args.static_batch_size + 1}")

                try:
                    _, _, _ = self.adapter.generate(
                        batch_prompts,
                        self.infer_args.output_token_num,
                        self.infer_args.temperature,
                        self.infer_args.top_p,
                        self.infer_args.top_k
                    )
                except Exception as e:
                    logger.error(f"Warmup batch failed: {e}")
                    logger.error("Exiting due to warmup failure. This may indicate:")

                    raise RuntimeError(f"Warmup failed: {e}")

        # Actual testing phase
        logger.info(f"Measurement phase ({self.config.measured_iterations} iterations)")

        measurement_start = time.perf_counter()

        for i in range(self.config.measured_iterations):
            # Calculate prompt indices for current batch
            start_idx = (self.config.warmup_iterations + i) * self.infer_args.static_batch_size
            end_idx = start_idx + self.infer_args.static_batch_size

            if start_idx >= total_prompts:
                logger.warning(f"Iteration {i+1}: Not enough prompts, reusing earlier prompts")
                start_idx = i * self.infer_args.static_batch_size % total_prompts
                end_idx = start_idx + self.infer_args.static_batch_size

            batch_prompts = prompts[start_idx:end_idx]

            logger.info(f"Running measurement iteration {i+1}/{self.config.measured_iterations}")

            # Call adapter to generate
            generated_texts, latencies, ttfts = self.adapter.generate(
                batch_prompts,
                self.infer_args.output_token_num,
                self.infer_args.temperature,
                self.infer_args.top_p,
                self.infer_args.top_k
            )

            # Collect data
            for latency in latencies:
                self.result.add_latency(latency)
            for ttft in ttfts:
                self.result.add_ttft(ttft)

            # Calculate throughput (tokens/s)
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                # Calculate total tokens in this batch
                batch_tokens = len(batch_prompts) * self.infer_args.output_token_num
                throughput = (batch_tokens * 1000) / avg_latency if avg_latency > 0 else 0
                self.result.add_throughput(throughput)
                logger.info(f"  Iteration {i+1}: avg_latency={avg_latency:.2f}ms, throughput={throughput:.2f} tokens/s")
            else:
                logger.warning(f"  Iteration {i+1}: No latency data collected")

            # Show progress
            progress = (i + 1) / self.config.measured_iterations * 100
            logger.info(f"Measurement progress: {progress:.1f}%")

        measurement_end = time.perf_counter()
        total_measurement_time = measurement_end - measurement_start
        logger.info(f"Measurement phase completed in {total_measurement_time:.2f}s")

    def collect_metrics(self) -> None:
        """Collect direct inference metrics"""
        logger.info("Collecting direct inference metrics")

        # Get peak memory usage
        peak_memory = 0.0
        if self.accelerator_monitor:
            try:
                self.accelerator_monitor.stop_monitoring()
                peak_memory = self.accelerator_monitor.get_peak_memory_gb()
                logger.info(f"Peak accelerator memory usage: {peak_memory:.6f} GB")
            except Exception as e:
                logger.warning(f"Failed to get peak memory from accelerator monitor: {e}")

        # Save to result
        self.result.peak_memory_usage = peak_memory

        # Calculate total tokens
        measured_batches = self.config.measured_iterations * self.infer_args.static_batch_size
        self.result.total_tokens = measured_batches * self.infer_args.output_token_num

        logger.info(f"Total tokens generated (measured only): {self.result.total_tokens}")
        logger.info(f"Warmup tokens (excluded): {self.config.warmup_iterations * self.infer_args.static_batch_size * self.infer_args.output_token_num}")

        # Calculate throughput statistics
        if self.result.throughput_data:
            avg_throughput = sum(self.result.throughput_data) / len(self.result.throughput_data)
            logger.info(f"Average throughput: {avg_throughput:.2f} tokens/s")

        # Extract the public methods
        self._add_perplexity_metric()
        self._add_accuracy_metric()

        # Calculate statistics
        stats = self.calculate_statistics()

        if 'avg_latency' in stats:
            logger.info(f"Average latency: {stats['avg_latency']:.2f} ms")

        if 'avg_ttft' in stats:
            logger.info(f"Average TTFT: {stats['avg_ttft']:.2f} ms")

    def _add_perplexity_metric(self):
        """Add The xx perplexity indicator"""
        perplexity = 0.0
        has_actual_value = False
    
        if self.config.test_dataset:
            try:
                perplexity = self._calculate_perplexity()
                has_actual_value = True
                logger.info(f"Perplexity calculated: {perplexity:.4f}")
            except Exception as e:
                logger.warning(f"Failed to calculate perplexity: {e}")
        
        self.result.add_metric(ScalarMetric(
            name="infer.ppl",
            value=perplexity,
            unit=None if has_actual_value else "placeholder"
        ))
        
    def _add_accuracy_metric(self):
        """Add The accuracy indicator """
        self.result.add_metric(ScalarMetric(
            name="infer.accuracy",
            value=0.0,  # placeholder
            unit="placeholder"
        ))

    def _calculate_perplexity(self) -> float:
        """Calculate perplexity"""
        if not hasattr(self.adapter, 'calculate_perplexity'):
            logger.warning("Adapter does not support perplexity calculation")
            return 0.0

        # Load test data
        test_data = self._load_test_data()
        if not test_data:
            logger.warning("No test data available for perplexity calculation")
            return 0.0

        try:
            return self.adapter.calculate_perplexity(test_data)
        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}")
            return 0.0

    def _load_test_data(self) -> List[str]:
        """Load test data"""
        test_file = self.config.test_dataset
        if not test_file:
            logger.warning("No test dataset specified")
            return []

        # Ensure Path is imported when using it
        test_path = Path(test_file)
        if not test_path.exists():
            logger.warning(f"Test dataset file not found: {test_file}")
            return []

        try:
            with open(test_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract text based on data format
            test_texts = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        text = item.get('text') or item.get('content') or item.get('prompt')
                        if text:
                            test_texts.append(str(text))
                    elif isinstance(item, str):
                        test_texts.append(item)
            elif isinstance(data, dict):
                # Could be multiple keys
                for key, value in data.items():
                    if isinstance(value, str):
                        test_texts.append(value)
                    elif isinstance(value, list):
                        test_texts.extend([str(v) for v in value if isinstance(v, str)])

            logger.info(f"Loaded {len(test_texts)} test samples from {test_file}")
            return test_texts[:100]  # Limit quantity to avoid excessive computation time

        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            return []

    def _generate_test_prompts(self) -> List[str]:
        """Generate test prompts"""
    
        total_prompts_needed = (self.config.warmup_iterations + self.config.measured_iterations) * self.infer_args.static_batch_size
    
        # Attempt to get prompt configuration from configuration file
        prompt_config = getattr(self.config, 'prompt_config', None)
    
        if prompt_config:
            # Special configuration, using PpromptGenerator
            try:
                from utils.prompt_generator import PromptGenerator
            
                # creat prompt generator
                generator = PromptGenerator(
                    method=prompt_config.get("method", "template"),
                    template_name=prompt_config.get("template_name", "ai_qa"),
                    topic_name=prompt_config.get("topic_name", "ai_ml"),
                    prompt_file=prompt_config.get("prompt_file"),
                    fixed_prompt=prompt_config.get("fixed_prompt"),
                    tokenizer=self.adapter.tokenizer if hasattr(self.adapter, 'tokenizer') else None
                )
            
                prompts = generator.generate_prompts(total_prompts_needed, self.infer_args.prompt_token_num)
                logger.info(f"Generated {len(prompts)} prompts using PromptGenerator")
                return prompts
            
            except ImportError as e:
                logger.warning(f"PromptGenerator not available: {e}, using fallback")
                return self._generate_fallback_prompts(total_prompts_needed)
    
        # No configuration, using a simple build
        logger.info("No prompt config provided, using default prompt generation")
        return self._generate_simple_prompts(total_prompts_needed)

    def _generate_fallback_prompts(self, total_prompts_needed: int) -> List[str]:
        """D used when XX is not available"""
        prompts = []
        topics = [
            "artificial intelligence and its applications in healthcare",
            "machine learning algorithms and their use cases", 
            "deep learning and neural networks",
            "natural language processing techniques",
            "computer vision and image recognition",
        ]
    
        for i in range(total_prompts_needed):
            topic = topics[i % len(topics)]
            base_prompt = f"Please provide a detailed explanation about {topic}. "
        
            # Adjust the length
            repeat_count = max(1, self.infer_args.prompt_token_num // len(base_prompt))
            prompt = base_prompt * repeat_count
            prompt = prompt[:self.infer_args.prompt_token_num]
        
            # Add a unique identifier
            random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
            prompt += f" [Request {i+1}:{random_suffix}]"
        
            prompts.append(prompt)
    
        logger.info(f"Generated {len(prompts)} fallback prompts")
        return prompts

    def _generate_simple_prompts(self, total_prompts_needed: int) -> List[str]:
        """Simple generation method"""
        prompts = []
        base_template = "Please provide a detailed explanation about {topic}. "
    
        topics = [
            "artificial intelligence and its applications in healthcare",
            "machine learning algorithms and their use cases", 
            "deep learning and neural networks",
            "natural language processing techniques",
            "computer vision and image recognition",
            "reinforcement learning and autonomous systems",
            "quantum computing and its potential impact",
            "blockchain technology and decentralized applications",
            "Internet of Things and smart devices",
            "cloud computing and distributed systems"
        ]
    
        for i in range(total_prompts_needed):
            topic = topics[i % len(topics)]
            base_prompt = base_template.format(topic=topic)
            repeat_count = max(1, self.infer_args.prompt_token_num // len(base_prompt))
        
            prompt = base_prompt * repeat_count
            prompt = prompt[:self.infer_args.prompt_token_num]
                   
            random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
            prompt += f" [Request {i+1}:{random_suffix}]"
            prompts.append(prompt)
    
        logger.info(f"Generated {len(prompts)} simple prompts")
        return prompts
        
