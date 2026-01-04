#!/usr/bin/env python3
"""
Unified Prompt Generator

Provides consistent prompt generation capabilities across all adapters.
Eliminates duplicate prompt generation logic.
"""

import random
import string
import logging
from typing import List, Optional, Callable

logger = logging.getLogger(__name__)


class PromptGenerator:
    """
    Unified prompt generator for inference testing.

    Provides multiple generation strategies and eliminates
    code duplication across adapters.
    """

    # Default topics for prompt generation
    DEFAULT_TOPICS = [
        "artificial intelligence and its applications in healthcare",
        "machine learning algorithms and their use cases",
        "deep learning and neural networks",
        "natural language processing techniques",
        "computer vision and image recognition",
        "reinforcement learning and autonomous systems",
        "quantum computing and its potential impact",
        "blockchain technology and decentralized applications",
        "Internet of Things and smart devices",
        "cloud computing and distributed systems",
        "edge computing and fog computing",
        "data science and big data analytics"
    ]

    def __init__(
        self,
        topics: Optional[List[str]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize prompt generator.

        Args:
            topics: List of topics (uses DEFAULT_TOPICS if None)
            seed: Random seed for reproducibility
        """
        self.topics = topics or self.DEFAULT_TOPICS
        if seed is not None:
            random.seed(seed)

    def generate_prompts(
        self,
        count: int,
        target_token_count: int,
        template: str = "Please provide a detailed explanation about {topic}. "
    ) -> List[str]:
        """
        Generate prompts with specified token count target.

        Args:
            count: Number of prompts to generate
            target_token_count: Target number of tokens per prompt
            template: Prompt template with {topic} placeholder

        Returns:
            List of generated prompts

        Examples:
            >>> gen = PromptGenerator()
            >>> prompts = gen.generate_prompts(10, target_token_count=100)
        """
        prompts = []

        for i in range(count):
            topic = self.topics[i % len(self.topics)]

            # Build base prompt
            base_prompt = template.format(topic=topic)

            # Adjust to target length
            prompt = self._adjust_length(base_prompt, target_token_count)

            # Add unique identifier
            prompt += self._generate_identifier(i + 1)

            prompts.append(prompt)

        logger.info(f"Generated {len(prompts)} prompts (~{target_token_count} tokens each)")
        return prompts

    def generate_simple_prompts(
        self,
        count: int,
        target_token_count: int
    ) -> List[str]:
        """
        Generate simple prompts with basic template.

        Simpler version for backward compatibility.
        """
        return self.generate_prompts(count, target_token_count)

    def _adjust_length(self, base_prompt: str, target_length: int) -> str:
        """
        Adjust prompt length to match target token count.

        Args:
            base_prompt: Base prompt string
            target_length: Target token count

        Returns:
            Adjusted prompt string
        """
        if len(base_prompt) >= target_length:
            return base_prompt[:target_length]

        # Calculate repeat count needed
        repeat_count = max(1, target_length // len(base_prompt))

        # Repeat and truncate
        prompt = base_prompt * repeat_count
        prompt = prompt[:target_length]

        return prompt

    def _generate_identifier(self, index: int) -> str:
        """Generate unique identifier for prompt."""
        random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        return f" [Request {index}:{random_suffix}]"

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer,
        target_token_count: int,
        count: int = 10
    ) -> 'PromptGenerator':
        """
        Create prompt generator that uses tokenizer for accurate sizing.

        Args:
            tokenizer: Tokenizer with encode() method
            target_token_count: Target token count
            count: Number of prompts

        Returns:
            Configured PromptGenerator
        """
        def token_aware_template(topic: str) -> str:
            # Generate a longer template that will be sized properly
            return (
                f"Please provide a comprehensive and detailed explanation about {topic}. "
                f"Include relevant examples, technical details, practical applications, "
                f"potential benefits and challenges, future prospects, and key considerations."
            )

        generator = cls()
        generator._adjust_length = lambda base, target: base  # Will be sized by tokenizer

        # Custom generate method that uses tokenizer
        original_generate = generator.generate_prompts

        def token_aware_generate(count, target_token_count, template=None):
            prompts = original_generate(count, target_token_count * 2, template or token_aware_template("{topic}"))

            # Adjust using tokenizer
            adjusted = []
            for prompt in prompts:
                tokens = tokenizer.encode(prompt)
                if len(tokens) > target_token_count:
                    # Truncate
                    adjusted_tokens = tokens[:target_token_count]
                    prompt = tokenizer.decode(adjusted_tokens, skip_special_tokens=True)
                adjusted.append(prompt)

            return adjusted

        generator.generate_prompts = token_aware_generate
        return generator


def generate_test_prompts(
    count: int,
    target_tokens: int,
    use_tokenizer: Optional[any] = None
) -> List[str]:
    """
    Convenience function for quick prompt generation.

    Args:
        count: Number of prompts
        target_tokens: Target token count per prompt
        use_tokenizer: Optional tokenizer for accurate sizing

    Returns:
        List of generated prompts

    Examples:
        >>> prompts = generate_test_prompts(100, 1024)
        >>> # For tokenizer-aware generation:
        >>> prompts = generate_test_prompts(100, 1024, use_tokenizer=my_tokenizer)
    """
    if use_tokenizer:
        generator = PromptGenerator.from_tokenizer(use_tokenizer, target_tokens, count)
    else:
        generator = PromptGenerator()

    return generator.generate_prompts(count, target_tokens)
