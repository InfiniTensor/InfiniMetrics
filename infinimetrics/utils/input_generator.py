#!/usr/bin/env python3
"""
Random Input Generator for InfiniMetrics
Generates random tensor data with configurable parameters and saves to files
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RandomInputGenerator:
    """Generate random tensor inputs and save to files (standalone tool)"""

    SUPPORTED_FORMATS = ['.npy', '.pt', '.pth']

    def __init__(self,
                 output_dir: Union[str, Path] = "./generated_data",
                 default_format: str = ".npy",
                 seed: Optional[int] = None):
        """
        Args:
            output_dir: Directory to save generated files
            default_format: Default file format (.npy, .pt, .pth)
            seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.default_format = default_format
        self.seed = seed

    def generate_input_config(self,
                            base_name: str,
                            shape: List[int],
                            dtype: str,
                            distribution: str = "uniform",
                            format: Optional[str] = None,
                            **distribution_kwargs) -> Dict[str, Any]:
        """
        Generate random data and return input config with file path

        Args:
            base_name: Base name for the tensor (e.g., "input_a")
            shape: Tensor shape
            dtype: Data type (e.g., "float16", "float32")
            distribution: Distribution type ("uniform", "normal", "randint")
            format: File format (.npy, .pt, .pth), defaults to self.default_format
            **distribution_kwargs: Distribution-specific parameters
                - For "uniform": low (default=0.0), high (default=1.0), scale, bias
                - For "normal": mean (default=0.0), std (default=1.0)
                - For "randint": low, high

        Returns:
            Dictionary with file_path and metadata
        """
        file_format = format or self.default_format

        if file_format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {file_format}. Supported: {self.SUPPORTED_FORMATS}")

        # Generate data
        data = self._generate_random_data(
            shape=shape,
            dtype=dtype,
            distribution=distribution,
            **distribution_kwargs
        )

        # Generate unique filename with timestamp
        file_path = self._generate_unique_path(
            base_name=base_name,
            shape=shape,
            dtype=dtype,
            file_format=file_format
        )

        # Save to file
        self._save_data(data, file_path, file_format)

        # Return config for JSON
        return {
            "name": base_name,
            "file_path": str(file_path),
            "dtype": dtype,
            "shape": shape,  # Include shape for validation
            "_metadata": {
                "distribution": distribution,
                "distribution_params": distribution_kwargs,
                "generated_at": datetime.now().isoformat()
            }
        }

    def _generate_random_data(self,
                            shape: List[int],
                            dtype: str,
                            distribution: str,
                            **kwargs) -> np.ndarray:
        """Generate random numpy array with specified distribution

        Supported distributions:
        - uniform: Uniform distribution (low, high, scale, bias)
        - normal: Normal/Gaussian distribution (mean, std)
        - randint: Random integers (low, high)
        - lognormal: Log-normal distribution (mean, std)
        - exponential: Exponential distribution (scale)
        - laplace: Laplace distribution (loc, scale)
        - cauchy: Cauchy distribution (loc, scale)
        - poisson: Poisson distribution (lam)
        - zipf: Zipf distribution (a)
        - standard_normal: Standard normal distribution (mean=0, std=1)
        - ones: All ones (value)
        - zeros: All zeros
        - identity: Identity matrix (only for 2D shapes)
        - orthogonal: Random orthogonal matrix (only for 2D shapes)
        - sparse: Sparse matrix with mostly zeros (sparsity, density)
        - """

        # Set seed if provided
        if self.seed is not None:
            np.random.seed(self.seed)

        # Map dtype
        np_dtype = self._map_dtype_to_numpy(dtype)

        # Check data size
        self._check_data_size(shape, dtype)

        # Generate data based on distribution
        if distribution == "uniform":
            low = kwargs.get("low", 0.0)
            high = kwargs.get("high", 1.0)
            scale = kwargs.get("scale", 1.0)
            bias = kwargs.get("bias", 0.0)
            data = np.random.uniform(low, high, shape).astype(np_dtype)
            data = data * scale + bias

        elif distribution == "normal":
            mean = kwargs.get("mean", 0.0)
            std = kwargs.get("std", 1.0)
            data = np.random.normal(mean, std, shape).astype(np_dtype)

        elif distribution == "standard_normal":
            data = np.random.randn(*shape).astype(np_dtype)

        elif distribution == "randint":
            low = kwargs.get("low", -100)
            high = kwargs.get("high", 100)
            data = np.random.randint(low, high, shape).astype(np_dtype)

        elif distribution == "lognormal":
            mean = kwargs.get("mean", 0.0)
            std = kwargs.get("std", 1.0)
            data = np.random.lognormal(mean, std, shape).astype(np_dtype)

        elif distribution == "exponential":
            scale = kwargs.get("scale", 1.0)
            data = np.random.exponential(scale, shape).astype(np_dtype)

        elif distribution == "laplace":
            loc = kwargs.get("loc", 0.0)
            scale = kwargs.get("scale", 1.0)
            data = np.random.laplace(loc, scale, shape).astype(np_dtype)

        elif distribution == "cauchy":
            loc = kwargs.get("loc", 0.0)
            scale = kwargs.get("scale", 1.0)
            # Cauchy: loc + scale * tan(PI * (U - 0.5))
            u = np.random.uniform(0, 1, shape)
            data = loc + scale * np.tan(np.pi * (u - 0.5))
            data = data.astype(np_dtype)

        elif distribution == "poisson":
            lam = kwargs.get("lam", 1.0)
            data = np.random.poisson(lam, shape).astype(np_dtype)

        elif distribution == "zipf":
            a = kwargs.get("a", 2.0)  # shape parameter > 1
            data = np.random.zipf(a, shape).astype(np_dtype)

        elif distribution == "ones":
            value = kwargs.get("value", 1.0)
            data = np.full(shape, value, dtype=np_dtype)

        elif distribution == "zeros":
            data = np.zeros(shape, dtype=np_dtype)

        elif distribution == "identity":
            if len(shape) != 2 or shape[0] != shape[1]:
                raise ValueError(f"Identity matrix requires square 2D shape, got {shape}")
            data = np.eye(shape[0], dtype=np_dtype)

        elif distribution == "orthogonal":
            if len(shape) != 2:
                raise ValueError(f"Orthogonal matrix requires 2D shape, got {shape}")
            # Generate random matrix with QR decomposition
            m, n = shape
            if m >= n:
                # Tall matrix: Q from QR of random matrix
                random_matrix = np.random.randn(m, n)
                q, _ = np.linalg.qr(random_matrix)
                data = q[:, :n].astype(np_dtype)
            else:
                # Wide matrix: Q from QR of random matrix
                random_matrix = np.random.randn(n, m)
                q, _ = np.linalg.qr(random_matrix)
                data = q[:m, :].astype(np_dtype)

        elif distribution == "sparse":
            density = kwargs.get("density", 0.1)  # fraction of non-zero elements
            sparsity = kwargs.get("sparsity", None)
            if sparsity is not None:
                density = 1.0 - sparsity

            # Generate sparse data
            data = np.zeros(shape, dtype=np_dtype)
            num_nonzeros = int(np.prod(shape) * density)

            # Random positions
            flat_indices = np.random.choice(np.prod(shape), num_nonzeros, replace=False)
            flat_data = np.random.randn(num_nonzeros).astype(np_dtype)
            data.flat[flat_indices] = flat_data

        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

        return data

    def _map_dtype_to_numpy(self, dtype: str) -> np.dtype:
        """Map dtype string to numpy dtype"""
        dtype_map = {
            "float16": np.float16,
            "float32": np.float32,
            "float64": np.float64,
            "bfloat16": np.float32,  # numpy doesn't have bfloat16, use float32
            "int8": np.int8,
            "int16": np.int16,
            "int32": np.int32,
            "int64": np.int64,
            "uint8": np.uint8,
            "bool": bool,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return dtype_map[dtype]

    def _generate_unique_path(self,
                             base_name: str,
                             shape: List[int],
                             dtype: str,
                             file_format: str) -> Path:
        """Generate unique file path based on timestamp with microseconds"""

        # Create timestamp string with microseconds
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # %f gives microseconds

        # Create filename
        shape_str = "x".join(map(str, shape))
        filename = f"{base_name}_{shape_str}_{dtype}_{timestamp}{file_format}"

        return self.output_dir / filename

    def _save_data(self, data: np.ndarray, file_path: Path, file_format: str):
        """Save data to file based on format"""

        if file_format == ".npy":
            np.save(file_path, data)
        elif file_format in [".pt", ".pth"]:
            torch.save(torch.from_numpy(data), file_path)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        logger.info(f"Saved random data to {file_path}")

    def _check_data_size(self, shape: List[int], dtype: str):
        """Check if data size is reasonable"""
        dtype_bytes = {
            "float16": 2, "bfloat16": 2, "float32": 4, "float64": 8,
            "int8": 1, "int16": 2, "int32": 4, "int64": 8
        }
        num_elements = np.prod(shape)
        size_mb = (num_elements * dtype_bytes.get(dtype, 4)) / (1024 * 1024)

        if size_mb > 1024:  # 1GB threshold
            logger.warning(f"Large data size: {size_mb:.2f}MB")


def generate_random_inputs_from_config(
    inputs_config: List[Dict[str, Any]],
    output_dir: Union[str, Path] = "./generated_data",
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Batch generate random inputs from configuration list

    Args:
        inputs_config: List of input configurations
        output_dir: Directory to save generated files
        seed: Random seed (same seed for all inputs)

    Returns:
        List of input configs with file_path fields

    Configuration format:
    [
        {
            "name": "input_a",
            "shape": [2048, 2048],
            "dtype": "float16",
            "_random": {
                "distribution": "uniform",
                "params": {"low": -1.0, "high": 1.0}
            }
        }
    ]
    """
    generator = RandomInputGenerator(output_dir=output_dir, seed=seed)

    results = []
    for idx, inp_config in enumerate(inputs_config):
        # Extract parameters
        base_name = inp_config.get("name", f"input_{idx}")
        shape = inp_config.get("shape")
        dtype = inp_config.get("dtype")

        if not shape or not dtype:
            raise ValueError(f"Input {idx} missing shape or dtype")

        # Get distribution config
        random_config = inp_config.get("_random", {})
        distribution = random_config.get("distribution", "uniform")
        distribution_params = random_config.get("params", {})

        # Generate
        result = generator.generate_input_config(
            base_name=base_name,
            shape=shape,
            dtype=dtype,
            distribution=distribution,
            **distribution_params
        )

        # Copy over name if specified
        if "name" in inp_config:
            result["name"] = inp_config["name"]

        results.append(result)

    return results
