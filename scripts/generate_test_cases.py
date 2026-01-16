#!/usr/bin/env python3
"""
Test Case Generator for InfiniMetrics

Generate batch test configurations with flexible shape distributions
and input tensor data generation.

Supported Shape Distributions:
    - fixed:           [1024, 768]
    - uniform:         uniform(min, max, samples)
    - log_uniform:     log_uniform(min, max, samples)
    - geometric:       geometric(start, ratio, count)
    - powers_of_2:     powers_of_2(min_exp, max_exp)
    - choice:          choice([values], count, replace=True)
    - randint:         randint(min, max, count, seed=None)

Supported Tensor Initializations:
    - zeros:           All zeros
    - ones:            All ones
    - random:          Uniform random [0, 1)
    - random_normal:   Normal distribution N(0, 1)
    - random_uniform:  Uniform [low, high]
    - identity:        Identity matrix (for 2D square)
    - diagonal:        Diagonal matrix
    - tril:            Lower triangular
    - triu:            Upper triangular

Usage:
    # Using shape distributions
    python scripts/generate_test_cases.py \\
        --operator matmul \\
        --shape-distribution "uniform(512, 4096, 4) uniform(512, 4096, 4)" \\
        --dtype float16 --device nvidia \\
        --output ./generated_tests/

    # With tensor generation
    python scripts/generate_test_cases.py \\
        --config test_templates/matmul_geometric.yaml \\
        --generate-tensors \\
        --tensor-init random_normal zeros \\
        --output ./generated_tests/
"""

import argparse
import json
import logging
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from itertools import product
from datetime import datetime
import copy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================
# Shape Distribution Generator
# ============================================

class ShapeDistributionGenerator:
    """Generate shapes based on distribution specifications."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional random seed."""
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            logger.info(f"Shape generator initialized with seed={seed}")

    def parse_distribution(self, dist_spec: Union[str, List]) -> List[int]:
        """
        Parse distribution specification into list of values.

        Args:
            dist_spec: Distribution spec string or list
                       e.g., "uniform(512, 2048, 4)" or [1024, 768]

        Returns:
            List of generated values
        """
        if isinstance(dist_spec, list):
            # Fixed shape: [1024, 768]
            return dist_spec

        if not isinstance(dist_spec, str):
            raise ValueError(f"Invalid distribution spec: {dist_spec}")

        # Parse distribution function calls
        dist_spec = dist_spec.strip()

        # Fixed shape in string format: "[1024, 768]"
        if dist_spec.startswith("[") and dist_spec.endswith("]"):
            return eval(dist_spec)

        # Distribution functions
        for dist_type in ["uniform", "log_uniform", "geometric",
                          "powers_of_2", "choice", "randint"]:
            if dist_spec.startswith(dist_type + "("):
                return self._parse_dist_function(dist_type, dist_spec)

        # Single integer: "1024"
        if dist_spec.isdigit():
            return [int(dist_spec)]

        raise ValueError(f"Unknown distribution specification: {dist_spec}")

    def _parse_dist_function(self, dist_type: str, func_str: str) -> List[int]:
        """Parse distribution function call."""
        # Extract arguments: "uniform(512, 2048, 4)" -> "512, 2048, 4"
        args_str = func_str[len(dist_type)+1:-1]  # Remove "func(" and ")"

        try:
            if dist_type == "uniform":
                return self._uniform(*self._parse_args(args_str))
            elif dist_type == "log_uniform":
                return self._log_uniform(*self._parse_args(args_str))
            elif dist_type == "geometric":
                return self._geometric(*self._parse_args(args_str))
            elif dist_type == "powers_of_2":
                return self._powers_of_2(*self._parse_args(args_str))
            elif dist_type == "choice":
                return self._choice(*self._parse_args(args_str, eval_first=True))
            elif dist_type == "randint":
                return self._randint(*self._parse_args(args_str))
        except Exception as e:
            raise ValueError(f"Error parsing {dist_type}({args_str}): {e}")

    def _parse_args(self, args_str: str, eval_first: bool = False) -> List:
        """Parse function arguments."""
        args = [a.strip() for a in args_str.split(",")]
        if eval_first:
            # First arg might be a list: "[512, 1024, 2048]"
            if args[0].startswith("["):
                args[0] = eval(args[0])
        return [int(a) if not isinstance(a, list) else a for a in args]

    def _uniform(self, min_val: int, max_val: int, samples: int) -> List[int]:
        """Uniform sampling in range [min, max]."""
        if samples == 1:
            return [int((min_val + max_val) / 2)]
        values = np.linspace(min_val, max_val, samples).astype(int)
        return values.tolist()

    def _log_uniform(self, min_val: int, max_val: int, samples: int) -> List[int]:
        """Log-uniform sampling (useful for size scales)."""
        if samples == 1:
            return [int(np.sqrt(min_val * max_val))]
        log_min = np.log(min_val)
        log_max = np.log(max_val)
        log_values = np.linspace(log_min, log_max, samples)
        values = np.exp(log_values).astype(int)
        return values.tolist()

    def _geometric(self, start: int, ratio: float, count: int) -> List[int]:
        """Geometric progression: start, start*ratio, start*ratio^2, ..."""
        values = [int(start * (ratio ** i)) for i in range(count)]
        return values

    def _powers_of_2(self, min_exp: int, max_exp: int) -> List[int]:
        """Generate powers of 2: 2^min_exp, ..., 2^max_exp."""
        return [2**i for i in range(min_exp, max_exp + 1)]

    def _choice(self, values: List[int], count: int, replace: bool = True) -> List[int]:
        """Random choice from values."""
        if replace:
            chosen = np.random.choice(values, count).tolist()
        else:
            chosen = np.random.choice(values, min(count, len(values)), replace=False).tolist()
        return chosen

    def _randint(self, min_val: int, max_val: int, count: int) -> List[int]:
        """Random integers in range [min, max]."""
        values = np.random.randint(min_val, max_val + 1, count).tolist()
        return sorted(values)

    def generate_shape_combinations(
        self,
        shape_distributions: List[str],
        max_combinations: Optional[int] = None
    ) -> List[List[int]]:
        """
        Generate all combinations of shapes from distributions.

        Args:
            shape_distributions: List of distribution specs for each dimension
                                e.g., ["uniform(512, 2048, 3)", "geometric(64, 2, 4)"]
            max_combinations: Limit number of combinations (for large grids)

        Returns:
            List of shape combinations
        """
        # Parse each dimension's distribution
        dim_values = []
        for i, dist_spec in enumerate(shape_distributions):
            values = self.parse_distribution(dist_spec)
            logger.debug(f"Dimension {i}: {dist_spec} -> {values}")
            dim_values.append(values)

        # Generate all combinations
        all_combinations = list(product(*dim_values))

        # Limit combinations if needed
        if max_combinations and len(all_combinations) > max_combinations:
            logger.warning(
                f"Too many combinations ({len(all_combinations)}), "
                f"limiting to {max_combinations}"
            )
            # Randomly sample
            indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
            all_combinations = [all_combinations[i] for i in sorted(indices)]

        # Convert tuples to lists
        return [list(comb) for comb in all_combinations]

    def generate_multi_tensor_shapes(
        self,
        tensor_distributions: List[List[str]],
        max_combinations: Optional[int] = None
    ) -> List[List[List[int]]]:
        """
        Generate shapes for multiple tensors.

        Args:
            tensor_distributions: List of distribution specs for each tensor
                                 e.g., [
                                     ["uniform(512, 2048, 2)", "uniform(512, 2048, 2)"],  # Tensor A
                                     ["uniform(512, 2048, 2)", "uniform(512, 2048, 2)"]   # Tensor B
                                 ]
            max_combinations: Limit number of combinations

        Returns:
            List of shape combinations for all tensors
        """
        # Generate shapes for each tensor independently
        tensor_shapes_list = []
        for i, tensor_dists in enumerate(tensor_distributions):
            shapes = self.generate_shape_combinations(tensor_dists, max_combinations)
            tensor_shapes_list.append(shapes)
            logger.debug(f"Tensor {i}: generated {len(shapes)} shapes")

        # Generate all combinations across tensors
        all_combinations = list(product(*tensor_shapes_list))

        # Limit if needed
        if max_combinations and len(all_combinations) > max_combinations:
            logger.warning(
                f"Too many tensor combinations ({len(all_combinations)}), "
                f"limiting to {max_combinations}"
            )
            indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
            all_combinations = [all_combinations[i] for i in sorted(indices)]

        # Format: [[shape_a, shape_b], [shape_a, shape_b], ...]
        return [list(combo) for combo in all_combinations]


# ============================================
# Tensor Generator
# ============================================

class TensorGenerator:
    """Generate and save input tensors with various initialization methods."""

    # Supported initialization methods
    INIT_METHODS = {
        "zeros",
        "ones",
        "random",
        "random_normal",
        "random_uniform",
        "identity",
        "diagonal",
        "tril",
        "triu",
    }

    def __init__(self, output_dir: Path, seed: Optional[int] = None):
        """
        Initialize tensor generator.

        Args:
            output_dir: Directory to save tensor files
            seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            logger.info(f"TensorGenerator initialized with seed={seed}")

    def get_dtype(self, dtype_str: str) -> np.dtype:
        """Map dtype string to numpy dtype."""
        dtype_map = {
            "float32": np.float32,
            "float16": np.float16,
            "bfloat16": np.float16,  # Fallback
            "int32": np.int32,
            "int64": np.int64,
            "bool": np.bool_,
        }
        return dtype_map.get(dtype_str.lower(), np.float32)

    def generate_tensor(
        self,
        shape: List[int],
        dtype: str,
        init_method: str,
        **kwargs
    ) -> np.ndarray:
        """
        Generate a tensor with specified initialization.

        Args:
            shape: Tensor shape
            dtype: Data type string
            init_method: Initialization method name
            **kwargs: Additional parameters for initialization

        Returns:
            Generated numpy array
        """
        np_dtype = self.get_dtype(dtype)

        if init_method == "zeros":
            tensor = np.zeros(shape, dtype=np_dtype)

        elif init_method == "ones":
            tensor = np.ones(shape, dtype=np_dtype)

        elif init_method == "random":
            tensor = np.random.random(shape).astype(np_dtype)

        elif init_method == "random_normal":
            mean = kwargs.get("mean", 0.0)
            std = kwargs.get("std", 1.0)
            tensor = np.random.normal(mean, std, shape).astype(np_dtype)

        elif init_method == "random_uniform":
            low = kwargs.get("low", -1.0)
            high = kwargs.get("high", 1.0)
            tensor = np.random.uniform(low, high, shape).astype(np_dtype)

        elif init_method == "identity":
            if len(shape) != 2 or shape[0] != shape[1]:
                raise ValueError(f"Identity matrix requires 2D square shape, got {shape}")
            tensor = np.eye(shape[0], dtype=np_dtype)

        elif init_method == "diagonal":
            if len(shape) != 2 or shape[0] != shape[1]:
                raise ValueError(f"Diagonal matrix requires 2D square shape, got {shape}")
            diag_val = kwargs.get("diag_val", 1.0)
            tensor = np.eye(shape[0], dtype=np_dtype) * diag_val

        elif init_method == "tril":
            if len(shape) != 2:
                raise ValueError(f"Lower triangular requires 2D shape, got {shape}")
            tensor = np.tril(np.ones(shape, dtype=np_dtype))

        elif init_method == "triu":
            if len(shape) != 2:
                raise ValueError(f"Upper triangular requires 2D shape, got {shape}")
            tensor = np.triu(np.ones(shape, dtype=np_dtype))

        else:
            raise ValueError(f"Unknown initialization method: {init_method}")

        logger.debug(f"Generated tensor: shape={shape}, dtype={dtype}, init={init_method}")
        return tensor

    def save_tensor(
        self,
        tensor: np.ndarray,
        tensor_name: str,
        run_id: str,
        dtype: str
    ) -> str:
        """
        Save tensor to .npy file.

        Args:
            tensor: Numpy array to save
            tensor_name: Name of the tensor (e.g., "in_0")
            run_id: Test run ID
            dtype: Data type string

        Returns:
            Relative path to saved tensor file
        """
        # Create filename: tensor_name_dtype.npy
        filename = f"{tensor_name}_{dtype}.npy"
        # Create subdirectory for this test case
        safe_run_id = run_id.replace(".", "_").replace("/", "_")
        tensor_dir = self.output_dir / "tensors" / safe_run_id
        tensor_dir.mkdir(parents=True, exist_ok=True)

        filepath = tensor_dir / filename
        np.save(filepath, tensor)

        # Return relative path from output_dir
        rel_path = filepath.relative_to(self.output_dir)
        logger.debug(f"Saved tensor to {rel_path}")
        return str(rel_path)

    def generate_and_save_inputs(
        self,
        input_specs: List[Dict[str, Any]],
        init_methods: List[str],
        run_id: str
    ) -> List[Dict[str, Any]]:
        """
        Generate and save all input tensors.

        Args:
            input_specs: List of input specifications with shape and dtype
            init_methods: List of initialization methods (one per input)
            run_id: Test run ID

        Returns:
            Updated input specs with 'data_path' field added
        """
        updated_inputs = []

        for i, (input_spec, init_method) in enumerate(zip(input_specs, init_methods)):
            # Generate tensor
            tensor = self.generate_tensor(
                shape=input_spec["shape"],
                dtype=input_spec["dtype"],
                init_method=init_method
            )

            # Save tensor
            data_path = self.save_tensor(
                tensor=tensor,
                tensor_name=input_spec["name"],
                run_id=run_id,
                dtype=input_spec["dtype"]
            )

            # Update input spec with data path
            updated_spec = input_spec.copy()
            updated_spec["data_path"] = data_path
            updated_inputs.append(updated_spec)

        return updated_inputs


# ============================================
# Template Generators
# ============================================

TEMPLATE_MATMUL = {
    "run_id": "{run_id}",
    "testcase": "operator.InfiniCore.Matmul.{desc}",
    "config": {
        "operator": "matmul",
        "device": "{device}",
        "torch_op": "torch.matmul",
        "infinicore_op": "infinicore.matmul",
        "inputs": [
            {"name": "in_0", "shape": None, "dtype": "{dtype}"},
            {"name": "in_1", "shape": None, "dtype": "{dtype}"}
        ],
        "attributes": [],
        "outputs": [{"name": "output", "shape": None, "dtype": "{dtype}"}],
        "warmup_iterations": 10,
        "measured_iterations": 100,
        "tolerance": {"atol": 1e-3, "rtol": 1e-3}
    },
    "metrics": [
        {"name": "operator.latency"},
        {"name": "operator.tensor_accuracy"},
        {"name": "operator.flops"},
        {"name": "operator.bandwidth"}
    ]
}

TEMPLATE_ADD = {
    "run_id": "{run_id}",
    "testcase": "operator.InfiniCore.Add.{desc}",
    "config": {
        "operator": "add",
        "device": "{device}",
        "torch_op": "torch.add",
        "infinicore_op": "infinicore.add",
        "inputs": [
            {"name": "a", "shape": None, "dtype": "{dtype}"},
            {"name": "b", "shape": None, "dtype": "{dtype}"}
        ],
        "attributes": [],
        "outputs": [{"name": "output", "shape": None, "dtype": "{dtype}"}],
        "warmup_iterations": 10,
        "measured_iterations": 100,
        "tolerance": {"atol": 1e-3, "rtol": 1e-3}
    },
    "metrics": [
        {"name": "operator.latency"},
        {"name": "operator.tensor_accuracy"},
        {"name": "operator.flops"},
        {"name": "operator.bandwidth"}
    ]
}

TEMPLATE_CONV2D = {
    "run_id": "{run_id}",
    "testcase": "operator.InfiniCore.Conv2D.{desc}",
    "config": {
        "operator": "conv2d",
        "device": "{device}",
        "torch_op": "torch.nn.functional.conv2d",
        "infinicore_op": "infinicore.conv2d",
        "inputs": [
            {"name": "input", "shape": None, "dtype": "{dtype}"},
            {"name": "weight", "shape": None, "dtype": "{dtype}"}
        ],
        "attributes": [
            {"name": "stride", "value": None},
            {"name": "padding", "value": None},
        ],
        "outputs": [{"name": "output", "shape": None, "dtype": "{dtype}"}],
        "warmup_iterations": 10,
        "measured_iterations": 100,
        "tolerance": {"atol": 1e-3, "rtol": 1e-3}
    },
    "metrics": [
        {"name": "operator.latency"},
        {"name": "operator.tensor_accuracy"},
        {"name": "operator.flops"},
        {"name": "operator.bandwidth"}
    ]
}


# ============================================
# Test Case Generator
# ============================================

class TestCaseGenerator:
    """Generate test case configurations."""

    def __init__(self, shape_seed: Optional[int] = None):
        """Initialize generator."""
        self.templates = {
            "matmul": TEMPLATE_MATMUL,
            "add": TEMPLATE_ADD,
            "conv2d": TEMPLATE_CONV2D,
        }
        self.shape_gen = ShapeDistributionGenerator(seed=shape_seed)
        self.tensor_gen = None  # Will be initialized if tensor generation is enabled

    def enable_tensor_generation(
        self,
        output_dir: Path,
        init_methods: List[str],
        tensor_seed: Optional[int] = None
    ):
        """
        Enable tensor data generation.

        Args:
            output_dir: Directory to save tensor files
            init_methods: List of initialization methods (one per input tensor)
            tensor_seed: Random seed for tensor generation
        """
        self.tensor_gen = TensorGenerator(output_dir, seed=tensor_seed)
        self.init_methods = init_methods
        logger.info(f"Tensor generation enabled with init methods: {init_methods}")

    def generate(
        self,
        operator: str,
        shapes: List[List[int]],
        dtype: str,
        device: str,
        run_id_prefix: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a single test case."""
        template = self.templates.get(operator.lower())
        if not template:
            raise ValueError(
                f"Unknown operator: {operator}. "
                f"Supported: {list(self.templates.keys())}"
            )

        if not run_id_prefix:
            run_id_prefix = f"test.{operator}"

        desc = self._generate_description(shapes, dtype, device)
        run_id = f"{run_id_prefix}.{desc}"

        shape_vars = self._prepare_shape_vars(operator, shapes, kwargs)

        config = self._fill_template(template, {
            "run_id": run_id,
            "desc": desc.replace(".", "_"),
            "device": device,
            "dtype": dtype,
            **shape_vars,
            **kwargs
        })

        # Generate tensor data if enabled
        if self.tensor_gen is not None:
            # Extract input specs from config
            inputs = config["config"]["inputs"]
            # Add tensor data paths
            updated_inputs = self.tensor_gen.generate_and_save_inputs(
                input_specs=inputs,
                init_methods=self.init_methods,
                run_id=run_id
            )
            # Update config with tensor paths
            config["config"]["inputs"] = updated_inputs

        return config

    def generate_from_distributions(
        self,
        operator: str,
        shape_distributions: List[List[str]],
        dtypes: List[str],
        devices: List[str],
        max_combinations: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate test cases from shape distributions.

        Args:
            operator: Operator name
            shape_distributions: Distributions for each tensor
                               e.g., [
                                   ["uniform(512, 2048, 3)", "uniform(512, 2048, 3)"],  # Tensor A
                                   ["uniform(512, 2048, 3)", "uniform(512, 2048, 3)"]   # Tensor B
                               ]
            dtypes: Data types to iterate
            devices: Devices to iterate
            max_combinations: Limit total combinations
            **kwargs: Operator-specific params

        Returns:
            List of test configurations
        """
        # Generate all shape combinations
        tensor_shapes_list = self.shape_gen.generate_multi_tensor_shapes(
            shape_distributions, max_combinations
        )

        # Generate configs for all combinations
        configs = []
        for shapes, dtype, device in product(tensor_shapes_list, dtypes, devices):
            config = self.generate(
                operator=operator,
                shapes=shapes,
                dtype=dtype,
                device=device,
                **kwargs
            )
            configs.append(config)

        return configs

    def _generate_description(
        self, shapes: List[List[int]], dtype: str, device: str
    ) -> str:
        """Generate description string."""
        shape_str = "_".join(["x".join(map(str, s)) for s in shapes])
        return f"{dtype}.{device}.{shape_str}"

    def _prepare_shape_vars(
        self, operator: str, shapes: List[List[int]], kwargs: Dict
    ) -> Dict[str, Any]:
        """Prepare shape variables for template. Returns a dict of paths to values."""
        if operator == "matmul":
            return {
                ("config", "inputs", 0, "shape"): shapes[0],
                ("config", "inputs", 1, "shape"): shapes[1],
                ("config", "outputs", 0, "shape"): self._compute_matmul_output(shapes[0], shapes[1])
            }
        elif operator == "add":
            return {
                ("config", "inputs", 0, "shape"): shapes[0],
                ("config", "inputs", 1, "shape"): shapes[0],
                ("config", "outputs", 0, "shape"): shapes[0],
            }
        elif operator == "conv2d":
            return {
                ("config", "inputs", 0, "shape"): shapes[0],
                ("config", "inputs", 1, "shape"): shapes[1],
                ("config", "outputs", 0, "shape"): self._compute_conv2d_output(
                    shapes[0], shapes[1],
                    kwargs.get("stride", 1),
                    kwargs.get("padding", 0)
                ),
                ("config", "attributes", 0, "value"): kwargs.get("stride", 1),
                ("config", "attributes", 1, "value"): kwargs.get("padding", 0),
            }
        return {}

    def _compute_matmul_output(self, shape_a: List[int], shape_b: List[int]) -> List[int]:
        """Compute output shape for matrix multiplication."""
        if len(shape_a) == 2 and len(shape_b) == 2:
            return [shape_a[0], shape_b[1]]
        elif len(shape_a) >= 2 and len(shape_b) >= 2:
            batch_dims = list(shape_a[:-2])
            return batch_dims + [shape_a[-2], shape_b[-1]]
        return shape_b

    def _compute_conv2d_output(
        self, input_shape: List[int], weight_shape: List[int],
        stride: int, padding: int
    ) -> List[int]:
        """Compute output shape for 2D convolution."""
        n, c_out = input_shape[0], weight_shape[0]
        h_in, w_in = input_shape[2], input_shape[3]
        kh, kw = weight_shape[2], weight_shape[3]

        h_out = (h_in + 2 * padding - kh) // stride + 1
        w_out = (w_in + 2 * padding - kw) // stride + 1

        return [n, c_out, h_out, w_out]

    def _fill_template(self, template: Any, vars: Dict) -> Any:
        """Recursively fill template with variables."""
        # First, handle path-based replacements (for shapes and values)
        path_vars = {k: v for k, v in vars.items() if isinstance(k, tuple)}
        string_vars = {k: v for k, v in vars.items() if isinstance(k, str)}

        # Deep copy template to avoid modifying original
        import copy
        result = copy.deepcopy(template)

        # Apply path-based replacements
        for path, value in path_vars.items():
            self._set_by_path(result, path, value)

        # Then apply string replacements
        self._fill_strings(result, string_vars)

        return result

    def _set_by_path(self, obj: Any, path: tuple, value: Any) -> None:
        """Set a value in a nested dict/list structure by path."""
        current = obj
        for key in path[:-1]:
            if isinstance(current, dict):
                current = current[key]
            elif isinstance(current, list):
                current = current[key]
        # Set final value
        final_key = path[-1]
        if isinstance(current, dict):
            current[final_key] = value
        elif isinstance(current, list):
            current[final_key] = value

    def _fill_strings(self, obj: Any, vars: Dict[str, Any]) -> None:
        """Recursively fill string placeholders in an object."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str):
                    obj[k] = self._replace_placeholders(v, vars)
                else:
                    self._fill_strings(v, vars)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, str):
                    obj[i] = self._replace_placeholders(item, vars)
                else:
                    self._fill_strings(item, vars)

    def _replace_placeholders(self, s: str, vars: Dict[str, Any]) -> str:
        """Replace {key} placeholders in a string."""
        result = s
        for key, value in vars.items():
            if isinstance(key, str) and not key.startswith("__"):
                result = result.replace("{" + key + "}", str(value))
        return result


# ============================================
# YAML Configuration Support
# ============================================

def generate_from_yaml(
    config_path: str,
    output_dir: Optional[Path] = None,
    generate_tensors: bool = False
) -> List[Dict[str, Any]]:
    """
    Generate test cases from YAML configuration.

    YAML format example:
        operator: matmul
        shape_distributions:
          - - "uniform(512, 2048, 3)"
            - "uniform(512, 2048, 3)"
          - - "uniform(512, 2048, 3)"
            - "uniform(512, 2048, 3)"
        dtypes: [float16, float32]
        devices: [nvidia]
        operator_params:
          stride: 1
          padding: 0
        tensor_generation:
          enabled: true
          init_methods: [random_normal, zeros]
          seed: 42
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    generator = TestCaseGenerator(
        shape_seed=config.get("random_seed", None)
    )

    # Enable tensor generation if configured
    tensor_config = config.get("tensor_generation", {})
    if generate_tensors and tensor_config.get("enabled", False):
        generator.enable_tensor_generation(
            output_dir=output_dir,
            init_methods=tensor_config.get("init_methods", ["random_normal"]),
            tensor_seed=tensor_config.get("seed", None)
        )

    return generator.generate_from_distributions(
        operator=config["operator"],
        shape_distributions=config["shape_distributions"],
        dtypes=config.get("dtypes", ["float16"]),
        devices=config.get("devices", ["nvidia"]),
        max_combinations=config.get("max_combinations", None),
        **config.get("operator_params", {})
    )


# ============================================
# CLI Interface
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate test case configurations with shape distributions and tensor data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Shape Distribution Examples:
  uniform(512, 2048, 4)        → [512, 1024, 1536, 2048]
  log_uniform(512, 8192, 3)    → [512, 2048, 8192]
  geometric(512, 2, 4)         → [512, 1024, 2048, 4096]
  powers_of_2(9, 12)           → [512, 1024, 2048, 4096]

Tensor Initialization Methods:
  zeros, ones, random, random_normal, random_uniform
  identity, diagonal, tril, triu

Usage Examples:
  # Generate test cases with tensor data
  python scripts/generate_test_cases.py \\
      --config test_templates/matmul_geometric.yaml \\
      --generate-tensors \\
      --tensor-init random_normal zeros \\
      --output ./generated_tests/

  # CLI with tensor generation
  python scripts/generate_test_cases.py \\
      --operator matmul \\
      --shape-distribution "powers_of_2(9, 11) powers_of_2(9, 11)" \\
      --dtype float16 --device nvidia \\
      --generate-tensors --tensor-init identity identity \\
      --output ./generated_tests/
        """
    )

    # Input mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", "-c", help="YAML configuration file")
    group.add_argument("--operator", "-op", help="Operator name (matmul, add, conv2d)")

    # Shape distributions
    parser.add_argument(
        "--shape-distribution", "-sd",
        nargs="+",
        help="Shape distributions (one spec per tensor dimension). "
             "For multi-tensor ops, separate tensor specs with '|'. "
             "E.g., 'uniform(512,2048,3) uniform(512,2048,3) | uniform(512,2048,3) uniform(512,2048,3)'"
    )

    # Common params
    parser.add_argument("--dtype", default="float16", help="Data type")
    parser.add_argument("--device", default="nvidia", help="Target device")
    parser.add_argument("--prefix", help="Run ID prefix")
    parser.add_argument("--seed", type=int, help="Random seed for shape generation")

    # Output options
    parser.add_argument("--output", "-o", default=".", help="Output directory or file")
    parser.add_argument("--max-combinations", type=int,
                       help="Limit maximum number of combinations")

    # Tensor generation options
    parser.add_argument("--generate-tensors", action="store_true",
                       help="Generate input tensor data files")
    parser.add_argument("--tensor-init", nargs="+",
                       help="Tensor initialization methods (one per input tensor). "
                            "Options: zeros, ones, random, random_normal, random_uniform, "
                            "identity, diagonal, tril, triu")
    parser.add_argument("--tensor-seed", type=int,
                       help="Random seed for tensor generation")

    # Conv2D specific
    parser.add_argument("--stride", type=int, default=1, help="Conv2D stride")
    parser.add_argument("--padding", type=int, default=0, help="Conv2D padding")

    args = parser.parse_args()

    output_path = Path(args.output)

    # Generate test cases
    if args.config:
        # Mode 1: Load from YAML
        logger.info(f"Loading config from {args.config}")
        configs = generate_from_yaml(
            config_path=args.config,
            output_dir=output_path,
            generate_tensors=args.generate_tensors
        )
    else:
        # Mode 2: CLI arguments
        if not args.shape_distribution:
            logger.error("--shape-distribution is required when using --operator")
            return 1

        # Parse shape distributions
        shape_distributions = []
        if "|" in " ".join(args.shape_distribution):
            # Multi-tensor case: split by "|"
            tensor_specs = " ".join(args.shape_distribution).split("|")
            for spec in tensor_specs:
                shape_distributions.append(spec.strip().split())
        else:
            # Single tensor or all dimensions for one tensor
            shape_distributions.append(args.shape_distribution)

        generator = TestCaseGenerator(shape_seed=args.seed)

        # Enable tensor generation if requested
        if args.generate_tensors:
            if not args.tensor_init:
                logger.error("--tensor-init is required when using --generate-tensors")
                return 1
            generator.enable_tensor_generation(
                output_dir=output_path,
                init_methods=args.tensor_init,
                tensor_seed=args.tensor_seed
            )

        configs = generator.generate_from_distributions(
            operator=args.operator,
            shape_distributions=shape_distributions,
            dtypes=[args.dtype],
            devices=[args.device],
            max_combinations=args.max_combinations,
            stride=args.stride,
            padding=args.padding,
        )

    # Save output
    if len(configs) == 1 and output_path.suffix == ".json":
        # Single file output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(configs[0], f, indent=2)
        logger.info(f"✅ Generated: {output_path}")
    else:
        # Directory output (multiple files)
        output_path.mkdir(parents=True, exist_ok=True)
        for i, config in enumerate(configs):
            filename = f"format_input_{config['testcase'].replace('.', '_')}.json"
            file_path = output_path / filename
            with open(file_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"✅ Generated: {file_path}")

    logger.info(f"📊 Total {len(configs)} test case(s) generated")
    if args.generate_tensors:
        logger.info(f"💾 Tensor data saved in: {output_path / 'tensors'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
