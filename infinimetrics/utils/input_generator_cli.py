#!/usr/bin/env python3
"""
CLI Tool for Random Input Generator
Flexible command-line interface with config file and parameter override
"""

import argparse
import json
import re
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from input_generator import RandomInputGenerator


def parse_input_spec(spec: str) -> Dict[str, Any]:
    """
    Parse input specification from command line

    Format: name:[shape]:dtype[:distribution]
    """
    pattern = r'^(\w+):\[([^\]]+)\]:(\w+)(?::(\w+))?$'
    match = re.match(pattern, spec)

    if not match:
        raise ValueError(f"Invalid input spec format: {spec}")

    name = match.group(1)
    shape_str = match.group(2)
    dtype = match.group(3)
    distribution = match.group(4) or "uniform"

    shape = [int(s.strip()) for s in shape_str.split(',')]

    return {
        "name": name,
        "shape": shape,
        "dtype": dtype,
        "distribution": distribution
    }


def generate_default_config() -> Dict[str, Any]:
    """Generate default configuration template"""
    return {
        "run_id": "template.matmul",
        "testcase": "operator.InfiniCore.MatMul",
        "config": {
            "operator": "matmul",
            "device": "nvidia",
            "data_base_dir": "./generated_data",
            "inputs": [
                {
                    "name": "a",
                    "shape": [1024, 1024],
                    "dtype": "float16",
                    "_random": {
                        "distribution": "uniform",
                        "params": {"low": -1.0, "high": 1.0}
                    }
                },
                {
                    "name": "b",
                    "shape": [1024, 1024],
                    "dtype": "float16",
                    "_random": {
                        "distribution": "uniform",
                        "params": {"low": -1.0, "high": 1.0}
                    }
                }
            ],
            "attributes": [],
            "outputs": [
                {
                    "name": "output",
                    "shape": [1024, 1024],
                    "dtype": "float16"
                }
            ],
            "warmup_iterations": 10,
            "measured_iterations": 100,
            "tolerance": {"atol": 1e-3, "rtol": 1e-3}
        },
        "metrics": [
            {"name": "operator.latency"},
            {"name": "operator.tensor_accuracy"}
        ]
    }


def load_config(config_path: str) -> Optional[Dict[str, Any]]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else data[0]
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return None


def vary_shape(base_shape: List[int], index: int, mode: str = "random", var_range: int = 100) -> List[int]:
    """Generate varied shape based on mode"""
    if mode == "none":
        return base_shape.copy()

    elif mode == "random":
        # Random variation within range
        return [
            max(1, dim + random.randint(-var_range, var_range))
            for dim in base_shape
        ]

    elif mode == "progressive":
        # Progressive growth: each input is larger
        growth = var_range * index
        return [dim + growth for dim in base_shape]

    return base_shape.copy()


def calculate_output_shape(operator_type: str, input_shapes: List[List[int]]) -> List[int]:
    """
    Calculate output shape based on operator type and input shapes.

    Supports common operators:
    - matmul: [M, K] @ [K, N] -> [M, N]
    - add: element-wise, output shape = input shape
    - sub: element-wise, output shape = input shape
    - mul: element-wise, output shape = input shape
    - div: element-wise, output shape = input shape
    - transpose: [M, N] -> [N, M]
    - softmax: output shape = input shape
    - relu: output shape = input shape
    - gelu: output shape = input shape
    - layer_norm: output shape = input shape
    - batch_norm: output shape = input shape
    - conv2d: [N, C, H, W] with kernel -> complex calculation
    """
    op_lower = operator_type.lower()

    if "matmul" in op_lower or "gemm" in op_lower or "linear" in op_lower:
        # Matrix multiplication: [M, K] @ [K, N] -> [M, N]
        if len(input_shapes) >= 2:
            return [input_shapes[0][0], input_shapes[1][1]]
        else:
            return input_shapes[0]

    elif op_lower in ["add", "sub", "mul", "div", "subtract", "multiply", "divide"]:
        # Element-wise operations: output shape = input shape (all inputs should have same shape)
        if input_shapes:
            return input_shapes[0].copy()
        return []

    elif "transpose" in op_lower:
        # Transpose: [M, N] -> [N, M]
        if input_shapes and len(input_shapes[0]) == 2:
            return [input_shapes[0][1], input_shapes[0][0]]
        return input_shapes[0] if input_shapes else []

    elif op_lower in ["softmax", "relu", "gelu", "sigmoid", "tanh", "layer_norm", "batch_norm", "group_norm"]:
        # Activation and normalization: output shape = input shape
        if input_shapes:
            return input_shapes[0].copy()
        return []

    elif "conv2d" in op_lower or "conv" in op_lower:
        # Simplified conv2d: assume padding='same' so output shape = input shape
        if input_shapes and len(input_shapes[0]) == 4:
            return input_shapes[0].copy()
        return input_shapes[0] if input_shapes else []

    elif "pool" in op_lower:
        # Pooling: output shape = input shape (assuming padding='same')
        if input_shapes:
            return input_shapes[0].copy()
        return []

    elif "flatten" in op_lower:
        # Flatten: [B, C, H, W] -> [B, C*H*W]
        if input_shapes and len(input_shapes[0]) >= 2:
            batch_size = input_shapes[0][0]
            rest = 1
            for dim in input_shapes[0][1:]:
                rest *= dim
            return [batch_size, rest]
        return input_shapes[0] if input_shapes else []

    elif "reshape" in op_lower:
        # Reshape: output shape is specified in attributes, not computed
        # Return input shape as fallback
        if input_shapes:
            return input_shapes[0].copy()
        return []

    elif "concat" in op_lower:
        # Concatenate: sum along the concat dimension
        # For simplicity, assume concat on last dimension
        if len(input_shapes) >= 2:
            result = input_shapes[0].copy()
            if len(result) >= 1:
                result[-1] = sum(shape[-1] for shape in input_shapes)
            return result
        return input_shapes[0] if input_shapes else []

    else:
        # Default: assume output shape = first input shape
        if input_shapes:
            return input_shapes[0].copy()
        return []


def override_config_with_cli(
    config: Dict[str, Any],
    shape: Optional[List[int]] = None,
    shapes: Optional[List[str]] = None,
    dtype: Optional[str] = None,
    distribution: Optional[str] = None,
    count: Optional[int] = None,
    shape_variation: str = "random",
    shape_var_range: int = 100
) -> Dict[str, Any]:
    """Override configuration with CLI parameters"""
    config = config.copy()
    config["config"] = config.get("config", {}).copy()

    inputs = config["config"].get("inputs", [])

    # If specific shapes provided, use them
    if shapes is not None:
        import ast
        parsed_shapes = []
        for s in shapes:
            try:
                parsed_shapes.append(ast.literal_eval(s))
            except:
                parsed_shapes.append(list(shape)) if shape else None

        # Extend inputs list to match number of shapes
        while len(inputs) < len(parsed_shapes):
            base_input = inputs[0].copy() if inputs else {}
            base_input["name"] = chr(ord('a') + len(inputs))
            inputs.append(base_input)

        # Assign specific shapes
        for i, inp in enumerate(inputs):
            if i < len(parsed_shapes) and parsed_shapes[i] is not None:
                inp["shape"] = list(parsed_shapes[i])

    # Override single shape for all inputs (base shape)
    elif shape is not None:
        for inp in inputs:
            inp["shape"] = list(shape)

    # Override dtype for all inputs
    if dtype is not None:
        for inp in inputs:
            inp["dtype"] = dtype

    # Override distribution for all inputs
    if distribution is not None:
        for inp in inputs:
            if "_random" in inp:
                inp["_random"]["distribution"] = distribution

    # When count is specified, expand to N test cases
    # Each test case contains all inputs with varied shapes
    if count is not None and count > 1:
        base_inputs = inputs.copy()
        expanded_inputs = []

        for test_idx in range(count):
            for inp in base_inputs:
                new_input = inp.copy()
                # Apply shape variation based on test case index
                if shape_variation != "none":
                    base_shape = inp.get("shape", [1024, 1024])
                    new_input["shape"] = vary_shape(
                        base_shape,
                        test_idx,
                        mode=shape_variation,
                        var_range=shape_var_range
                    )
                expanded_inputs.append(new_input)

        inputs = expanded_inputs

    config["config"]["inputs"] = inputs
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate random tensor inputs for testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate template config file
  python input_generator_cli.py

  # Generate from config file
  python input_generator_cli.py --config format_input_matmul.json

  # Override config parameters
  python input_generator_cli.py --config format_input_matmul.json --shape 2048 2048 --dtype float32

  # Auto-generate multiple inputs
  python input_generator_cli.py --config format_input_matmul.json --auto --count 3

  # Specify inputs directly (no config file)
  python input_generator_cli.py --inputs "a:[1024,1024]:float16" "b:[1024,1024]:float16"
        """
    )

    parser.add_argument(
        "--config", "-c",
        help="Configuration JSON file (generates template if not specified)"
    )
    parser.add_argument(
        "--inputs", "-i",
        nargs='+',
        help="Input specifications (format: name:[shape]:dtype[:distribution]). "
             "Overrides --config if specified."
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-generate inputs based on config or defaults"
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        help="Number of inputs to auto-generate"
    )
    parser.add_argument(
        "--shape",
        nargs='+',
        type=int,
        help="Base shape for all inputs (e.g., --shape 1024 1024)"
    )
    parser.add_argument(
        "--shapes",
        nargs='+',
        action='append',
        help="Specific shapes for each input (e.g., --shapes '[1024,1024]' '[2048,1024]')"
    )
    parser.add_argument(
        "--shape-variation",
        choices=["none", "random", "progressive"],
        default="random",
        help="How to vary shapes for multiple inputs (default: random)"
    )
    parser.add_argument(
        "--shape-var-range",
        type=int,
        default=100,
        help="Range for random shape variation (default: 100)"
    )
    parser.add_argument(
        "--dtype",
        help="Override data type (e.g., --dtype float16)"
    )
    parser.add_argument(
        "--distribution", "-d",
        choices=[
            "uniform", "normal", "standard_normal", "randint",
            "lognormal", "exponential", "laplace", "cauchy",
            "poisson", "zipf", "ones", "zeros", "identity",
            "orthogonal", "sparse"
        ],
        help="Override distribution type for all inputs"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./generated_data",
        help="Output directory (default: ./generated_data)"
    )
    parser.add_argument(
        "--output-json",
        help="Save final config with file paths"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--format",
        default=".npy",
        choices=[".npy", ".pt", ".pth"],
        help="File format (default: .npy)"
    )
    parser.add_argument(
        "--low",
        type=float,
        help="Low bound for uniform/randint"
    )
    parser.add_argument(
        "--high",
        type=float,
        help="High bound for uniform/randint"
    )
    parser.add_argument(
        "--mean",
        type=float,
        default=0.0,
        help="Mean for normal distribution"
    )
    parser.add_argument(
        "--std",
        type=float,
        default=1.0,
        help="Standard deviation for normal distribution"
    )
    parser.add_argument(
        "--scale",
        type=float,
        help="Scale parameter (for uniform, exponential, laplace, cauchy)"
    )
    parser.add_argument(
        "--bias",
        type=float,
        help="Bias parameter (for uniform distribution)"
    )
    parser.add_argument(
        "--loc",
        type=float,
        help="Location parameter (for laplace, cauchy)"
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=1.0,
        help="Lambda parameter for poisson distribution"
    )
    parser.add_argument(
        "--zipf-a",
        type=float,
        dest="zipf_a",
        default=2.0,
        help="Shape parameter for zipf distribution (must be > 1)"
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.1,
        help="Density of non-zero elements for sparse distribution (0-1)"
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        help="Sparsity for sparse distribution (0-1, alternative to density)"
    )
    parser.add_argument(
        "--value",
        type=float,
        default=1.0,
        help="Value for ones distribution"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Case 1: No arguments - generate template config
    if not args.config and not args.inputs:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        template_path = f"format_input_matmul_{timestamp}.json"
        config = generate_default_config()

        with open(template_path, 'w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"✓ Generated template config: {template_path}")
        print(f"\nEdit this file, then run:")
        print(f"  python infinimetrics/utils/input_generator_cli.py --config {template_path}")
        return 0

    # Case 2: Load or generate config
    if args.inputs:
        # Parse inputs directly from CLI
        inputs_config = []
        for spec_str in args.inputs:
            try:
                inputs_config.append(parse_input_spec(spec_str))
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1

        # Build a minimal config
        config = {
            "run_id": "cli_generated",
            "testcase": "operator.InfiniCore.Custom",
            "config": {
                "operator": "custom",
                "device": "nvidia",
                "data_base_dir": args.output_dir,
                "inputs": []
            }
        }

        # Convert inputs_config to config format
        for inp in inputs_config:
            config["config"]["inputs"].append({
                "name": inp["name"],
                "shape": inp["shape"],
                "dtype": inp["dtype"],
                "_random": {
                    "distribution": inp["distribution"],
                    "params": {}
                }
            })
    else:
        # Load from config file
        config = load_config(args.config)
        if config is None:
            return 1

        # Parse shapes if provided
        shapes = None
        if args.shapes:
            import ast
            shapes = [ast.literal_eval(s) for s in args.shapes]

        # Override with CLI parameters
        config = override_config_with_cli(
            config,
            shape=args.shape,
            shapes=shapes,
            dtype=args.dtype,
            distribution=args.distribution,
            count=args.count,
            shape_variation=args.shape_variation,
            shape_var_range=args.shape_var_range
        )

    # Extract inputs config
    inputs_config_list = config.get("config", {}).get("inputs", [])

    # Apply auto mode if requested
    if args.auto:
        print(f"Auto-generating {len(inputs_config_list)} inputs...")

    # Build distribution parameters
    dist_params = {}
    if args.low is not None:
        dist_params["low"] = args.low
    if args.high is not None:
        dist_params["high"] = args.high
    if args.mean is not None:
        dist_params["mean"] = args.mean
    if args.std is not None:
        dist_params["std"] = args.std
    if args.scale is not None:
        dist_params["scale"] = args.scale
    if args.bias is not None:
        dist_params["bias"] = args.bias
    if args.loc is not None:
        dist_params["loc"] = args.loc
    if args.lam is not None:
        dist_params["lam"] = args.lam
    if args.zipf_a is not None:
        dist_params["a"] = args.zipf_a
    if args.density is not None:
        dist_params["density"] = args.density
    if args.sparsity is not None:
        dist_params["sparsity"] = args.sparsity
    if args.value is not None:
        dist_params["value"] = args.value

    # Print what we're about to generate
    print(f"\nGenerating {len(inputs_config_list)} input(s):")
    for inp in inputs_config_list:
        shape = "x".join(map(str, inp.get("shape", [])))
        dist = inp.get("_random", {}).get("distribution", "uniform")
        print(f"  - {inp['name']}: shape={shape}, dtype={inp['dtype']}, dist={dist}")

    # Generate inputs
    try:
        generator = RandomInputGenerator(
            output_dir=args.output_dir,
            default_format=args.format,
            seed=args.seed
        )

        generated_inputs = []
        for inp_config in inputs_config_list:
            # Get distribution config
            random_config = inp_config.get("_random", {})
            distribution = random_config.get("distribution", "uniform")
            params = random_config.get("params", {})

            # Override with CLI params
            params.update(dist_params)

            result = generator.generate_input_config(
                base_name=inp_config["name"],
                shape=inp_config["shape"],
                dtype=inp_config["dtype"],
                distribution=distribution,
                **params
            )
            generated_inputs.append(result)

        print(f"\n✓ Successfully generated {len(generated_inputs)} file(s):")
        for inp in generated_inputs:
            file_path = inp["file_path"]
            shape = "x".join(map(str, inp["shape"]))
            print(f"  - {inp['name']}: {file_path}")

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # Save output config(s) if requested
    if args.output_json:
        try:
            output_path = Path(args.output_json)
            config["config"]["data_base_dir"] = str(Path(args.output_dir).absolute())

            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # If count is specified, generate N complete test configs
            # Each config contains all inputs (e.g., a, b, c...)
            if args.count is not None:
                # Calculate original number of inputs per test case
                # (before expansion by count parameter)
                if args.count > 0 and len(generated_inputs) % args.count == 0:
                    num_inputs_per_test = len(generated_inputs) // args.count
                else:
                    num_inputs_per_test = 2  # fallback to 2 (common for matmul, add, etc.)

                num_test_cases = args.count
                operator_type = config["config"].get("operator", "unknown")

                saved_files = []
                for test_idx in range(num_test_cases):
                    # Create individual config with all inputs for this test case
                    individual_config = config.copy()
                    individual_config["config"] = config["config"].copy()
                    individual_config["config"]["inputs"] = []

                    # Collect input shapes for this test case
                    input_shapes_for_test = []

                    # Add all inputs for this test case
                    for input_idx in range(num_inputs_per_test):
                        global_idx = test_idx * num_inputs_per_test + input_idx
                        if global_idx < len(generated_inputs):
                            inp = generated_inputs[global_idx]
                            # Convert file_path to absolute path
                            file_path = Path(inp["file_path"])
                            if not file_path.is_absolute():
                                file_path = file_path.absolute()
                            individual_config["config"]["inputs"].append({
                                "name": inp["name"],
                                "file_path": str(file_path),
                                "dtype": inp["dtype"],
                                "shape": inp["shape"]
                            })
                            input_shapes_for_test.append(inp["shape"])

                    # Update output shapes based on input shapes
                    if "outputs" in individual_config["config"] and individual_config["config"]["outputs"]:
                        calculated_output_shape = calculate_output_shape(operator_type, input_shapes_for_test)
                        if calculated_output_shape:
                            for output in individual_config["config"]["outputs"]:
                                output["shape"] = calculated_output_shape

                    # Update run_id to make it unique
                    individual_config["run_id"] = f"{config.get('run_id', 'test')}.case{test_idx + 1}"

                    # Generate filename
                    if output_path.suffix:
                        base_name = output_path.stem
                        output_file = output_path.parent / f"{base_name}_case{test_idx + 1}_{timestamp}{output_path.suffix}"
                    else:
                        output_file = output_path / f"config_case{test_idx + 1}_{timestamp}.json"

                    with open(output_file, 'w') as f:
                        json.dump(individual_config, f, indent=2, ensure_ascii=False)
                    saved_files.append(str(output_file))

                print(f"\n✓ Saved {len(saved_files)} configuration file(s):")
                for f in saved_files:
                    print(f"  - {f}")
            else:
                # No count specified: save single config with all inputs
                individual_config = config.copy()
                individual_config["config"] = config["config"].copy()
                individual_config["config"]["inputs"] = []

                input_shapes_for_test = []

                for inp in generated_inputs:
                    # Convert file_path to absolute path
                    file_path = Path(inp["file_path"])
                    if not file_path.is_absolute():
                        file_path = file_path.absolute()
                    individual_config["config"]["inputs"].append({
                        "name": inp["name"],
                        "file_path": str(file_path),
                        "dtype": inp["dtype"],
                        "shape": inp["shape"]
                    })
                    input_shapes_for_test.append(inp["shape"])

                # Update output shapes based on input shapes
                operator_type = config["config"].get("operator", "unknown")
                if "outputs" in individual_config["config"] and individual_config["config"]["outputs"]:
                    calculated_output_shape = calculate_output_shape(operator_type, input_shapes_for_test)
                    if calculated_output_shape:
                        for output in individual_config["config"]["outputs"]:
                            output["shape"] = calculated_output_shape

                with open(output_path, 'w') as f:
                    json.dump(individual_config, f, indent=2, ensure_ascii=False)

                print(f"\n✓ Configuration saved to: {output_path}")

        except Exception as e:
            print(f"✗ Error saving output: {e}", file=sys.stderr)
            return 1

    print("\n✓ Generation complete!")

    # Print next steps
    if not args.output_json:
        print(f"\nNext steps:")
        print(f"  python infinimetrics/utils/input_generator_cli.py \\")
        print(f"      --config {args.config or 'your_config.json'} \\")
        print(f"      --output-json config_output.json")
        print(f"\nNote: This will generate N config files (one per input)")
        print(f"      with names like: config_output_input_a_20250119_143025.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
