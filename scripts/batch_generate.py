#!/usr/bin/env python3
"""
Batch Input Generator for InfiniMetrics

Simple batch generation tool for creating multiple test cases with shape variations.
Uses modular design for easy operator and JSON format extensions.
"""

import argparse
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from infinimetrics.utils.input_generator import RandomInputGenerator
from operator_specs import get_operator_spec, list_supported_operators
from json_generator import InfiniMetricsJSONGenerator, JSONFileWriter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments for batch generation."""
    parser = argparse.ArgumentParser(
        description="Generate batch test inputs for InfiniMetrics with shape variations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available benchmark shapes
  python scripts/batch_generate.py --list-benchmark-shapes

  # Generate 10 matmul tests with fixed shapes
  python scripts/batch_generate.py --operator matmul --count 10 --shape-base 1024 1024 --dtype float16 --output ./batch_output

  # Generate 20 add tests with random shape variation
  python scripts/batch_generate.py --operator add --count 20 --shape-base 1024 1024 --shape-range 512 --shape-variation random --dtype float16 --output ./batch_output

  # Generate benchmark shapes (performance testing)
  python scripts/batch_generate.py --operator matmul --count 18 --shape-variation benchmark --dtype float16 --output ./benchmark_test

Available operators:
  matmul, add, sub, mul, div

Shape Variation Modes:
  - none:         All tests use the same shape (from --shape-base)
  - random:       Random variation: shape_base ± shape_range (uniform)
  - progressive:  Progressive growth: shape_base + (shape_range × index)
  - benchmark:    Performance benchmark shapes (preset shapes for comprehensive testing)
        """
    )

    # Special flag to list benchmark shapes
    parser.add_argument(
        "--list-benchmark-shapes",
        action="store_true",
        help="List all available benchmark shapes and exit"
    )

    # Required arguments
    parser.add_argument(
        "--operator", "-op",
        required=False,  # Will be checked after parse_args
        choices=list_supported_operators(),
        help="Operator type"
    )

    parser.add_argument(
        "--count", "-n",
        type=int,
        required=False,  # Will be checked after parse_args
        help="Number of test cases to generate"
    )

    parser.add_argument(
        "--shape-base",
        nargs="+",
        type=int,
        required=False,  # Will be checked after parse_args
        help="Base shape dimensions (e.g., 1024 1024 for matmul)"
    )

    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32", "float64", "int32", "int64"],
        help="Data type for all tensors (default: float16)"
    )

    parser.add_argument(
        "--output", "-o",
        default="./batch_output",
        help="Output directory for batch results (default: ./batch_output)"
    )

    # Shape variation options
    parser.add_argument(
        "--shape-variation",
        choices=["none", "random", "progressive", "benchmark"],
        default="none",
        help="Shape variation mode (default: none)"
    )

    parser.add_argument(
        "--shape-range",
        type=int,
        default=0,
        help="Range for shape variation (default: 0, no variation)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )

    # Distribution options
    parser.add_argument(
        "--mixed-distributions",
        action="store_true",
        help="Use different distributions for each input"
    )

    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=[
            "uniform", "normal", "standard_normal", "randint",
            "lognormal", "exponential", "laplace", "cauchy",
            "poisson", "zipf", "ones", "zeros", "identity", "orthogonal", "sparse"
        ],
        help="Specific distributions to use (with --mixed-distributions)"
    )

    # Format options
    parser.add_argument(
        "--format",
        choices=[".npy", ".pt", ".pth"],
        default=".npy",
        help="Data file format (default: .npy)"
    )

    # Performance options
    parser.add_argument(
        "--device",
        default="nvidia",
        help="Target device (default: nvidia)"
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations (default: 10)"
    )

    parser.add_argument(
        "--measured",
        type=int,
        default=100,
        help="Measured iterations (default: 100)"
    )

    args = parser.parse_args()

    # Handle --list-benchmark-shapes
    if args.list_benchmark_shapes:
        print("=" * 60)
        print("Available Benchmark Shapes (MatMul & Element-wise)")
        print("=" * 60)
        shapes = BenchmarkShapeGenerator.list_shapes(ndim=2)
        print(f"Total: {len(shapes)} preset shapes\n")
        for i, shape in enumerate(shapes):
            print(f"  {i:2d}. {shape[0]:4d} × {shape[1]:4d}")
        print("=" * 60)
        sys.exit(0)

    # Check required arguments for normal operation
    if not args.operator:
        parser.error("--operator is required (unless using --list-benchmark-shapes)")
    if not args.count:
        parser.error("--count is required (unless using --list-benchmark-shapes)")
    if not args.shape_base:
        parser.error("--shape-base is required (unless using --list-benchmark-shapes)")

    return args


class ShapeVariationGenerator:
    """Generate shapes based on variation mode."""

    def __init__(self,
                 base_shape: List[int],
                 variation_mode: str = "none",
                 variation_range: int = 0,
                 seed: Optional[int] = None):
        """
        Args:
            base_shape: Base shape template
            variation_mode: "none", "random", "progressive", or "benchmark"
            variation_range: Range for variation
            seed: Random seed for reproducibility
        """
        self.base_shape = base_shape
        self.mode = variation_mode
        self.range = variation_range
        self.seed = seed

        if seed is not None:
            random.seed(seed)

        # Initialize benchmark generator if needed
        if self.mode == "benchmark":
            self.benchmark_gen = BenchmarkShapeGenerator(seed)

    def generate_shape(self, index: int) -> List[int]:
        """
        Generate a shape for the given test case index.

        Args:
            index: Test case index (0-based)

        Returns:
            Shape as list of integers
        """
        if self.mode == "none":
            return self.base_shape.copy()

        elif self.mode == "random":
            # Random variation: base ± range
            return [
                max(1, dim + random.randint(-self.range, self.range))
                for dim in self.base_shape
            ]

        elif self.mode == "progressive":
            # Progressive: base + (range × index)
            return [
                max(1, dim + self.range * index)
                for dim in self.base_shape
            ]

        elif self.mode == "benchmark":
            # Benchmark preset shapes
            return self.benchmark_gen.get_shape(index, len(self.base_shape))

        else:
            raise ValueError(f"Unknown variation mode: {self.mode}")


class BenchmarkShapeGenerator:
    """
    Generate benchmark-optimized shapes for performance testing.

    These shapes are designed to test different performance characteristics:
    - Small matrices: Cache and latency effects
    - Medium matrices: Balanced compute/memory
    - Large matrices: Memory bandwidth bound
    - Square vs rectangular: Different access patterns
    - Power-of-2 vs non-power-of-2: Alignment effects
    """

    # Preset benchmark shapes for 2D matrices (matmul)
    BENCHMARK_2D_SHAPES = [
        # Small matrices - cache/latency tests
        (64, 64),
        (128, 128),
        (256, 256),

        # Medium matrices - balanced tests
        (512, 512),
        (768, 768),
        (1024, 768),  # Rectangular
        (768, 1024),  # Rectangular

        # Large matrices - memory bandwidth tests
        (1024, 1024),
        (1536, 1536),
        (2048, 2048),

        # Very large - stress tests
        (3072, 3072),
        (4096, 4096),

        # Non-power-of-2 - alignment tests
        (1000, 1000),
        (1008, 1008),  # Aligned but not power-of-2
        (1020, 1020),

        # Extreme aspect ratios
        (128, 4096),
        (4096, 128),
    ]

    # Preset benchmark shapes for element-wise ops (add, sub, mul, div)
    BENCHMARK_ELEMENTWISE_SHAPES = [
        # Small - cache tests
        (64, 64),
        (128, 128),

        # Medium - balanced
        (512, 512),
        (768, 768),
        (1024, 1024),

        # Large - bandwidth tests
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),

        # Non-power-of-2
        (1000, 1000),
        (1020, 1020),
    ]

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize benchmark generator.

        Args:
            seed: Random seed (for future extensibility)
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def get_shape(self, index: int, ndim: int) -> List[int]:
        """
        Get benchmark shape for given index.

        Args:
            index: Test case index (0-based)
            ndim: Number of dimensions (typically 2 for matrices)

        Returns:
            Shape as list of integers
        """
        if ndim == 2:
            shapes = self.BENCHMARK_2D_SHAPES
        else:
            # For higher dimensions, use 2D shapes and extend
            shapes = self.BENCHMARK_2D_SHAPES

        if index >= len(shapes):
            raise ValueError(
                f"Benchmark mode only supports {len(shapes)} test cases. "
                f"Requested index {index}, but --count should be <= {len(shapes)}."
            )

        shape = shapes[index]
        return list(shape)

    @classmethod
    def get_max_count(cls, ndim: int = 2) -> int:
        """Get maximum number of benchmark shapes available."""
        if ndim == 2:
            return len(cls.BENCHMARK_2D_SHAPES)
        else:
            return len(cls.BENCHMARK_2D_SHAPES)

    @classmethod
    def list_shapes(cls, ndim: int = 2) -> List[tuple]:
        """List all available benchmark shapes."""
        if ndim == 2:
            return cls.BENCHMARK_2D_SHAPES.copy()
        else:
            return cls.BENCHMARK_2D_SHAPES.copy()


class DistributionSelector:
    """Select distributions for input tensors."""

    DISTRIBUTIONS = [
        "uniform", "normal", "standard_normal", "randint",
        "lognormal", "exponential", "laplace", "cauchy",
        "poisson", "zipf", "ones", "zeros", "identity",
        "orthogonal", "sparse"
    ]

    def __init__(self,
                 mixed_mode: bool = False,
                 specific_dists: Optional[List[str]] = None,
                 seed: Optional[int] = None):
        """
        Args:
            mixed_mode: Use different distribution per input
            specific_dists: List of specific distributions to use
            seed: Random seed
        """
        self.mixed_mode = mixed_mode
        self.specific_dists = specific_dists or []
        self.seed = seed

        if seed is not None:
            random.seed(seed)

    def get_distribution(self, input_index: int, input_name: str) -> str:
        """
        Get distribution for a specific input.

        Args:
            input_index: Index of input tensor
            input_name: Name of input tensor

        Returns:
            Distribution name
        """
        if not self.mixed_mode:
            return "uniform"  # Default

        if self.specific_dists:
            # Use provided distributions (cycle if needed)
            return self.specific_dists[input_index % len(self.specific_dists)]

        # Auto-select based on input characteristics
        if "weight" in input_name or "kernel" in input_name:
            return random.choice(["normal", "orthogonal"])
        elif "bias" in input_name:
            return "zeros"
        elif "mask" in input_name:
            return "sparse"
        else:
            return "uniform"

    def get_distribution_params(self, distribution: str) -> Dict[str, Any]:
        """Get default parameters for a distribution."""
        params = {
            "uniform": {"low": -1.0, "high": 1.0},
            "normal": {"mean": 0.0, "std": 0.1},
            "standard_normal": {},
            "randint": {"low": -100, "high": 100},
            "lognormal": {"mean": 0.0, "std": 0.5},
            "exponential": {"scale": 1.0},
            "laplace": {"loc": 0.0, "scale": 1.0},
            "cauchy": {"loc": 0.0, "scale": 1.0},
            "poisson": {"lam": 5.0},
            "zipf": {"a": 2.0},
            "ones": {"value": 1.0},
            "zeros": {},
            "identity": {},
            "orthogonal": {},
            "sparse": {"density": 0.1}
        }
        return params.get(distribution, {})


class BatchGenerator:
    """Generate batch test configurations and data files."""

    def __init__(self, args):
        """
        Initialize batch generator.

        Args:
            args: Parsed CLI arguments
        """
        self.args = args
        self.output_dir = Path(args.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get operator specification
        self.op_spec = get_operator_spec(args.operator)
        logger.info(f"Operator: {self.op_spec.display_name}")

        # Create batch directory with microsecond timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        microseconds = datetime.now().microsecond
        self.batch_dir = self.output_dir / f"{args.operator}.batch.{timestamp}_{microseconds:06d}"
        self.batch_dir.mkdir(parents=True, exist_ok=True)

        # Create data subdirectory
        self.data_dir = self.batch_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize generators
        self.shape_gen = ShapeVariationGenerator(
            base_shape=args.shape_base,
            variation_mode=args.shape_variation,
            variation_range=args.shape_range,
            seed=args.seed
        )

        self.dist_selector = DistributionSelector(
            mixed_mode=args.mixed_distributions,
            specific_dists=args.distributions,
            seed=args.seed
        )

        # Initialize RandomInputGenerator
        self.input_gen = RandomInputGenerator(
            output_dir=str(self.data_dir),
            default_format=args.format,
            seed=args.seed
        )

        # Initialize JSON generator
        self.json_gen = InfiniMetricsJSONGenerator(
            operator=args.operator,
            device=args.device,
            data_base_dir="./data",  # Relative path for portability
            warmup_iterations=args.warmup,
            measured_iterations=args.measured,
            tolerance_atol=1e-3,
            tolerance_rtol=1e-3
        )

        logger.info(f"Batch directory: {self.batch_dir}")
        logger.info(f"Shape variation: {args.shape_variation} (range: {args.shape_range})")
        logger.info(f"Mixed distributions: {args.mixed_distributions}")

    def generate_single_config(self, index: int) -> Dict[str, Any]:
        """
        Generate a single test case configuration.

        Args:
            index: Test case index (0-based)

        Returns:
            Complete test configuration dictionary
        """
        # Generate shapes using operator spec
        varied_shape = self.shape_gen.generate_shape(index)
        input_shapes = self.op_spec.get_input_shapes(varied_shape, index)
        output_shape = self.op_spec.get_output_shape(input_shapes)
        input_names = self.op_spec.get_input_names()

        # Generate input configs with file paths
        inputs = []
        for i, (shape, name) in enumerate(zip(input_shapes, input_names)):
            distribution = self.dist_selector.get_distribution(i, name)
            dist_params = self.dist_selector.get_distribution_params(distribution)

            # Generate data and get config
            input_config = self.input_gen.generate_input_config(
                base_name=name,
                shape=shape,
                dtype=self.args.dtype,
                distribution=distribution,
                format=self.args.format,
                **dist_params
            )

            # Convert to relative path
            file_path = Path(input_config["file_path"])
            input_config["file_path"] = f"./data/{file_path.name}"

            inputs.append(input_config)

        # Generate JSON config using JSON generator
        config = self.json_gen.generate_batch_config(
            index=index,
            operator=self.args.operator,
            inputs=inputs,
            output_shape=output_shape
        )

        return config

    def generate_batch(self) -> List[str]:
        """
        Generate all test cases in the batch.

        Returns:
            List of generated config file paths
        """
        config_files = []
        total = self.args.count

        logger.info(f"Generating {total} test cases...")

        for i in range(total):
            # Generate config
            config = self.generate_single_config(i)

            # Write to JSON file
            filename = f"format_input_{self.args.operator}_batch.{i:04d}.json"
            config_path = self.batch_dir / filename
            JSONFileWriter.write_config(config, str(config_path))
            config_files.append(str(config_path))

            # Progress indicator
            if (i + 1) % max(1, total // 10) == 0 or i == 0:
                progress = (i + 1) / total * 100
                logger.info(f"Progress: {i+1}/{total} ({progress:.1f}%)")

        logger.info(f"✅ Generated {len(config_files)} test configurations")
        logger.info(f"📁 Output directory: {self.batch_dir}")
        logger.info(f"📊 Data files: {len(list(self.data_dir.glob('*')))}")

        return config_files


def main():
    """Main entry point for batch generation."""
    args = parse_arguments()

    # Log parameters
    logger.info("=" * 60)
    logger.info("Batch Input Generator for InfiniMetrics")
    logger.info("=" * 60)
    logger.info(f"Operator: {args.operator}")
    logger.info(f"Count: {args.count}")
    logger.info(f"Base shape: {args.shape_base}")
    logger.info(f"Shape variation: {args.shape_variation}")
    logger.info(f"Shape range: {args.shape_range}")
    logger.info(f"Data type: {args.dtype}")
    logger.info(f"Format: {args.format}")
    logger.info(f"Mixed distributions: {args.mixed_distributions}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 60)

    # Validate inputs
    if args.shape_variation == "benchmark":
        max_count = BenchmarkShapeGenerator.get_max_count(ndim=len(args.shape_base))
        if args.count > max_count:
            logger.error(f"Error: --count for benchmark mode must be <= {max_count}")
            logger.error(f"Use --list-benchmark-shapes to see all available shapes")
            return 1
    elif args.shape_variation != "none" and args.shape_range == 0:
        logger.error("Error: --shape-range must be > 0 when using --shape-variation")
        return 1

    if args.shape_variation == "progressive" and args.count > 1:
        logger.warning("Progressive mode: shapes will grow linearly with test index")
        logger.warning(f"Final shape will be: base + {args.shape_range * (args.count - 1)}")

    # Generate batch
    try:
        generator = BatchGenerator(args)
        config_files = generator.generate_batch()

        # Summary
        logger.info("=" * 60)
        logger.info("✅ Batch Generation Complete")
        logger.info("=" * 60)
        logger.info(f"Config files: {len(config_files)}")
        logger.info(f"Data directory: {generator.batch_dir}")

        # Print usage instructions
        logger.info("\n📖 Usage:")
        logger.info(f"cd {generator.batch_dir}")
        logger.info(f"python /path/to/InfiniMetrics/main.py format_input_{args.operator}_batch.0000.json")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Error during batch generation: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
