#!/usr/bin/env python3
"""
Test script for RandomInputGenerator
Validates basic functionality
"""

import sys
import os
import json
from pathlib import Path

# Add infinimetrics to path
sys.path.insert(0, str(Path(__file__).parent))

from infinimetrics.utils.input_generator import RandomInputGenerator, generate_random_inputs_from_config


def test_basic_generation():
    """Test basic input generation"""
    print("=" * 60)
    print("Test 1: Basic Input Generation")
    print("=" * 60)

    generator = RandomInputGenerator(output_dir="./test_generated_data", seed=42)

    # Generate a simple input
    config = generator.generate_input_config(
        base_name="test_input",
        shape=[4, 4],
        dtype="float32",
        distribution="uniform",
        low=-1.0,
        high=1.0
    )

    print(f"Generated config:")
    print(json.dumps(config, indent=2))

    # Verify file exists
    file_path = Path(config["file_path"])
    if file_path.exists():
        print(f"✓ File created: {file_path}")
        print(f"  File size: {file_path.stat().st_size} bytes")
    else:
        print(f"✗ File not created: {file_path}")
        return False

    print()
    return True


def test_batch_generation():
    """Test batch generation from config"""
    print("=" * 60)
    print("Test 2: Batch Generation")
    print("=" * 60)

    inputs_config = [
        {
            "name": "input_a",
            "shape": [2, 3],
            "dtype": "float16",
            "_random": {
                "distribution": "uniform",
                "params": {"low": -1.0, "high": 1.0}
            }
        },
        {
            "name": "input_b",
            "shape": [2, 3],
            "dtype": "float16",
            "_random": {
                "distribution": "normal",
                "params": {"mean": 0.0, "std": 0.5}
            }
        }
    ]

    generated = generate_random_inputs_from_config(
        inputs_config=inputs_config,
        output_dir="./test_generated_data",
        seed=42
    )

    print(f"Generated {len(generated)} inputs:")
    for inp in generated:
        print(f"  - {inp['name']}: {inp['file_path']}")
        file_path = Path(inp['file_path'])
        if file_path.exists():
            print(f"    ✓ File exists ({file_path.stat().st_size} bytes)")
        else:
            print(f"    ✗ File not found")
            return False

    print()
    return True


def test_different_distributions():
    """Test different distribution types"""
    print("=" * 60)
    print("Test 3: Different Distributions")
    print("=" * 60)

    generator = RandomInputGenerator(output_dir="./test_generated_data", seed=42)

    distributions = [
        ("uniform", {"low": 0.0, "high": 1.0}),
        ("normal", {"mean": 0.0, "std": 1.0}),
        ("randint", {"low": -10, "high": 10})
    ]

    for dist_name, params in distributions:
        try:
            config = generator.generate_input_config(
                base_name=f"test_{dist_name}",
                shape=[3, 3],
                dtype="float32",
                distribution=dist_name,
                **params
            )
            print(f"✓ {dist_name}: {config['file_path']}")
        except Exception as e:
            print(f"✗ {dist_name}: {e}")
            return False

    print()
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("RandomInputGenerator Test Suite")
    print("=" * 60 + "\n")

    tests = [
        ("Basic Generation", test_basic_generation),
        ("Batch Generation", test_batch_generation),
        ("Different Distributions", test_different_distributions)
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Test '{name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Print summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(result for _, result in results)
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed!"))
    print(f"Generated files are in: ./test_generated_data")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
