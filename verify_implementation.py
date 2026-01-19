#!/usr/bin/env python3
"""
Verification script for file-based input functionality
Checks implementation without running actual tests
"""

import sys
from pathlib import Path

print("=" * 70)
print("File-Based Input Implementation Verification")
print("=" * 70)

errors = []
warnings = []

# Check 1: InfiniMetrics constants.py
print("\n[1] Checking InfiniMetrics/infinimetrics/common/constants.py...")
constants_file = Path("infinimetrics/common/constants.py")
if constants_file.exists():
    content = constants_file.read_text()
    if 'FILE_PATH = "file_path"' in content:
        print("  ✓ FILE_PATH constant defined")
    else:
        errors.append("FILE_PATH constant not found in constants.py")
else:
    errors.append(f"File not found: {constants_file}")

# Check 2: InfiniCore load_utils.py
print("\n[2] Checking InfiniCore/test/infinicore/framework/utils/load_utils.py...")
load_utils_file = Path("../InfiniCore/test/infinicore/framework/utils/load_utils.py")
if load_utils_file.exists():
    content = load_utils_file.read_text()
    if "'file_path' in spec_dict" in content:
        print("  ✓ file_path field detection implemented")
    else:
        errors.append("file_path detection not found in load_utils.py")

    if "TensorInitializer.FROM_FILE" in content:
        print("  ✓ FROM_FILE init_mode implemented")
    else:
        errors.append("FROM_FILE init_mode not found in load_utils.py")

    if "init_mode" in content and "TensorInitializer.RANDOM" in content:
        print("  ✓ init_mode logic implemented")
    else:
        errors.append("init_mode logic not found in load_utils.py")
else:
    errors.append(f"File not found: {load_utils_file}")

# Check 3: InfiniMetrics infinicore_adapter.py
print("\n[3] Checking infinimetrics/operators/infinicore_adapter.py...")
adapter_file = Path("infinimetrics/operators/infinicore_adapter.py")
if adapter_file.exists():
    content = adapter_file.read_text()
    if 'TensorSpec.FILE_PATH in inp' in content or '"file_path"' in content:
        print("  ✓ file_path field handling implemented")
    else:
        errors.append("file_path handling not found in adapter.py")

    if "data_base_dir" in content:
        print("  ✓ data_base_dir for relative path support implemented")
    else:
        errors.append("data_base_dir not found in adapter.py")

    if "Path(data_base_dir)" in content:
        print("  ✓ Relative path resolution implemented")
    else:
        errors.append("Relative path resolution not found in adapter.py")
else:
    errors.append(f"File not found: {adapter_file}")

# Check 4: RandomInputGenerator
print("\n[4] Checking infinimetrics/utils/input_generator.py...")
generator_file = Path("infinimetrics/utils/input_generator.py")
if generator_file.exists():
    content = generator_file.read_text()
    if "class RandomInputGenerator" in content:
        print("  ✓ RandomInputGenerator class defined")
    else:
        errors.append("RandomInputGenerator class not found")

    if "def generate_input_config" in content:
        print("  ✓ generate_input_config method defined")
    else:
        errors.append("generate_input_config method not found")

    if "def _generate_random_data" in content:
        print("  ✓ _generate_random_data method defined")
    else:
        errors.append("_generate_random_data method not found")

    if "def _generate_unique_path" in content:
        print("  ✓ _generate_unique_path method defined")
    else:
        errors.append("_generate_unique_path method not found")

    if "def _save_data" in content:
        print("  ✓ _save_data method defined")
    else:
        errors.append("_save_data method not found")

    if "datetime.now().strftime" in content:
        print("  ✓ Timestamp-based filename generation implemented")
    else:
        warnings.append("Timestamp-based filename not confirmed")

    # Check distribution support
    distributions = ["uniform", "normal", "randint"]
    for dist in distributions:
        if f'"{dist}"' in content or f"'{dist}'" in content:
            print(f"  ✓ {dist} distribution support")
        else:
            warnings.append(f"{dist} distribution support not confirmed")
else:
    errors.append(f"File not found: {generator_file}")

# Check 5: CLI tool
print("\n[5] Checking infinimetrics/utils/input_generator_cli.py...")
cli_file = Path("infinimetrics/utils/input_generator_cli.py")
if cli_file.exists():
    content = cli_file.read_text()
    if "def main" in content:
        print("  ✓ CLI main function defined")
    else:
        errors.append("CLI main function not found")

    if "argparse" in content:
        print("  ✓ ArgumentParser for CLI arguments")
    else:
        errors.append("ArgumentParser not found in CLI")

    if "--config" in content:
        print("  ✓ --config argument defined")
    else:
        errors.append("--config argument not found")

    if "--output-dir" in content:
        print("  ✓ --output-dir argument defined")
    else:
        errors.append("--output-dir argument not found")

    if "--seed" in content:
        print("  ✓ --seed argument defined")
    else:
        errors.append("--seed argument not found")
else:
    errors.append(f"File not found: {cli_file}")

# Check 6: Test configuration
print("\n[6] Checking test configuration files...")
test_config = Path("test_file_based_input.json")
if test_config.exists():
    content = test_config.read_text()
    if '"data_base_dir"' in content:
        print("  ✓ data_base_dir in test config")
    else:
        warnings.append("data_base_dir not in test config")

    if '"_random"' in content:
        print("  ✓ _random config for generation")
    else:
        warnings.append("_random config not in test config")
else:
    warnings.append(f"Test config not found: {test_config}")

# Summary
print("\n" + "=" * 70)
print("Verification Summary")
print("=" * 70)

if errors:
    print(f"\n✗ ERRORS ({len(errors)}):")
    for error in errors:
        print(f"  - {error}")

if warnings:
    print(f"\n⚠ WARNINGS ({len(warnings)}):")
    for warning in warnings:
        print(f"  - {warning}")

if not errors and not warnings:
    print("\n✓ All checks passed! Implementation looks complete.")
    print("\nNext steps:")
    print("  1. Install dependencies: pip install numpy torch")
    print("  2. Run test: python test_generator.py")
    print("  3. Generate inputs: python infinimetrics/utils/input_generator_cli.py --config test_file_based_input.json")
    print("  4. Run test with generated files: python main.py test_file_based_input.json")
elif not errors:
    print("\n✓ Implementation complete with minor warnings.")
    print("\nYou can proceed with testing once dependencies are installed.")
else:
    print("\n✗ Implementation has errors that need to be fixed.")
    sys.exit(1)

print("\n" + "=" * 70)
