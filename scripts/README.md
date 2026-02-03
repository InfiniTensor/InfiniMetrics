# Testing Scripts

Collection of automated testing scripts for InfiniMetrics.

## 🚀 Quick Start

```bash
# Run all tests
./scripts/run_all_tests.sh

# Run specific test suite
./scripts/test_hardware.sh
./scripts/test_operator.sh
./scripts/test_inference.sh
./scripts/test_communication.sh

# Run specific test with input file
./scripts/test_hardware.sh format_input_hardware.json
./scripts/run_all_tests.sh format_input_comprehensive.json
```

## 📁 Structure

```
scripts/
├── run_all_tests.sh           # Main entry: run all test suites
├── test_hardware.sh           # Hardware testing (CUDA, memory bandwidth)
├── test_operator.sh           # Operator testing (InfiniCore operators)
├── test_inference.sh         # Inference testing (vLLM, InfiniLM)
└── common/                   # Shared utilities
    ├── check_deps.sh         # Dependency checking functions
    └── prepare_env.sh        # Environment preparation functions
```

## 📝 Script Organization

### Each Test Script Contains

1. **Dependencies Check** - Verify required tools/packages are installed
   - Example: `check_cuda`, `check_infinicore`, `check_vllm`

2. **Test Execution** - Run tests with proper error handling
   - Automatic output directory creation
   - Timestamp logging
   - JSON validation

### Common Functions (`common/`)

**`check_deps.sh`**: Check if dependencies are installed
- `check_cuda` - Check NVIDIA CUDA toolkit
- `check_nvml` - Check NVIDIA management library
- `check_infinicore` - Check InfiniCore package
- `check_vllm` - Check vLLM package
- `check_infinilm` - Check InfiniLM package

**`prepare_env.sh`**: Environment setup utilities
- `prepare_output_dir` - Create test output directories
- `log_test_start/end` - Logging functions
- `validate_json` - Validate JSON input files

## 🎯 Usage Examples

### Run All Tests
```bash
# Auto-discover and run all test files
./scripts/run_all_tests.sh
```

### Run Specific Test Suite
```bash
# Hardware tests
./scripts/test_hardware.sh

# Operator tests
./scripts/test_operator.sh

# Inference tests
./scripts/test_inference.sh

# Communication tests
./scripts/test_communication.sh
```

### Run Specific Test File
```bash
# Test with specific input
./scripts/test_hardware.py format_input_hardware.json

# All tests with same input
./scripts/run_all_tests.sh my_test.json
```

## 📊 Output

All test results are saved to:
```
output/
├── hardware/
├── operator/
├── inference/
└── communication/
```

## 🔧 Adding New Test Scripts

1. Create new test script in `scripts/`
2. Source common functions:
   ```bash
   source scripts/common/check_deps.sh
   source scripts/common/prepare_env.sh
   ```
3. Follow the template:
   ```bash
   check_xxx_deps() { ... }
   run_xxx_tests() { ... }
   ```
4. Add to `run_all_tests.sh` TESTS array
5. Update this README

## ⚠️ Requirements

- Python 3.10+
- Bash 4.0+
- Appropriate test dependencies (CUDA, InfiniCore, etc.)

## 🛠️ Troubleshooting

**Script not executable?**
```bash
chmod +x scripts/*.sh scripts/common/*.sh
```

**Permission denied?**
```bash
bash scripts/run_all_tests.sh
```

**Test file not found?**
```bash
# Scripts will search for matching patterns:
# Hardware: *hardware*.json, *mem*.json
# Operator: *operator*.json, *infinicore*.json
# Inference: *inference*.json, *vllm*.json, *infinilm*.json
# Communication: *nccl*.json, *communication*.json
```
