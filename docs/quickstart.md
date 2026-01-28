# Quick Start Guide

This guide will help you get up and running with InfiniMetrics in minutes.

## Prerequisites

Before you begin, ensure you have:
- **Python 3.8+** installed
- **Git** installed (for cloning the repository)
- **GPU** (optional, for hardware tests)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/InfiniTensor/InfiniMetrics.git
cd InfiniMetrics
```

### 2. Initialize Submodules

```bash
git submodule update --init --recursive
```

### 3. Install Dependencies

```bash
# Core dependencies
pip install numpy torch pandas

# Optional: For vLLM support
pip install vllm
```

### 4. Build Hardware Benchmarks (Optional)

If you plan to run hardware tests:

```bash
cd infinimetrics/hardware/cuda-memory-benchmark
bash build.sh
```

## Run Your First Test

### Hardware Test

Run a comprehensive hardware benchmark with a single command:

```bash
python main.py format_input_comprehensive_hardware.json
```

This will test:
- Memory bandwidth (H2D, D2H, D2D, bidirectional)
- STREAM benchmark
- GPU cache performance

### Inference Test

Run an inference benchmark:

```bash
cd infinimetrics/inference
python infer_main.py --config config.json
```

## Understanding the Output

### Test Results Location

Results are saved in:
```
./summary_output/
└── dispatcher_summary_YYYYMMDD_HHMMSS.json    # Overall summary
./output/
└── <test_case_name>/
    ├── seperated_test_result.json              # Seperated test result
    └── metrics.csv                             # Timeseries metrics
```

### Example: Hardware Test Output

```json
{
  "total_tests": 1,
  "successful_tests": 1,
  "failed_tests": 0,
  "results": [{
    "run_id": "test_001",
    "testcase": "hardware.cudaUnified.Comprehensive",
    "result_code": 0,
    "result_file": "./output/hardware.cudaUnified.Comprehensive/metrics.json"
  }]
}
```

## Next Steps

- **Configure Tests**: See [Configuration Guide](./configuration.md) for customization
