# InfiniMetrics

<div align="center">

**An InfiniTensor-Featured Comprehensive Accelerator Evaluation Framework**

</div>

---

## 🎯 Overview

**InfiniMetrics** is a unified, modular testing framework designed for comprehensive performance evaluation of accelerator hardware and software stacks. It provides standardized interfaces for benchmarking across multiple layers:

- **Hardware-Level**: GPU memory bandwidth, cache performance, compute capabilities
- **Operator-Level**: Individual operation performance (FLOPS, latency)
- **Inference-Level**: End-to-end model inference throughput and latency
- **Communication-Level**: NCCL collective operations and inter-GPU communication

### Key Features

✨ **Unified Adapter Interface** - Consistent API across all test types and frameworks
🔧 **Extensible Architecture** - Easy to add new test types, frameworks, and metrics
📊 **Comprehensive Metrics** - Scalar values, time-series data, custom measurements
🎛️ **Framework Agnostic** - Support for InfiniLM, vLLM, InfiniCore, and more
🚀 **Production Ready** - Robust error handling, logging, and result aggregation

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Project Architecture](#project-architecture)
- [Supported Test Types](#supported-test-types)
- [Configuration Guide](#configuration-guide)
- [Output and Results](#output-and-results)
- [Examples](#examples)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/InfiniTensor/InfiniMetrics.git
cd InfiniMetrics
git submodule update --init --recursive
```

### 2. Run Hardware Benchmark

```bash
# Run comprehensive hardware tests
python main.py format_input_comprehensive_hardware.json
```

### 3. Run Inference Evaluation

```bash
# Run InfiniLM inference benchmark
python infinimetrics/inference/infer_main.py --config <path_to_config>
```

---

## 💾 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **Compiler**: GCC 11.3
- **CMake**: 3.20+

### Dependencies

```bash
# Core Python dependencies (install as needed)
pip install numpy torch  # For InfiniLM adapter
pip install vllm        # For vLLM adapter
pip install pandas       # For data processing
```

### Build Hardware Benchmarks (Optional)

If using hardware testing modules:

```bash
cd infinimetrics/hardware/cuda-memory-benchmark
bash build.sh
```

---

## 🏗️ Project Architecture

```
InfiniMetrics/
├── main.py                          # Main entry point
├── infinimetrics/
│   ├── adapter.py                  # Base adapter interface
│   ├── dispatcher.py               # Test orchestration
│   ├── executor.py                 # Universal test executor
│   ├── input.py                    # Test input data classes
│   │
│   ├── common/                     # Shared utilities
│   │   ├── constants.py            # Test types, metrics, enums
│   │   ├── metrics.py              # Metric definitions
│   │   └── testcase_utils.py      # Test case utilities
│   │
│   ├── hardware/                   # Hardware testing modules
│   │   └── cuda-memory-benchmark/  # CUDA memory benchmark suite
│   │       ├── include/            # C++ headers
│   │       ├── src/                # CUDA/C++ sources
│   │       ├── CMakeLists.txt     # Build configuration
│   │       ├── build.sh            # Build script
│   │       └── QUICKSTART.md       # Hardware test quick start guide
│   │
│   ├── operators/                  # Operator-level testing
│   │   ├── infinicore_adapter.py   # InfiniCore operations
│   │   └── flops_calculator.py    # FLOPS calculation
│   │
│   ├── inference/                  # Inference evaluation
│   │   ├── adapters/
│   │   │   ├── infinilm_adapter.py # InfiniLM adapter
│   │   │   └── vllm_adapter.py     # vLLM adapter
│   │   ├── infer_main.py           # Inference entry point
│   │   └── utils/                  # Inference utilities
│   │
│   ├── communication/              # Communication testing
│   │   └── nccl_adapter.py         # NCCL adapter
│   │
│   └── utils/                      # Utilities
│       └── input_loader.py        # Input file loader
│
├── submodules/
│   └── nccl-tests/                 # NCCL test suite (git submodule)
│
├── format_input_comprehensive_hardware.json  # Example configuration
└── summary_output/                # Test results directory
```

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                          InfiniMetrics                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────┐        │
│  │  Input   │───▶│ Dispatcher  │───▶│    Executor     │    │
│  │  Files   │    │             │    │                  │    │
│  └──────────┘    └─────────────┘    └────────┬─────────┘    │
│                                              │                   │
│                                    ┌─────────▼─────────┐    │
│                                    │                   │    │
│                       ┌──────────────┴──────────────┐    │
│                       │                              │    │
│  ┌────────────────────▼──────────────────────┐    │
│  │          Adapter Registry                │    │
│  │  ┌─────────┐  ┌──────────┐  ┌──────────┐  │    │
│  │  │ Hardware│  │ Operator │  │Inference │  │    │
│  │  │  Tests  │  │  Tests   │  │  Tests   │  │    │
│  │  └────┬────┘  └────┬─────┘  └────┬─────┘  │    │
│  └───────┼────────────┼─────────────┼──────────┘    │
│          │            │             │                   │
│  ┌───────▼───────────▼─────┬───────▼───────┐    │
│  │         Concrete Adapters           │    │
│  │  ┌─────────┐  ┌──────────┐  ┌──────┐  │    │
│  │  │ CUDA    │  │InfiniCore│  │vLLM  │  │    │
│  │  │ Memory  │  │  Adapter │  │Adapter│  │    │
│  │  │ Benchmark│  │          │  │     │  │    │
│  │  └─────────┘  └──────────┘  └──────┘  │    │
│  └────────────────────────────────────┘    │
│                                           │    │
│  ┌────────────────────────────────────┐    │
│  │         Metrics System             │    │
│  │  • Scalar metrics                  │    │
│  │  • Time-series metrics             │    │
│  │  • Custom metric definitions       │    │
│  └────────────────────────────────────┘    │
│                                           │    │
└───────────────────────────────────────────┘
```

---

## 🧪 Supported Test Types

### 1. Hardware Tests (`hardware.*`)

#### CUDA Memory Benchmarks

| Test Name | Description | Metrics |
|-----------|-------------|---------|
| `hardware.mem_sweep_h2d` | Host to Device sweep (64KB-1GB) | Bandwidth (GB/s), Time (ms) |
| `hardware.mem_sweep_d2h` | Device to Host sweep | Bandwidth (GB/s), Time (ms) |
| `hardware.mem_sweep_d2d` | Device to Device sweep | Bandwidth (GB/s), Time (ms) |
| `hardware.mem_bw_h2d` | H2D bandwidth (fixed size) | Bandwidth (GB/s) |
| `hardware.mem_bw_d2h` | D2H bandwidth (fixed size) | Bandwidth (GB/s) |
| `hardware.mem_bw_d2d` | D2D bandwidth (fixed size) | Bandwidth (GB/s) |

#### STREAM Benchmark

| Test Name | Description | Bytes/Element |
|-----------|-------------|---------------|
| `hardware.stream_copy` | Copy operation | 2 |
| `hardware.stream_scale` | Scale operation | 2 |
| `hardware.stream_add` | Add operation | 3 |
| `hardware.stream_triad` | Triad operation | 3 |

#### GPU Cache Tests

| Test Name | Description |
|-----------|-------------|
| `hardware.gpu_cache_l1` | L1 cache bandwidth |
| `hardware.gpu_cache_l2` | L2 cache bandwidth |

### 2. Operator Tests (`operator.*`)

| Test Name | Framework | Description |
|-----------|----------|-------------|
| `operator.infinicore.*` | InfiniCore | Individual operator performance, FLOPS calculation |

### 3. Inference Tests (`infer.*`)

| Test Name | Framework | Metrics |
|-----------|----------|---------|
| `infer.infinilm.direct` | InfiniLM | Throughput (tokens/s), Latency (ms), Memory Usage |
| `infer.infinilm.prefill` | InfiniLM | Prefill stage metrics |
| `infer.vllm.*` | vLLM | Various vLLM inference modes |

### 4. Communication Tests (`comm.*`)

| Test Name | Framework | Description |
|-----------|----------|-------------|
| `comm.nccltest.*` | NCCL | NCCL collective operations benchmarks |

---

## ⚙️ Configuration Guide

### Input File Format

Test specifications are provided in JSON format:

```json
{
    "run_id": "unique_run_identifier",
    "testcase": "hardware.cudaUnified.Comprehensive",
    "config": {
        "device": "nvidia",
        "array_size": 67108864,
        "buffer_size_mb": 256,
        "output_dir": "./output"
    },
    "metrics": [
        {"name": "hardware.mem_bw_h2d"},
        {"name": "hardware.mem_bw_d2h"},
        {"name": "hardware.stream_triad"}
    ]
}
```

### Configuration Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `run_id` | string | Unique test run identifier | Required |
| `testcase` | string | Test type identifier | Required |
| `config.device` | string | Accelerator type (nvidia/amd/huawei/cambricon) | nvidia |
| `config.array_size` | int | Array size for STREAM tests | 67108864 |
| `config.buffer_size_mb` | int | Buffer size in MB for memory tests | 256 |
| `config.output_dir` | string | Output directory path | ./output |

### Test Case Naming Convention

Format: `<category>.<framework>.<test_name>`

- **Categories**: `hardware`, `operator`, `infer`, `comm`
- **Frameworks**: `cudaUnified`, `infinicore`, `infinilm`, `vllm`, `nccltest`
- **Examples**:
  - `hardware.cudaUnified.Comprehensive`
  - `operator.infinicore.Conv2D`
  - `infer.infinilm.direct`
  - `comm.nccltest.AllReduce`

---

## 📊 Output and Results

### Output Directory Structure

```
./summary_output/
├── dispatcher_summary_YYYYMMDD_HHMMSS.json    # Overall test summary
└── ./output/
    ├── hardware.cudaUnified.Comprehensive/
    │   ├── metrics.json                            # Test metrics
    │   ├── trace.json                              # Execution trace (if available)
    │   └── log.txt                                 # Detailed log
    └── ...
```

### Summary Format

```json
{
  "total_tests": 3,
  "successful_tests": 2,
  "failed_tests": 1,
  "results": [
    {
      "run_id": "test_run_001",
      "testcase": "hardware.cudaUnified.Comprehensive",
      "result_code": 0,
      "result_file": "./output/hardware.cudaUnified.Comprehensive/metrics.json"
    }
  ],
  "timestamp": "2026-01-22T10:56:09.338373"
}
```

### Example Output

#### Hardware Benchmark Output

```
===================================================
Memory Copy Bandwidth Sweep Test
Direction: Host to Device
Memory Type: Pinned
===================================================

Size (MB)       Time (ms)  Bandwidth (GB/s)     CV (%)
------------------------------------------------------
     0.06            0.12             25.60        1.20
     0.12            0.24             25.80        0.90
     0.25            0.48             26.10        0.85
   256.00           98.45             26.20        1.10
   512.00          195.12             26.50        0.95
```

#### Inference Benchmark Output

```json
{
  "throughput_tokens_per_sec": 1250.5,
  "latency_ms": 8.5,
  "memory_usage_mb": 2048,
  "model_name": "infinilm-7b",
  "batch_size": 32
}
```

---

## 📚 Examples

### Example 1: Comprehensive Hardware Benchmark

```bash
python main.py format_input_comprehensive_hardware.json
```

**What it tests:**
- Memory bandwidth (H2D, D2H, D2D) across multiple buffer sizes
- STREAM benchmark (Copy, Scale, Add, Triad operations)
- GPU cache performance (L1, L2)

### Example 2: Multiple Test Configurations

```bash
# Run all JSON configs in a directory
python main.py ./test_configs/
```

### Example 3: Inference Evaluation

```bash
cd infinimetrics/inference
python infer_main.py --config config.json --model infinilm-7b
```

### Example 4: Verbose Output

```bash
python main.py input.json --verbose --output ./results
```

---

## 🔧 Development

### Adding a New Adapter

1. **Create adapter class** inheriting from `BaseAdapter`:

```python
from infinimetrics.adapter import BaseAdapter

class MyCustomAdapter(BaseAdapter):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your adapter

    def setup(self):
        # Prepare test environment
        pass

    def process(self, test_input):
        # Execute test and return metrics
        return {"my_metric": 42.0}

    def teardown(self):
        # Cleanup
        pass
```

2. **Register adapter in Dispatcher**:

```python
# In dispatcher.py
self.adapter_registry = {
    ("operator", "myframework"): MyCustomAdapter,
    # ... existing adapters ...
}
```

3. **Define test case and metrics**:

```json
{
    "run_id": "my_test",
    "testcase": "operator.myframework.MyTest",
    "config": {...},
    "metrics": [
        {"name": "my.metric"}
    ]
}
```

### Adding New Metrics

Define metrics in `infinimetrics/common/metrics.py`:

```python
class CustomMetric(Metric):
    def __init__(self, name: str, value: float, unit: str = ""):
        super().__init__(name, value, unit)
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: CUDA not found
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**Issue**: Import errors for InfiniLM/vLLM
```bash
pip install infinilm vllm
```

**Issue**: Hardware tests fail to compile
```bash
# Ensure CUDA toolkit is installed
nvcc --version
# Check CMake version
cmake --version
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions and support, please open an issue on GitHub or contact the InfiniTensor team.

---

<div align="center">

**Built with ❤️ by the InfiniTensor Team**

[Website](https://infinitensor.org) | [Documentation](https://docs.infinitensor.org) | [GitHub](https://github.com/InfiniTensor)

</div>
