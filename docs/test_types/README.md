# Test Types Overview

InfiniMetrics supports four main categories of tests, each targeting different layers of the accelerator and software stack.

## Test Categories

### 1. Hardware Tests (`hardware.*`)
Evaluate physical hardware capabilities including memory bandwidth, cache performance, and compute throughput.

- [Hardware Tests Documentation](./hardware_tests.md)

### 2. Operator Tests (`operator.*`)
Measure performance of individual operations and kernels, useful for optimizing specific computational patterns.

- [Operator Tests Documentation](./operator_tests.md)

### 3. Inference Tests (`infer.*`)
Assess end-to-end model inference performance, including throughput, latency, and memory usage.

- [Inference Tests Documentation](./inference_tests.md)

### 4. Communication Tests (`comm.*`)
Benchmark inter-GPU and inter-node communication performance using NCCL collective operations.

- [Communication Tests Documentation](./communication_tests.md)

## Test Naming Convention

All tests follow the format: `<category>.<framework>.<test_name>`

### Examples
- `hardware.cudaUnified.Comprehensive` - Comprehensive CUDA hardware test
- `operator.infinicore.Conv2D` - InfiniCore 2D convolution
- `infer.infinilm.direct` - Direct mode InfiniLM inference
- `comm.nccltest.AllReduce` - NCCL AllReduce benchmark

## Running Tests

See [Configuration Guide](../configuration.md) for details on how to configure and run tests.

Quick start:
```bash
python main.py format_input_comprehensive_hardware.json
```

## Adding New Tests

To add a new test type, see [Development Guide](../development.md).

## Test-Specific Documentation

Select a test category for detailed information:
