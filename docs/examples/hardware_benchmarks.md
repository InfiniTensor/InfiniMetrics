# Hardware Benchmark Examples

This document provides examples for running hardware benchmarks in InfiniMetrics.

## Example 1: Comprehensive Hardware Benchmark

Test all hardware capabilities including memory bandwidth, STREAM benchmark, and cache performance.

### Configuration

```json
{
    "run_id": "hw_comprehensive_001",
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
        {"name": "hardware.mem_bw_d2d"},
        {"name": "hardware.stream_copy"},
        {"name": "hardware.stream_scale"},
        {"name": "hardware.stream_add"},
        {"name": "hardware.stream_triad"},
        {"name": "hardware.gpu_cache_l1"},
        {"name": "hardware.gpu_cache_l2"}
    ]
}
```

### Running

```bash
python main.py format_input_comprehensive_hardware.json
```

### What It Tests
- Memory bandwidth (H2D, D2H, D2D) across multiple buffer sizes
- STREAM benchmark (Copy, Scale, Add, Triad operations)
- GPU cache performance (L1, L2)

## Example 2: Memory Bandwidth Only

Focus solely on memory transfer performance.

### Configuration

```json
{
    "run_id": "hw_memory_001",
    "testcase": "hardware.cudaUnified.MemoryBandwidth",
    "config": {
        "device": "nvidia",
        "buffer_size_mb": 512,
        "output_dir": "./output"
    },
    "metrics": [
        {"name": "hardware.mem_bw_h2d"},
        {"name": "hardware.mem_bw_d2h"},
        {"name": "hardware.mem_bw_d2d"}
    ]
}
```

### Running

```bash
python main.py memory_only_config.json
```

## Example 3: STREAM Benchmark

Test sustainable memory bandwidth for computation patterns.

### Configuration

```json
{
    "run_id": "hw_stream_001",
    "testcase": "hardware.cudaUnified.STREAM",
    "config": {
        "device": "nvidia",
        "array_size": 67108864,
        "output_dir": "./output"
    },
    "metrics": [
        {"name": "hardware.stream_copy"},
        {"name": "hardware.stream_scale"},
        {"name": "hardware.stream_add"},
        {"name": "hardware.stream_triad"}
    ]
}
```

### Running

```bash
python main.py stream_config.json
```

## Example 4: Cache Performance

Evaluate L1 and L2 cache bandwidth.

### Configuration

```json
{
    "run_id": "hw_cache_001",
    "testcase": "hardware.cudaUnified.Cache",
    "config": {
        "device": "nvidia",
        "output_dir": "./output"
    },
    "metrics": [
        {"name": "hardware.gpu_cache_l1"},
        {"name": "hardware.gpu_cache_l2"}
    ]
}
```

### Running

```bash
python main.py cache_config.json
```

## Example 5: Multiple Configurations

Run multiple hardware tests in batch.

### Directory Structure

```
test_configs/
├── memory.json
├── stream.json
└── cache.json
```

### Running All

```bash
python main.py ./test_configs/
```

## Understanding Results

### Memory Bandwidth

Results in GB/s:
- H2D/D2H: 20-30 GB/s (PCIe limited)
- D2D: 300-700 GB/s (HBM bandwidth)

### STREAM Results

Measures sustainable bandwidth for computation:
- Copy: 2 bytes/element
- Scale: 2 bytes/element
- Add: 3 bytes/element
- Triad: 3 bytes/element

Typical: 300-700 GB/s for modern GPUs

### Cache Results

- L1: 1-2 TB/s
- L2: 500-800 GB/s

## Output Files

Results are saved in:
```
./output/
└── hardware.cudaUnified.Comprehensive/
    ├── metrics.json
    ├── trace.json
    └── log.txt
```

## Troubleshooting

### Build Errors

If hardware tests fail to compile:

```bash
cd infinimetrics/hardware/cuda-memory-benchmark
bash build.sh
```

### Out of Memory

Reduce buffer size:

```json
{
    "config": {
        "buffer_size_mb": 128
    }
}
```

## Next Steps

- See [Hardware Tests Documentation](../test_types/hardware_tests.md) for details
- Explore [Advanced Usage](./advanced_usage.md) for more examples
