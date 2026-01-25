# Hardware Tests

Hardware tests evaluate the physical capabilities of accelerator hardware including memory bandwidth, cache performance, and compute throughput.

## CUDA Memory Benchmarks

### Host to Device (H2D) Tests

| Test Name | Description | Metrics |
|-----------|-------------|---------|
| `hardware.mem_sweep_h2d` | Host to Device sweep (64KB-1GB) | Bandwidth (GB/s), Time (ms) |
| `hardware.mem_bw_h2d` | H2D bandwidth (fixed size) | Bandwidth (GB/s) |

### Device to Host (D2H) Tests

| Test Name | Description | Metrics |
|-----------|-------------|---------|
| `hardware.mem_sweep_d2h` | Device to Host sweep | Bandwidth (GB/s), Time (ms) |
| `hardware.mem_bw_d2h` | D2H bandwidth (fixed size) | Bandwidth (GB/s) |

### Device to Device (D2D) Tests

| Test Name | Description | Metrics |
|-----------|-------------|---------|
| `hardware.mem_sweep_d2d` | Device to Device sweep | Bandwidth (GB/s), Time (ms) |
| `hardware.mem_bw_d2d` | D2D bandwidth (fixed size) | Bandwidth (GB/s) |

## STREAM Benchmark

The STREAM benchmark measures sustainable memory bandwidth and simple computation rate for simple vector kernels.

| Test Name | Description | Bytes/Element |
|-----------|-------------|---------------|
| `hardware.stream_copy` | Copy operation | 2 |
| `hardware.stream_scale` | Scale operation | 2 |
| `hardware.stream_add` | Add operation | 3 |
| `hardware.stream_triad` | Triad operation | 3 |

### STREAM Operations

- **Copy**: `a[i] = b[i]`
- **Scale**: `a[i] = q * b[i]`
- **Add**: `a[i] = b[i] + c[i]`
- **Triad**: `a[i] = b[i] + q * c[i]`

## GPU Cache Tests

| Test Name | Description |
|-----------|-------------|
| `hardware.gpu_cache_l1` | L1 cache bandwidth |
| `hardware.gpu_cache_l2` | L2 cache bandwidth |

## Running Hardware Tests

### Configuration Example

```json
{
    "run_id": "hw_test_001",
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

### Command

```bash
python main.py format_input_comprehensive_hardware.json
```

## Understanding Results

### Memory Bandwidth

Results are reported in GB/s (gigabytes per second). Higher values indicate better performance.

Typical values:
- H2D: 20-30 GB/s (PCIe 3.0/4.0 limited)
- D2H: 20-30 GB/s (PCIe 3.0/4.0 limited)
- D2D: 300-700 GB/s (HBM2/HBM2e bandwidth)

### STREAM Results

STREAM metrics measure sustainable memory bandwidth for computation patterns.

Typical values: 300-700 GB/s for modern GPUs

### Cache Results

Cache tests show bandwidth for L1 and L2 caches.

Typical values:
- L1: 1-2 TB/s
- L2: 500-800 GB/s

## Building Hardware Tests

If you need to build the CUDA memory benchmark suite:

```bash
cd infinimetrics/hardware/cuda-memory-benchmark
bash build.sh
```

For more details, see [Installation Guide](../installation.md).

## Troubleshooting

See [Troubleshooting](../troubleshooting.md) for common hardware test issues.
