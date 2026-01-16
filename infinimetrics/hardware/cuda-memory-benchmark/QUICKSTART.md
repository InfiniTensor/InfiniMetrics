# Quick Start Guide

## 1. Build the Project

```bash
cd benchmarks/hardware/cuda-memory-benchmark
./build.sh
```

## 2. Run All Tests

```bash
./build/cuda_perf_suite --all
```

## 3. Run Individual Test Suites

### Memory Bandwidth Tests
```bash
./build/cuda_perf_suite --memory
```
Tests Host↔Device and Device↔Device transfer bandwidths with multiple buffer sizes.

### Bandwidth Tests
```bash
./build/cuda_perf_suite --bandwidth
```
Tests device-to-device, device-to-host, and host-to-device bandwidth with 32MB transfers.

### STREAM Benchmark
```bash
./build/cuda_perf_suite --stream
```
Standard STREAM benchmark measuring sustainable memory bandwidth.

### Cache Tests
```bash
./build/cuda_perf_suite --cache
```
Tests L1 and L2 cache performance with varying working set sizes.


## 4. Common Usage Patterns

### Quick Performance Check (Default Settings)
```bash
./build/cuda_perf_suite --all
```

### Quick Bandwidth Test (Fast)
```bash
./build/cuda_perf_suite --bandwidth
```

### Detailed STREAM Benchmark (More Iterations)
```bash
./build/cuda_perf_suite --stream --iterations 50
```

### Large Buffer Memory Test
```bash
./build/cuda_perf_suite --memory --buffer-size 1024
```

### Test Specific GPU
```bash
./build/cuda_perf_suite --all --device 1
```

### Quiet Mode (Less Output)
```bash
./build/cuda_perf_suite --all --quiet
```

## 5. Understanding Output

The tests report:
- **Time (ms)**: Average execution time
- **Bandwidth (GB/s)**: Data transfer rate
- **CV (%)**: Coefficient of Variation (consistency measure)

Lower CV = more consistent results.

## 6. Example Output

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║        CUDA Performance Benchmark Suite v1.0                  ║
║                                                               ║
║        Comprehensive GPU Memory & Cache Testing               ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

=== System Information ===
CUDA Devices: 1

Device 0: NVIDIA A100-SXM4-40GB
  Compute Capability: 8.0
  Total Global Memory: 39.25 GB
  L2 Cache Size: 40960 KB
  Multiprocessors: 108
  Max Threads per Block: 1024

...

Memory Copy Bandwidth Sweep Test
Direction: Host to Device
Memory Type: Pinned

Size (MB)       Time (ms)  Bandwidth (GB/s)     CV (%)
--------------------------------------------------------------
     0.06            1.234             25.60        2.30
     0.13            2.456             25.80        1.90
     ...
```

## 7. Next Steps

- Read [README.md](README.md) for detailed documentation
- Adjust test parameters for your specific use case
- Integrate into your performance testing workflow
