# CUDA Performance Benchmark Suite

A comprehensive, modern CUDA performance testing suite written in C++17 with CMake build system.

## Features

- **Memory Bandwidth Tests**: Host-to-Device, Device-to-Host, Device-to-Device transfers
- **STREAM Benchmark**: Standard memory bandwidth benchmark (Copy, Scale, Add, Triad operations)
- **Cache Performance Tests**: L1 and L2 cache bandwidth analysis
- **Modern C++ Design**: RAII patterns, smart pointers, exception safety
- **CMake Build System**: Cross-platform, easy to configure
- **Comprehensive Metrics**: Statistical analysis with trimmed mean, coefficient of variation

## Requirements

- CUDA Toolkit 11.0 or higher
- CMake 3.18 or higher
- C++17 compatible compiler
- NVIDIA GPU with compute capability 8.0+ (configurable)

## Building

### Quick Start

```bash
cd benchmarks/hardware/cuda-memory-benchmark
./build.sh
```

### Manual Build

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Adjusting CUDA Architecture

Edit `CMakeLists.txt` to specify your GPU architecture:

```cmake
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90" CACHE STRING "CUDA architectures")
```

Common architectures:
- 80: A100, RTX 3090
- 86: RTX 3080, A30
- 89: RTX 4080, 4090
- 90: H100

## Usage

### Run All Tests

```bash
./build/cuda_perf_suite --all
```

### Run Specific Tests

```bash
# Memory bandwidth tests only
./build/cuda_perf_suite --memory

# STREAM benchmark only
./build/cuda_perf_suite --stream

# Cache performance tests only
./build/cuda_perf_suite --cache
```

### Command Line Options

```
Options:
  --all                    Run all tests (default)
  --memory                 Run memory bandwidth tests only
  --stream                 Run STREAM benchmark only
  --cache                  Run cache benchmarks only
  --device <id>            Specify CUDA device ID (default: 0)
  --iterations <n>         Number of measurement iterations (default: 10)
  --array-size <size>      Array size for STREAM test (default: 67108864)
  --quiet                  Reduce output verbosity
  --help                   Show help message
```

### Examples

```bash
# Run STREAM benchmark with 128M elements
./build/cuda_perf_suite --stream --array-size 134217728

# Run memory tests with 512MB buffer
./build/cuda_perf_suite --memory --buffer-size 512

# Run all tests with 20 iterations
./build/cuda_perf_suite --all --iterations 20

# Run tests on specific GPU
./build/cuda_perf_suite --all --device 1
```

## Test Descriptions

### 1. Memory Bandwidth Tests

Tests data transfer bandwidth between different memory spaces:

- **Host to Device (Pinned)**: Measures PCIe bandwidth using page-locked memory
- **Device to Host (Pinned)**: Measures PCIe read bandwidth
- **Device to Device**: Measures GPU internal memory bandwidth

**Output**: Transfer time and bandwidth for various buffer sizes (64KB to 1GB)

### 2. STREAM Benchmark

Standard STREAM benchmark with 4 operations:

- **Copy**: `a[i] = b[i]` - 2 bytes per element
- **Scale**: `a[i] = scalar * b[i]` - 2 bytes per element
- **Add**: `a[i] = b[i] + c[i]` - 3 bytes per element
- **Triad**: `a[i] = b[i] + scalar * c[i]` - 3 bytes per element

**Output**: Bandwidth in GB/s for each operation

### 3. Cache Performance Tests

Tests L1 and L2 cache bandwidth by varying working set sizes

**Output**: Bandwidth vs working set size, revealing cache hierarchy characteristics

## Understanding Results

### Bandwidth Metrics

- **Average Time**: Mean execution time
- **Trimmed Mean**: Average excluding min/max values (more robust)
- **Coefficient of Variation (CV)**: Relative standard deviation (lower is better)

## Project Structure

```
cuda-memory-benchmark/
├── include/               # Header files
│   ├── performance_test.h           # Base testing framework
│   ├── cuda_utils.h                 # CUDA utilities (RAII wrappers)
│   ├── memory_bandwidth_test.h      # Memory copy tests
│   ├── stream_benchmark.h           # STREAM benchmark
│   └── cache_benchmark.h            # Cache tests
├── src/
│   └── main.cu                      # Main program entry
├── CMakeLists.txt                   # CMake build configuration
├── build.sh                         # Build script
└── README.md                        # This file
```
