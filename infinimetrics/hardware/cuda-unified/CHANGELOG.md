# Changelog

## [Unreleased]

### Removed
- **Device Communication Tests**: Removed `device_communication_test.h` and related functionality
  - Removed PeerAccessTest class
  - Removed DeviceCommunicationSuite class
  - Removed `--comm` command-line option
  - Updated all documentation to reflect removal

### Reason for Removal
The device communication tests (originally based on bandwidthTest) are not needed for the current testing requirements. The suite now focuses on:

1. **Memory Bandwidth Tests** - Host↔Device and Device↔Device transfers
2. **STREAM Benchmark** - Standard memory bandwidth benchmark
3. **Cache Performance Tests** - L1 and L2 cache analysis

### Files Modified
- `src/main.cu` - Removed communication test includes and execution logic
- `README.md` - Updated feature list and usage examples
- `QUICKSTART.md` - Removed multi-GPU communication section
- Deleted: `include/device_communication_test.h`

### Impact
- Reduced binary size
- Simplified command-line interface
- Faster compilation
- Focused testing scope

### Migration Guide
If you were using the `--comm` option:
```bash
# Old (no longer available)
./build/cuda_perf_suite --comm

# Use alternative tools for multi-GPU testing:
# - NVIDIA bandwidthTest sample
# - NCCL benchmarks
# - Custom P2P testing tools
```
