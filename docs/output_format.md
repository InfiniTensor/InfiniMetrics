# Output Format

This document describes the output format and directory structure of test results.

## Output Directory Structure

```
./summary_output/
└── dispatcher_summary_YYYYMMDD_HHMMSS.json    # Overall summary
./output/
└── <test_case_name>/
    ├── seperated_test_result.json              # Seperated test result
    └── metrics.csv                             # Timeseries metrics
```

## Summary Format

The dispatcher summary file contains overall test execution results:

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

### Summary Fields

| Field | Type | Description |
|-------|------|-------------|
| `total_tests` | int | Total number of tests executed |
| `successful_tests` | int | Number of successful tests |
| `failed_tests` | int | Number of failed tests |
| `results` | array | Array of test result objects |
| `timestamp` | string | Execution timestamp in ISO format |

## Test-Specific Output

### Hardware Benchmark Output

#### Metrics JSON

```json
{
   {
      "name": "hardware.mem_sweep_d2d",
      "type": "timeseries",
      "raw_data_url": "./mem_sweep_d2d_hardware.cudaUnified.Comprehensive.fadfsdf_20260126_110701.csv",
      "unit": "GB/s"
    },
    {
      "name": "hardware.stream_copy",
      "value": 435.88,
      "type": "scalar",
      "unit": "GB/s"
    },
}
```

#### Console Output

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

### Inference Benchmark Output

#### Metrics JSON

```json
{
  "throughput_tokens_per_sec": 1250.5,
  "latency_ms": 8.5,
  "memory_usage_mb": 2048,
  "model_name": "infinilm-7b",
  "batch_size": 32
}
```

## Result Code Reference

| Code | Meaning |
|------|---------|
| 0 | Success |
| Non-zero | Failure (error-specific) |
