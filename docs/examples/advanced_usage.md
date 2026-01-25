# Advanced Usage Examples

This document covers advanced usage patterns and configurations for InfiniMetrics.

## Example 1: Running Multiple Tests in Batch

Execute multiple test configurations in sequence.

### Directory Setup

```
test_configs/
├── hardware_memory.json
├── hardware_stream.json
├── inference_infinilm.json
└── inference_vllm.json
```

### Running All Tests

```bash
python main.py ./test_configs/
```

### Output

Each test creates its own output directory:
```
./output/
├── hardware.cudaUnified.MemoryBandwidth/
├── hardware.cudaUnified.STREAM/
├── infer.infinilm.direct/
└── infer.vllm.default/
```

## Example 2: Custom Output Directory

Specify a custom output location for better organization.

### Configuration

```json
{
    "run_id": "test_001",
    "testcase": "hardware.cudaUnified.Comprehensive",
    "config": {
        "output_dir": "./my_results/hardware_tests"
    },
    "metrics": [...]
}
```

### Command Line Alternative

```bash
python main.py input.json --output ./my_results
```

## Example 3: Verbose Output

Enable detailed logging for debugging and analysis.

### Command

```bash
python main.py input.json --verbose
```

### Benefits
- Detailed execution information
- Metric collection progress
- Error stack traces
- Performance timing breakdown

## Example 4: Combining Hardware and Inference Tests

Create a comprehensive evaluation pipeline.

### Step 1: Hardware Tests

```bash
python main.py hardware_config.json
```

### Step 2: Operator Tests

```bash
python main.py operator_config.json
```

### Step 3: Inference Tests

```bash
python main.py inference_config.json
```

### Step 4: Aggregate Results

```bash
# All results in ./summary_output/
cat ./summary_output/dispatcher_summary_*.json
```

## Example 5: Testing Different Accelerator Types

Configure tests for different hardware vendors.

### NVIDIA

```json
{
    "config": {
        "device": "nvidia"
    }
}
```

### AMD

```json
{
    "config": {
        "device": "amd"
    }
}
```

### Huawei

```json
{
    "config": {
        "device": "huawei"
    }
}
```

## Example 6: Custom Metrics Collection

Define custom metrics for specific needs.

### Configuration

```json
{
    "run_id": "custom_test_001",
    "testcase": "hardware.cudaUnified.Custom",
    "config": {
        "custom_param": "value",
        "output_dir": "./output"
    },
    "metrics": [
        {"name": "custom.metric1"},
        {"name": "custom.metric2"},
        {"name": "standard.metric"}
    ]
}
```

## Example 7: Scripted Test Execution

Create a shell script to automate test execution.

### run_tests.sh

```bash
#!/bin/bash

# Set output directory
OUTPUT_DIR="./results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Run tests
python main.py hardware_config.json --output $OUTPUT_DIR
python main.py inference_config.json --output $OUTPUT_DIR

# Generate summary
echo "Test complete. Results in $OUTPUT_DIR"
```

### Running

```bash
chmod +x run_tests.sh
./run_tests.sh
```

## Example 8: Programmatic Execution

Use InfiniMetrics programmatically in Python.

### script.py

```python
from infinimetrics.dispatcher import Dispatcher

# Create dispatcher
dispatcher = Dispatcher()

# Load configuration
config = {
    "run_id": "prog_test_001",
    "testcase": "hardware.cudaUnified.Comprehensive",
    "config": {...},
    "metrics": [...]
}

# Run test
result = dispatcher.run(config)

# Access results
print(f"Test result: {result}")
```

## Example 9: Result Analysis

Parse and analyze test results programmatically.

### analyze_results.py

```python
import json
import glob

# Find all summary files
summaries = glob.glob("./summary_output/dispatcher_summary_*.json")

for summary_file in summaries:
    with open(summary_file, 'r') as f:
        data = json.load(f)

    print(f"Total tests: {data['total_tests']}")
    print(f"Successful: {data['successful_tests']}")
    print(f"Failed: {data['failed_tests']}")

    # Process each result
    for result in data['results']:
        if result['result_code'] == 0:
            with open(result['result_file'], 'r') as f:
                metrics = json.load(f)
                print(f"Metrics: {metrics}")
```

## Example 10: Continuous Benchmarking

Set up periodic benchmarking for performance regression testing.

### cron_job.sh

```bash
#!/bin/bash

# Run daily benchmarking
python main.py comprehensive_config.json \
    --output ./benchmark_results/$(date +%Y%m%d)

# Compare with baseline
python compare_results.py \
    --current ./benchmark_results/$(date +%Y%m%d) \
    --baseline ./baseline_results
```

### Crontab Entry

```
0 2 * * * /path/to/cron_job.sh
```

## Best Practices

### Organization
- Use separate directories for different test categories
- Include timestamps in output directories
- Keep a copy of input configs with results

### Reproducibility
- Version control your configuration files
- Document hardware and software versions
- Use fixed random seeds for stochastic tests

### Performance
- Run tests in parallel when possible
- Use batch execution for multiple tests
- Monitor system resources during execution

### Debugging
- Always use `--verbose` for new configurations
- Check log files in output directories
- Verify hardware status before running tests

## Next Steps

- See [Configuration Guide](../configuration.md) for parameter details
- Refer to [Development Guide](../development.md) for extending functionality
