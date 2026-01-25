# Inference Evaluation Examples

This document provides examples for running inference benchmarks with InfiniLM and vLLM.

## Example 1: InfiniLM Direct Mode

Run InfiniLM inference in direct mode for baseline performance.

### Configuration

```json
{
    "run_id": "infer_infinilm_direct_001",
    "testcase": "infer.infinilm.direct",
    "config": {
        "model_path": "/path/to/infinilm-7b",
        "batch_size": 32,
        "prompt_length": 128,
        "max_length": 256,
        "output_dir": "./output"
    },
    "metrics": [
        {"name": "infer.throughput"},
        {"name": "infer.latency"},
        {"name": "infer.memory_usage"}
    ]
}
```

### Running

```bash
python main.py infinilm_direct_config.json
```

### Metrics Collected
- **Throughput**: tokens per second
- **Latency**: milliseconds per request
- **Memory Usage**: GPU memory in MB

## Example 2: InfiniLM Prefill

Test prefill stage performance separately.

### Configuration

```json
{
    "run_id": "infer_infinilm_prefill_001",
    "testcase": "infer.infinilm.prefill",
    "config": {
        "model_path": "/path/to/infinilm-7b",
        "batch_size": 16,
        "prompt_length": 512,
        "output_dir": "./output"
    },
    "metrics": [
        {"name": "infer.prefill_time"},
        {"name": "infer.prefill_throughput"}
    ]
}
```

### Running

```bash
python main.py infinilm_prefill_config.json
```

## Example 3: vLLM Inference

Run inference using vLLM framework.

### Configuration

```json
{
    "run_id": "infer_vllm_001",
    "testcase": "infer.vllm.default",
    "config": {
        "model": "facebook/opt-125m",
        "tensor_parallel_size": 1,
        "batch_size": 16,
        "max_length": 128,
        "output_dir": "./output"
    },
    "metrics": [
        {"name": "infer.throughput"},
        {"name": "infer.latency"}
    ]
}
```

### Running

```bash
python main.py vllm_config.json
```

## Example 4: Batch Size Comparison

Compare performance across different batch sizes.

### Configuration 1 (Small Batch)

```json
{
    "run_id": "infer_batch_small_001",
    "testcase": "infer.infinilm.direct",
    "config": {
        "model_path": "/path/to/infinilm-7b",
        "batch_size": 1,
        "prompt_length": 128,
        "max_length": 256,
        "output_dir": "./output"
    },
    "metrics": [
        {"name": "infer.throughput"},
        {"name": "infer.latency"}
    ]
}
```

### Configuration 2 (Large Batch)

```json
{
    "run_id": "infer_batch_large_001",
    "testcase": "infer.infinilm.direct",
    "config": {
        "model_path": "/path/to/infinilm-7b",
        "batch_size": 64,
        "prompt_length": 128,
        "max_length": 256,
        "output_dir": "./output"
    },
    "metrics": [
        {"name": "infer.throughput"},
        {"name": "infer.latency"}
    ]
}
```

### Running Both

```bash
python main.py batch_small_config.json
python main.py batch_large_config.json
```

## Example 5: Using Inference Module Directly

You can also use the inference module directly without going through main.py.

### Command

```bash
cd infinimetrics/inference
python infer_main.py --config config.json --model /path/to/model --batch-size 32
```

### Config File (config.json)

```json
{
    "model_path": "/path/to/infinilm-7b",
    "batch_size": 32,
    "prompt_length": 128,
    "max_length": 256,
    "output_file": "./inference_results.json"
}
```

## Understanding Results

### Typical Performance (7B Model)

| Batch Size | Throughput (tokens/s) | Latency (ms) |
|------------|----------------------|--------------|
| 1 | 50-100 | 20-50 |
| 8 | 200-500 | 50-150 |
| 16 | 400-800 | 80-200 |
| 32 | 500-1500 | 100-300 |
| 64 | 600-2000 | 150-400 |

*Values vary by hardware and model size*

### Interpreting Metrics

- **Throughput**: Higher is better for batch processing
- **Latency**: Lower is better for interactive applications
- **Memory Usage**: Should be within GPU memory limits

## Performance Tips

### Optimize for Throughput
- Increase batch size
- Use tensor parallelism for larger models
- Enable KV cache optimization

### Optimize for Latency
- Use smaller batch sizes
- Reduce prompt and max length
- Use quantization (FP16/BF16)

### Memory Optimization
- Reduce batch size
- Use smaller models
- Enable gradient checkpointing (for training)

## Troubleshooting

### Out of Memory

```json
{
    "config": {
        "batch_size": 8  // Reduce from 32
    }
}
```

### Slow Performance

1. Check GPU utilization: `nvidia-smi`
2. Verify model is loaded correctly
3. Ensure proper CUDA version
4. Try different batch sizes

### Import Errors

```bash
pip install infinilm vllm torch
```

## Next Steps

- See [Inference Tests Documentation](../test_types/inference_tests.md) for details
- Explore [Advanced Usage](./advanced_usage.md) for more examples
