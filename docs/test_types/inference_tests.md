# Inference Tests

Inference tests evaluate end-to-end model inference performance, measuring throughput, latency, and resource utilization.

## Supported Frameworks

### InfiniLM Tests

| Test Name | Framework | Metrics |
|-----------|----------|---------|
| `infer.infinilm.direct` | InfiniLM | Throughput (tokens/s), Latency (ms), Memory Usage |
| `infer.infinilm.prefill` | InfiniLM | Prefill stage metrics |
| `infer.infinilm.service` | InfiniLM | Service mode performance |

### vLLM Tests

| Test Name | Framework | Metrics |
|-----------|----------|---------|
| `infer.vllm.*` | vLLM | Various vLLM inference modes |

## Metrics

### Throughput
- **Unit**: tokens per second (tokens/s)
- **Description**: Number of tokens processed per second
- **Higher is better**

### Latency
- **Unit**: milliseconds (ms)
- **Description**: Time to process a request
- **Lower is better**

### Memory Usage
- **Unit**: megabytes (MB)
- **Description**: GPU memory consumption
- **Varies by model and batch size**

## Configuration Examples

### InfiniLM Direct Mode

```json
{
    "run_id": "infer_infinilm_001",
    "testcase": "infer.infinilm.direct",
    "config": {
        "model_path": "/path/to/model",
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

### vLLM Inference

```json
{
    "run_id": "infer_vllm_001",
    "testcase": "infer.vllm.default",
    "config": {
        "model": "facebook/opt-125m",
        "tensor_parallel_size": 1,
        "batch_size": 16,
        "output_dir": "./output"
    },
    "metrics": [
        {"name": "infer.throughput"},
        {"name": "infer.latency"}
    ]
}
```

## Running Inference Tests

### Command Line

```bash
# Using main.py
python main.py inference_config.json

# Using inference module directly
cd infinimetrics/inference
python infer_main.py --config config.json --model infinilm-7b
```

## Understanding Results

### Typical Performance

| Model | Batch Size | Throughput (tokens/s) | Latency (ms) |
|-------|------------|----------------------|--------------|
| 7B | 1 | 50-100 | 20-50 |
| 7B | 32 | 500-1500 | 100-300 |
| 13B | 1 | 30-60 | 30-70 |
| 13B | 32 | 300-800 | 150-400 |

*Values vary by hardware and configuration*

### Prefill vs Decode

- **Prefill**: Processing input prompt
- **Decode**: Generating output tokens
- Prefill is typically faster than decode

## Performance Optimization

### Batch Size
- Larger batches improve throughput
- May increase latency per request
- Memory limited

### Tensor Parallelism
- Split model across multiple GPUs
- Enables larger models
- Communication overhead

### KV Cache
- Caches key-value pairs
- Improves decoding performance
- Memory intensive

## Troubleshooting

### Out of Memory

Reduce batch size or use smaller model:

```json
{
    "config": {
        "batch_size": 8  // Reduce from 32
    }
}
```

### Slow Performance

1. Check GPU utilization: `nvidia-smi`
2. Verify tensor parallelism settings
3. Ensure proper CUDA version

For more help, see [Troubleshooting](../troubleshooting.md).

## Examples

See [Inference Examples](../examples/inference_evaluation.md) for detailed examples.
