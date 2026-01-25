# Operator Tests

Operator tests measure the performance of individual computational operations and kernels, providing detailed insights into specific computational patterns.

## InfiniCore Operator Tests

| Test Name | Framework | Description |
|-----------|----------|-------------|
| `operator.infinicore.*` | InfiniCore | Individual operator performance, FLOPS calculation |

## Supported Operators

InfiniCore tests support various operations including:
- Convolution (Conv2D)
- Matrix multiplication (MatMul)
- Element-wise operations
- Reduction operations

## FLOPS Calculation

Operator tests include automatic FLOPS (Floating Point Operations Per Second) calculation:

- **Conv2D**: `2 * H_out * W_out * C_in * K_h * K_w * C_out`
- **MatMul**: `2 * M * N * K`

## Configuration Example

```json
{
    "run_id": "op_test_001",
    "testcase": "operator.infinicore.Conv2D",
    "config": {
        "input_shape": [1, 64, 224, 224],
        "kernel_shape": [64, 64, 3, 3],
        "stride": 1,
        "padding": 1,
        "output_dir": "./output"
    },
    "metrics": [
        {"name": "operator.flops"},
        {"name": "operator.latency"},
        {"name": "operator.throughput"}
    ]
}
```

## Running Operator Tests

```bash
python main.py operator_test_config.json
```

## Understanding Results

### FLOPS

Floating Point Operations Per Second - measures computational throughput.

- Higher values indicate better performance
- Compare against theoretical peak (FLOPs/cycle * cores * clock)

### Latency

Time taken to complete a single operation.

- Lower values indicate better performance
- Important for real-time applications

### Throughput

Operations completed per unit time.

- Higher values indicate better performance
- Important for batch processing

## Performance Optimization Tips

1. **Batch Size**: Larger batches often improve throughput
2. **Tensor Layout**: Use optimal memory layouts (NHWC vs NCHW)
3. **Precision**: Consider mixed precision (FP16/BF16) for better performance
4. **Operator Fusion**: Fused operators reduce memory transfer

## Adding New Operators

To add new operator tests, see [Development Guide](../development.md).

## Examples

See [Examples](../examples/README.md) for more operator test configurations.
