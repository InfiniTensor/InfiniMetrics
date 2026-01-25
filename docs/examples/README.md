# Usage Examples

This section provides practical examples for using InfiniMetrics across different scenarios.

## Example Categories

### Hardware Benchmarks
- [Hardware Benchmark Examples](./hardware_benchmarks.md) - Memory, STREAM, and cache tests

### Inference Evaluation
- [Inference Examples](./inference_evaluation.md) - InfiniLM and vLLM inference tests

### Advanced Usage
- [Advanced Usage](./advanced_usage.md) - Multiple tests, custom configurations, batching

## Quick Examples

### Run a Single Test

```bash
python main.py format_input_comprehensive_hardware.json
```

### Run Multiple Tests

```bash
# Run all JSON configs in a directory
python main.py ./test_configs/
```

### Verbose Output

```bash
python main.py input.json --verbose
```

### Custom Output Directory

```bash
python main.py input.json --output ./my_results
```

## Example Configuration Files

The repository includes several example configurations:

- `format_input_comprehensive_hardware.json` - Comprehensive hardware benchmark
- Additional examples can be found in individual example documents

## Creating Your Own Configuration

1. Start with an example configuration
2. Modify parameters for your needs
3. Run with `--verbose` to verify
4. Check output in `./output/` directory

## Next Steps

- Explore specific example categories
- See [Configuration Guide](../configuration.md) for parameter details
- Refer to [Test Types](../test_types/README.md) for available tests
