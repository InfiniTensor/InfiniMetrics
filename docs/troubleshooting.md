# Troubleshooting

This document covers common issues and their solutions when using InfiniMetrics.

## Common Issues

### CUDA Not Found

**Symptoms**: Error messages like "CUDA not found" or "nvcc not found"

**Solutions**:

1. Check CUDA installation:
```bash
nvcc --version
```

2. Set CUDA environment variables:
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

3. Verify CUDA driver version matches toolkit version

### Import Errors for InfiniLM/vLLM

**Symptoms**: `ModuleNotFoundError: No module named 'infinilm'` or similar

**Solutions**:

1. Install required dependencies:
```bash
pip install infinilm vllm
```

2. Verify installation:
```bash
python -c "import infinilm; print(infinilm.__version__)"
python -c "import vllm; print(vllm.__version__)"
```

3. If using virtual environment, ensure it's activated

### Hardware Tests Fail to Compile

**Symptoms**: Build errors when compiling CUDA memory benchmarks

**Solutions**:

1. Ensure CUDA toolkit is installed:
```bash
nvcc --version
cmake --version
```

2. Check compiler compatibility (GCC 11.3 recommended)

3. Clean and rebuild:
```bash
cd infinimetrics/hardware/cuda-memory-benchmark
rm -rf build
bash build.sh
```

### Out of Memory Errors

**Symptoms**: CUDA out of memory errors during testing

**Solutions**:

1. Reduce buffer size in configuration:
```json
{
    "config": {
        "buffer_size_mb": 128  // Reduce from 256
    }
}
```

2. Run tests individually instead of in batches

3. Close other GPU-intensive applications

### NCCL Tests Fail

**Symptoms**: NCCL test failures or communication errors

**Solutions**:

1. Verify NCCL installation:
```bash
python -c "import pynvml; print('NCCL available')"
```

2. Check multi-GPU setup:
```bash
nvidia-smi
```

3. Ensure proper network configuration for multi-node tests

### Permission Errors

**Symptoms**: "Permission denied" when writing output

**Solutions**:

1. Check output directory permissions:
```bash
ls -la ./output
```

2. Create output directory if missing:
```bash
mkdir -p ./output
chmod 755 ./output
```

3. Specify a different output directory in config

### JSON Configuration Errors

**Symptoms**: "Invalid JSON" or "Missing required field" errors

**Solutions**:

1. Validate JSON syntax:
```bash
python -m json.tool input.json
```

2. Ensure all required fields are present:
   - `run_id`
   - `testcase`
   - `config`

3. Check test case naming convention:
```
<category>.<framework>.<test_name>
```

## Getting Help

If you're still experiencing issues:

1. **Check the logs**: Look at `log.txt` in the test output directory
2. **Enable verbose mode**: Run with `--verbose` flag
3. **Search existing issues**: Check [GitHub Issues](https://github.com/InfiniTensor/InfiniMetrics/issues)
4. **Create a new issue**: Include:
   - Error message
   - Configuration file
   - System information (OS, GPU, driver versions)
   - Log files

## Debug Mode

For detailed debugging information:

```bash
python main.py input.json --verbose --output ./debug_output
```

This will:
- Print detailed execution information
- Save comprehensive logs
- Include stack traces in output

## System Information

When reporting issues, please provide:

```bash
# OS and kernel
uname -a

# Python version
python --version

# GPU information
nvidia-smi

# CUDA version
nvcc --version

# Installed packages
pip list | grep -E "(torch|vllm|infinilm|numpy|pandas)"
```
