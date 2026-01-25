# 故障排除

本文档涵盖使用 InfiniMetrics 时的常见问题和解决方案。

## 常见问题

### 未找到 CUDA

**症状**: 出现类似 "CUDA not found" 或 "nvcc not found" 的错误消息

**解决方案**:

1. 检查 CUDA 安装：
```bash
nvcc --version
```

2. 设置 CUDA 环境变量：
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

3. 验证 CUDA 驱动版本与工具包版本匹配

### InfiniLM/vLLM 导入错误

**症状**: `ModuleNotFoundError: No module named 'infinilm'` 或类似错误

**解决方案**:

1. 安装所需的依赖：
```bash
pip install infinilm vllm
```

2. 验证安装：
```bash
python -c "import infinilm; print(infinilm.__version__)"
python -c "import vllm; print(vllm.__version__)"
```

3. 如果使用虚拟环境，确保已激活

### 硬件测试编译失败

**症状**: 编译 CUDA 内存基准测试时出现构建错误

**解决方案**:

1. 确保已安装 CUDA 工具包：
```bash
nvcc --version
cmake --version
```

2. 检查编译器兼容性（推荐 GCC 11.3）

3. 清理并重新构建：
```bash
cd infinimetrics/hardware/cuda-memory-benchmark
rm -rf build
bash build.sh
```

### 内存不足错误

**症状**: 测试期间出现 CUDA 内存不足错误

**解决方案**:

1. 减少配置中的缓冲区大小：
```json
{
    "config": {
        "buffer_size_mb": 128  // 从 256 减少
    }
}
```

2. 单独运行测试而不是批量运行

3. 关闭其他 GPU 密集型应用程序

### NCCL 测试失败

**症状**: NCCL 测试失败或通信错误

**解决方案**:

1. 验证 NCCL 安装：
```bash
python -c "import pynvml; print('NCCL available')"
```

2. 检查多 GPU 设置：
```bash
nvidia-smi
```

3. 确保多节点测试的网络配置正确

### 权限错误

**症状**: 写入输出时出现 "Permission denied" 错误

**解决方案**:

1. 检查输出目录权限：
```bash
ls -la ./output
```

2. 如果缺失则创建输出目录：
```bash
mkdir -p ./output
chmod 755 ./output
```

3. 在配置中指定不同的输出目录

### JSON 配置错误

**症状**: "Invalid JSON" 或 "Missing required field" 错误

**解决方案**:

1. 验证 JSON 语法：
```bash
python -m json.tool input.json
```

2. 确保所有必填字段都存在：
   - `run_id`
   - `testcase`
   - `config`

3. 检查测试用例命名约定：
```
<category>.<framework>.<test_name>
```

## 获取帮助

如果您仍然遇到问题：

1. **检查日志**: 查看测试输出目录中的 `log.txt`
2. **启用详细模式**: 使用 `--verbose` 标志运行
3. **搜索现有问题**: 查看 [GitHub Issues](https://github.com/InfiniTensor/InfiniMetrics/issues)
4. **创建新问题**: 包括：
   - 错误消息
   - 配置文件
   - 系统信息（操作系统、GPU、驱动版本）
   - 日志文件

## 调试模式

获取详细的调试信息：

```bash
python main.py input.json --verbose --output ./debug_output
```

这将：
- 打印详细的执行信息
- 保存全面的日志
- 在输出中包含堆栈跟踪

## 系统信息

报告问题时，请提供：

```bash
# 操作系统和内核
uname -a

# Python 版本
python --version

# GPU 信息
nvidia-smi

# CUDA 版本
nvcc --version

# 已安装的包
pip list | grep -E "(torch|vllm|infinilm|numpy|pandas)"
```
