# 推理评估示例

本文档提供使用 InfiniLM 和 vLLM 运行推理基准测试的示例。

## 示例 1：InfiniLM 直接模式

以直接模式运行 InfiniLM 推理以获得基线性能。

### 配置

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

### 运行

```bash
python main.py infinilm_direct_config.json
```

### 收集的指标
- **吞吐量**: tokens 每秒
- **延迟**: 每个请求的毫秒数
- **内存使用**: GPU 内存（MB）

## 示例 2：InfiniLM 预填充

单独测试预填充阶段性能。

### 配置

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

### 运行

```bash
python main.py infinilm_prefill_config.json
```

## 示例 3：vLLM 推理

使用 vLLM 框架运行推理。

### 配置

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

### 运行

```bash
python main.py vllm_config.json
```

## 示例 4：批大小比较

比较不同批大小的性能。

### 配置 1（小批次）

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

### 配置 2（大批次）

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

### 运行两个测试

```bash
python main.py batch_small_config.json
python main.py batch_large_config.json
```

## 示例 5：直接使用推理模块

您也可以直接使用推理模块而不通过 main.py。

### 命令

```bash
cd infinimetrics/inference
python infer_main.py --config config.json --model /path/to/model --batch-size 32
```

### 配置文件 (config.json)

```json
{
    "model_path": "/path/to/infinilm-7b",
    "batch_size": 32,
    "prompt_length": 128,
    "max_length": 256,
    "output_file": "./inference_results.json"
}
```

## 理解结果

### 典型性能（7B 模型）

| 批大小 | 吞吐量 (tokens/s) | 延迟 (ms) |
|------------|----------------------|--------------|
| 1 | 50-100 | 20-50 |
| 8 | 200-500 | 50-150 |
| 16 | 400-800 | 80-200 |
| 32 | 500-1500 | 100-300 |
| 64 | 600-2000 | 150-400 |

*值因硬件和模型大小而异*

### 解释指标

- **吞吐量**: 对于批处理，越高越好
- **延迟**: 对于交互式应用，越低越好
- **内存使用**: 应在 GPU 内存限制内

## 性能技巧

### 优化吞吐量
- 增加批大小
- 对更大的模型使用张量并行
- 启用 KV 缓存优化

### 优化延迟
- 使用较小的批大小
- 减少提示和最大长度
- 使用量化（FP16/BF16）

### 内存优化
- 减少批大小
- 使用更小的模型
- 启用梯度检查点（用于训练）

## 故障排除

### 内存不足

```json
{
    "config": {
        "batch_size": 8  // 从 32 减少
    }
}
```

### 性能缓慢

1. 检查 GPU 利用率: `nvidia-smi`
2. 验证模型正确加载
3. 确保正确的 CUDA 版本
4. 尝试不同的批大小

### 导入错误

```bash
pip install infinilm vllm torch
```

## 后续步骤

- 参阅[推理测试文档](../test_types/inference_tests.md)了解详情
- 探索[高级用法](./advanced_usage.md)了解更多示例
