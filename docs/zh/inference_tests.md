# 推理测试

推理测试评估端到端模型推理性能，测量吞吐量、延迟和资源利用率。

## 支持的框架

### InfiniLM 测试

| 测试名称 | 框架 | 指标 |
|-----------|----------|---------|
| `infer.infinilm.direct` | InfiniLM | 吞吐量 (tokens/s)、延迟 (ms)、内存使用 |
| `infer.infinilm.prefill` | InfiniLM | 预填充阶段指标 |
| `infer.infinilm.service` | InfiniLM | 服务模式性能 |

### vLLM 测试

| 测试名称 | 框架 | 指标 |
|-----------|----------|---------|
| `infer.vllm.*` | vLLM | 各种 vLLM 推理模式 |

## 指标

### 吞吐量
- **单位**: tokens 每秒 (tokens/s)
- **描述**: 每秒处理的 token 数
- **越高越好**

### 延迟
- **单位**: 毫秒 (ms)
- **描述**: 处理请求的时间
- **越低越好**

### 内存使用
- **单位**: 兆字节 (MB)
- **描述**: GPU 内存消耗
- **因模型和批大小而异**

## 配置示例

### InfiniLM 直接模式

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

### vLLM 推理

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

## 运行推理测试

### 命令行

```bash
# 使用 main.py
python main.py inference_config.json

# 直接使用推理模块
cd infinimetrics/inference
python infer_main.py --config config.json --model infinilm-7b
```

## 理解结果

### 典型性能

| 模型 | 批大小 | 吞吐量 (tokens/s) | 延迟 (ms) |
|-------|------------|----------------------|--------------|
| 7B | 1 | 50-100 | 20-50 |
| 7B | 32 | 500-1500 | 100-300 |
| 13B | 1 | 30-60 | 30-70 |
| 13B | 32 | 300-800 | 150-400 |

*值因硬件和配置而异*

### 预填充 vs 解码

- **预填充**: 处理输入提示
- **解码**: 生成输出 token
- 预填充通常比解码快

## 性能优化

### 批大小
- 较大的批次提高吞吐量
- 可能增加每个请求的延迟
- 受内存限制

### 张量并行
- 跨多个 GPU 分割模型
- 支持更大的模型
- 通信开销

### KV 缓存
- 缓存键值对
- 提高解码性能
- 内存密集

## 故障排除

### 内存不足

减少批大小或使用更小的模型：

```json
{
    "config": {
        "batch_size": 8  // 从 32 减少
    }
}
```

### 性能缓慢

1. 检查 GPU 利用率: `nvidia-smi`
2. 验证张量并行设置
3. 确保正确的 CUDA 版本

更多帮助，请参阅[故障排除](./troubleshooting.md)。

## 示例

详细示例，请参阅[推理评估示例](./inference_evaluation.md)。
