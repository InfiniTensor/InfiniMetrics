# Testcase 命名规范

## 概述

Dispatcher 通过解析 `testcase` 字段来确定测试类型（test_type）和使用的框架（framework）。

## 命名格式

testcase 必须遵循以下格式：

```
<test_type>.<Framework>.<SpecificTest>
```

### 字段说明

1. **test_type**: 测试类型
   - `infer`: 推理测试
   - `train`: 训练测试
   - `eval`: 评估测试

2. **Framework**: 使用的框架
   - `InfiniLM`: InfiniLM 推理框架
   - `InfiniTrain`: InfiniTrain 训练框架
   - `vLLM`: vLLM 推理框架
   - 其他自定义框架名称

3. **SpecificTest**: 具体测试名称
   - 可以包含多个子段，用 `.` 分隔
   - 例如: `Direct`, `Batch`, `Conv`, `SFT` 等

## 解析规则

### test_type 映射

Dispatcher 将 testcase 的第一段映射为内部 test_type：

| testcase 第一段 | 内部 test_type | 说明 |
|----------------|---------------|------|
| `infer` | `inference` | 推理测试 |
| `train` | `operator` | 训练/算子测试 |
| `eval` | `operator` | 评估/算子测试 |
| 其他 | `operator` | 默认为算子测试 |

### framework 提取

framework 从 testcase 的第二段提取，并转换为小写：

| testcase 第二段 | 提取的 framework |
|----------------|-----------------|
| `InfiniLM` | `infinilm` |
| `InfiniTrain` | `infinitrain` |
| `vLLM` | `vllm` |

## 示例

### 推理测试

```
testcase: "infer.InfiniLM.Direct"
解析结果:
  - test_type: "inference"
  - framework: "infinilm"
```

```
testcase: "infer.vLLM.Batch"
解析结果:
  - test_type: "inference"
  - framework: "vllm"
```

### 训练测试

```
testcase: "train.InfiniTrain.SFT"
解析结果:
  - test_type: "operator"
  - framework: "infinitrain"
```

```
testcase: "train.Operator.Conv"
解析结果:
  - test_type: "operator"
  - framework: "operator"
```

### 评估测试

```
testcase: "eval.Evaluator.Accuracy"
解析结果:
  - test_type: "operator"
  - framework: "evaluator"
```

## 错误处理

如果 testcase 格式无效（少于 2 个段），Dispatcher 会：

1. 记录警告日志
2. 使用默认值：
   - test_type: `operator`
   - framework: `operator`

示例：
```python
testcase: "invalid"
解析结果:
  - test_type: "operator"
  - framework: "operator"
  - 日志: WARNING - Invalid testcase format: invalid, using defaults
```

## Framework 优先级

Dispatcher 从 testcase 解析的 framework **会覆盖** config 中的 framework 设置：

```python
payload = {
    'testcase': 'infer.InfiniLM.Direct',
    'config': {
        'framework': 'vllm'  # 会被忽略
    }
}

# 实际使用的 framework 是 'infinilm'（从 testcase 解析）
```

这确保了 testcase 名称与实际使用的框架保持一致。

## 配置文件示例

### 单个推理测试

```json
{
  "run_id": "test_001",
  "testcase": "infer.InfiniLM.Direct",
  "config": {
    "model": "Qwen3-1.7B",
    "model_path": "/path/to/model",
    "device": {"accelerator": "nvidia"},
    "infer_args": {
      "parallel": {"tp": 1},
      "static_batch_size": 1
    }
  }
}
```

### 多个框架对比测试

```json
[
  {
    "run_id": "test_infinilm",
    "testcase": "infer.InfiniLM.Direct",
    "config": {
      "model": "Qwen3-1.7B",
      "model_path": "/path/to/model"
    }
  },
  {
    "run_id": "test_vllm",
    "testcase": "infer.vLLM.Direct",
    "config": {
      "model": "Qwen3-1.7B",
      "model_path": "/path/to/model"
    }
  }
]
```

### 训练测试

```json
{
  "run_id": "train_sft",
  "testcase": "train.InfiniTrain.SFT",
  "config": {
    "operator": "Conv",
    "device": "cuda",
    "train_args": {
      "epochs": 10,
      "batch_size": 32
    }
  }
}
```

## 最佳实践

1. **命名一致性**: testcase 名称应清晰反映测试类型和使用的框架
2. **大小写**: framework 段使用 PascalCase（如 InfiniLM），会被自动转换为小写
3. **层次结构**: 使用多级 `.` 分隔来组织测试层次
   - 好: `infer.InfiniLM.Direct.SingleToken`
   - 不好: `infer.InfiniLM.direct_single_token`
4. **文档化**: 为每个 testcase 命名模式编写文档，便于团队理解

## 代码示例

### 手动解析 testcase

```python
from dispatcher import Dispatcher

dispatcher = Dispatcher({})

# 解析 testcase
test_type, framework = dispatcher._parse_testcase('infer.InfiniLM.Direct')
print(f"test_type: {test_type}")      # test_type: inference
print(f"framework: {framework}")      # framework: infinilm
```

### 自定义 test_type 映射

如果需要添加新的 test_type 映射，修改 [`_parse_testcase()`](dispatcher.py#L104) 方法中的 `test_type_mapping`：

```python
test_type_mapping = {
    'infer': 'inference',
    'train': 'operator',
    'eval': 'operator',
    'benchmark': 'benchmark',  # 新增
    'custom': 'custom'         # 新增
}
```

## 相关文档

- [DISPATCHER_USAGE.md](DISPATCHER_USAGE.md) - Dispatcher 使用指南
- [BATCH_DISPATCH_USAGE.md](BATCH_DISPATCH_USAGE.md) - 批量调度使用指南
- [VALIDATION_GUIDE.md](VALIDATION_GUIDE.md) - 配置校验指南
