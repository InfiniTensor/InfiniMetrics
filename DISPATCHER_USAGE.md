# Dispatcher 使用指南

## 概述

Dispatcher 现在支持基于配置的 adapter 初始化，不再需要从外部传入 adapter 实例。

## 新的 API 使用方式

### 1. 基本用法

```python
from dispatcher import Dispatcher

# 准备配置
config = {
    'framework': 'infinilm',  # 指定使用的框架
    'model': 'Qwen3-1.7B',
    'model_path': '/path/to/model',
    'device': {
        'accelerator': 'nvidia'
    },
    'output_dir': './output'
}

# 创建 Dispatcher（只传入 config）
dispatcher = Dispatcher(config)

# 分发测试任务
payload = {
    'run_id': 'test_001',
    'testcase': 'infer.InfiniLM.Direct',
    'config': config
}

result = dispatcher.dispatch(payload)
```

### 2. 关键变化

**之前的设计**：
```python
# 需要手动创建 adapter
adapters = {
    'inference': InfiniLMAdapter(config),
    'operator': OperatorAdapter(config)
}
dispatcher = Dispatcher(adapters)
```

**新的设计**：
```python
# 只需传入 config，Dispatcher 内部创建 adapter
dispatcher = Dispatcher(config)
```

### 3. Adapter 创建逻辑

Dispatcher 会根据 testcase 类型自动选择合适的 adapter：

- `testcase` 包含 'infer' → 创建 Inference Adapter
- `testcase` 包含 'train' 或 'operator' → 创建 Operator Adapter
- 其他情况 → 默认使用 Inference Adapter

### 4. 框架支持

当前支持的框架：
- `infinilm` - InfiniLM 框架（已实现）
- `vllm` - VLLM 框架（待实现）

### 5. 错误处理

当 adapter 创建失败时，Dispatcher 会自动使用 Error Adapter 返回错误信息，而不会崩溃：

```python
result = dispatcher.dispatch(payload)
if result['failed_tests'] > 0:
    print(f"Test failed: {result['results'][0].get('error')}")
```

### 6. 生命周期管理

每次 `dispatch()` 调用都会：
1. 创建新的 adapter 实例
2. Executor 调用 `adapter.setup()`
3. 执行 `adapter.process()`
4. Executor 调用 `adapter.teardown()`

这样确保每个测试都是独立的状态，避免状态污染。

## 测试

运行测试验证功能：

```bash
python -m unittest tests.test_dispatcher -v
```

所有测试应该通过（13 个测试用例）。

## 配置示例

### InfiniLM 框架完整配置

```python
config = {
    # 框架类型
    'framework': 'infinilm',

    # 模型配置
    'model': 'Qwen3-1.7B',
    'model_path': '/path/to/model',
    'model_config': '/path/to/model/config.json',

    # 设备配置
    'device': {
        'accelerator': 'nvidia',  # nvidia, amd, intel, cpu
        'device_ids': [0],
        'cpu_only': False
    },

    # 推理参数
    'infer_args': {
        'parallel': {
            'tp': 1,  # tensor parallelism
            'dp': 1,  # data parallelism
        },
        'static_batch_size': 1,
        'prompt_token_num': 1024,
        'output_token_num': 128,
        'max_seq_len': 4096,
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 50
    },

    # 执行参数
    'warmup_iterations': 10,
    'measured_iterations': 100,

    # 输出目录
    'output_dir': './output'
}
```

## 迁移指南

如果你有使用旧 API 的代码，需要做以下修改：

### 旧代码
```python
from adapter import InfiniLMAdapter, OperatorAdapter
from dispatcher import Dispatcher

adapters = {
    'inference': InfiniLMAdapter(config),
    'operator': OperatorAdapter(config)
}
dispatcher = Dispatcher(adapters)
```

### 新代码
```python
from dispatcher import Dispatcher

dispatcher = Dispatcher(config)
```

只需删除 adapter 创建代码，直接传入 config 即可。
