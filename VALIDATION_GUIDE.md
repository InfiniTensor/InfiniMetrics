# 配置校验指南

## 概述

Dispatcher 提供了 `validate_config()` 方法，允许你自定义配置校验逻辑。

## 默认行为

默认的 `validate_config()` 只检查 `testcase` 字段是否存在：

```python
def validate_config(self, config: Dict[str, Any]) -> bool:
    """默认校验：检查 testcase 是否存在"""
    return 'testcase' in config
```

## 自定义校验

### 方法 1: 继承 Dispatcher 并覆盖 validate_config

```python
from dispatcher import Dispatcher

class MyDispatcher(Dispatcher):
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """自定义校验逻辑"""
        # 基础检查
        if 'testcase' not in config:
            return False

        # 检查必需字段
        required_fields = ['model', 'model_path', 'device']
        for field in required_fields:
            if field not in config:
                logger.warning(f"Missing required field: {field}")
                return False

        # 检查模型路径是否存在
        from pathlib import Path
        model_path = config.get('model_path')
        if model_path and not Path(model_path).exists():
            logger.warning(f"Model path does not exist: {model_path}")
            return False

        # 检查设备配置
        device = config.get('device', {})
        if not device:
            logger.warning("Device configuration is empty")
            return False

        # 所有检查通过
        return True

# 使用自定义 Dispatcher
dispatcher = MyDispatcher(default_config)
results = dispatcher.dispatch(configs)
```

### 方法 2: 动态替换方法

```python
from dispatcher import Dispatcher

def my_validator(config):
    """自定义校验函数"""
    if 'testcase' not in config:
        return False

    # 检查 framework
    framework = config.get('framework')
    if framework not in ['infinilm', 'vllm']:
        logger.warning(f"Unsupported framework: {framework}")
        return False

    return True

# 创建 Dispatcher 并替换 validate_config 方法
dispatcher = Dispatcher(default_config)
dispatcher.validate_config = my_validator

results = dispatcher.dispatch(configs)
```

## 常见校验场景

### 场景 1: 检查必需字段

```python
def validate_config(self, config):
    required_fields = ['testcase', 'framework', 'model_path']
    return all(field in config for field in required_fields)
```

### 场景 2: 检查字段值

```python
def validate_config(self, config):
    # 检查 batch_size 范围
    batch_size = config.get('infer_args', {}).get('static_batch_size', 1)
    if batch_size < 1 or batch_size > 128:
        logger.warning(f"Invalid batch_size: {batch_size}")
        return False

    return True
```

### 场景 3: 检查文件/目录存在性

```python
def validate_config(self, config):
    from pathlib import Path

    model_path = config.get('model_path')
    if not model_path or not Path(model_path).exists():
        logger.warning(f"Model path not found: {model_path}")
        return False

    return True
```

### 场景 4: 条件校验

```python
def validate_config(self, config):
    testcase = config.get('testcase', '')

    # 推理测试需要 model_path
    if 'infer' in testcase.lower():
        if 'model_path' not in config:
            logger.warning("Inference test requires model_path")
            return False

    # 训练测试需要 operator
    if 'train' in testcase.lower():
        if 'operator' not in config:
            logger.warning("Training test requires operator")
            return False

    return True
```

## 日志输出

当配置被跳过时，会记录警告日志：

```
WARNING - Skipping invalid config (validation failed): infer.InfiniLM.Direct
```

统计信息会显示多少配置被跳过：

```
INFO - Dispatcher: Processing 5 valid test payloads (skipped 2 invalid)
```

## 完整示例

```python
#!/usr/bin/env python3
from dispatcher import Dispatcher
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class StrictDispatcher(Dispatcher):
    """严格校验的 Dispatcher"""

    def validate_config(self, config):
        """严格的配置校验"""
        # 检查 testcase
        if 'testcase' not in config:
            logger.error("Missing testcase field")
            return False

        # 检查 framework
        framework = config.get('framework')
        if framework not in ['infinilm', 'vllm']:
            logger.error(f"Invalid framework: {framework}")
            return False

        # 检查 model_path
        model_path = config.get('model_path')
        if not model_path:
            logger.error("Missing model_path")
            return False

        if not Path(model_path).exists():
            logger.error(f"Model path does not exist: {model_path}")
            return False

        # 检查 device 配置
        device = config.get('device')
        if not device:
            logger.error("Missing device configuration")
            return False

        # 检查 accelerator
        accelerator = device.get('accelerator')
        if accelerator not in ['nvidia', 'amd', 'intel', 'cpu']:
            logger.error(f"Invalid accelerator: {accelerator}")
            return False

        logger.info(f"Config validation passed: {config['testcase']}")
        return True

# 使用示例
if __name__ == '__main__':
    dispatcher = StrictDispatcher({'framework': 'infinilm'})
    results = dispatcher.dispatch(configs)
```

## 最佳实践

1. **快速失败**：先检查简单字段，再检查耗时操作（如文件存在性）
2. **详细日志**：记录校验失败的具体原因
3. **可配置性**：考虑将校验规则配置化
4. **不要过度校验**：保持校验逻辑简单，避免重复 Executor/Adapter 的校验

## 注意事项

- `validate_config()` 在每个 config 上调用，要确保执行速度快
- 返回 `False` 的 config 会被静默跳过，只记录警告日志
- 不要在 `validate_config()` 中做耗时操作（如网络请求、加载模型等）
- 如果需要复杂的校验，考虑创建单独的校验工具类
