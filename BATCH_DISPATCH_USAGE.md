# 批量调度使用指南

## 概述

Dispatcher 现在支持批量调度，可以一次性执行多个测试任务。

## 新功能

### 1. 批量调度

`dispatch` 方法现在支持接受单个配置或配置列表：

```python
from dispatcher import Dispatcher

# 单个配置（自动转换为列表）
config = {
    'testcase': 'infer.InfiniLM.Direct',
    'config': {...}
}
dispatcher = Dispatcher(default_config)
result = dispatcher.dispatch(config)

# 批量配置（推荐）
configs = [
    {'testcase': 'infer.InfiniLM.Direct', 'config': {...}},
    {'testcase': 'infer.InfiniLM.Batch', 'config': {...}},
    {'testcase': 'train.Operator.Conv', 'config': {...}}
]
dispatcher = Dispatcher(default_config)
result = dispatcher.dispatch(configs)  # 一次性执行所有测试
```

### 2. main.py 命令行工具

新增了 `main.py` 作为主入口，支持从文件或目录加载配置：

```bash
# 从单个配置文件运行测试
python main.py config.json

# 从多个配置文件运行测试
python main.py config1.json config2.json

# 从目录加载所有 JSON 配置文件
python main.py /path/to/configs/

# 混合文件和目录
python main.py config.json /path/to/configs/

# 指定输出目录
python main.py config.json --output ./results

# 指定框架
python main.py config.json --framework infinilm

# 启用详细日志
python main.py config.json --verbose
```

## 配置文件格式

### 单配置文件 (config.json)

```json
{
  "run_id": "test_001",
  "testcase": "infer.InfiniLM.Direct",
  "config": {
    "framework": "infinilm",
    "model": "Qwen3-1.7B",
    "model_path": "/path/to/model",
    "device": {
      "accelerator": "nvidia"
    },
    "infer_args": {
      "parallel": {"tp": 1},
      "static_batch_size": 1
    },
    "output_dir": "./output"
  }
}
```

### 多配置文件 (configs.json)

```json
[
  {
    "run_id": "test_001",
    "testcase": "infer.InfiniLM.Direct",
    "config": {...}
  },
  {
    "run_id": "test_002",
    "testcase": "infer.InfiniLM.Batch",
    "config": {...}
  }
]
```

## 工作流程

### 1. 准备配置文件

创建一个或多个 JSON 配置文件：

```bash
configs/
├── inference_test1.json
├── inference_test2.json
└── operator_test.json
```

### 2. 运行测试

```bash
python main.py configs/
```

### 3. 查看结果

main.py 会自动打印测试摘要：

```
============================================================
Test Summary
============================================================
Total tests:   3
Successful:    2
Failed:        1
Success rate:  66.7%

Failed tests:
  - infer.InfiniLM.Direct: Out of memory

Summary files: ./output/dispatcher_summary_*.json
============================================================
```

### 4. 查看详细结果

每个测试的详细结果保存在独立文件中：

```bash
# 查看 summary 获取所有测试的概览
cat ./output/dispatcher_summary_20250107_123000.json

# 查看特定测试的详细结果
cat ./output/infer_InfiniLM_Direct_20250107_123001_results.json
```

## API 变化

### Dispatcher.dispatch()

**之前**：
```python
def dispatch(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """只接受单个 payload"""
```

**现在**：
```python
def dispatch(self, configs: Any) -> Dict[str, Any]:
    """
    接受单个配置或配置列表
    - 单个 dict -> 自动转换为列表
    - list of dict -> 批量处理
    """
```

### 新增方法

```python
def _create_adapter_for_config(self, test_type: str, config: Dict[str, Any]) -> BaseAdapter:
    """
    根据特定的 test config 创建 adapter
    允许不同的测试使用不同的配置
    """
```

## 使用场景

### 场景 1: 对比不同配置

```python
configs = [
    {'testcase': 'infer.Test', 'config': {'batch_size': 1}},
    {'testcase': 'infer.Test', 'config': {'batch_size': 4}},
    {'testcase': 'infer.Test', 'config': {'batch_size': 8}},
]
result = dispatcher.dispatch(configs)
# 对比不同 batch size 的性能
```

### 场景 2: 测试多个模型

```python
configs = [
    {'testcase': 'infer.Model', 'config': {'model': 'Qwen3-1.7B'}},
    {'testcase': 'infer.Model', 'config': {'model': 'Qwen3-7B'}},
    {'testcase': 'infer.Model', 'config': {'model': 'Llama3-8B'}},
]
result = dispatcher.dispatch(configs)
# 对比不同模型的性能
```

### 场景 3: 回归测试

```bash
# 将所有回归测试配置放在一个目录
python main.py tests/regression/
```

## 命令行参数

```
usage: main.py [-h] [--output OUTPUT] [--framework {infinilm,vllm}] [--verbose]
                inputs [inputs ...]

positional arguments:
  inputs                配置文件或目录

optional arguments:
  -h, --help           显示帮助信息
  --output, -o OUTPUT  输出目录 (默认: ./output)
  --framework, -f {infinilm,vllm}
                       默认框架 (默认: infinilm)
  --verbose, -v        启用详细日志
```

## 示例

### 示例 1: 快速测试

```bash
# 使用示例配置文件
python main.py example_config.json
```

### 示例 2: 批量测试

```bash
# 从目录加载所有配置
mkdir -p test_configs
# ... 添加配置文件到 test_configs/
python main.py test_configs/
```

### 示例 3: 指定输出目录

```bash
python main.py config.json --output ./test_results_$(date +%Y%m%d)
```

### 示例 4: 程序化使用

```python
from main import load_configs, run_tests

# 加载配置
configs = load_configs(['config1.json', 'config2.json'])

# 运行测试
default_config = {'framework': 'infinilm', 'output_dir': './output'}
results = run_tests(configs, default_config)

# 处理结果
print(f"Success: {results['successful_tests']}/{results['total_tests']}")
```

## 注意事项

1. **配置一致性**：每个配置必须包含 `testcase` 字段
2. **输出目录**：如果不指定，所有测试结果保存在 `./output` 目录
3. **错误处理**：单个测试失败不会中断整个批处理
4. **资源管理**：批量测试时注意资源使用，避免 OOM
5. **文件引用**：summary 文件只包含文件引用，不包含详细数据

## 迁移指南

如果你有旧代码使用单个 payload：

### 旧代码
```python
payload = {'testcase': '...', 'config': {...}}
result = dispatcher.dispatch(payload)
```

### 新代码（无需改动）
```python
# 仍然支持，自动转换为批量模式
payload = {'testcase': '...', 'config': {...}}
result = dispatcher.dispatch(payload)

# 或者使用显式的批量模式
payloads = [{'testcase': '...', 'config': {...}}]
result = dispatcher.dispatch(payloads)
```

完全向后兼容！
