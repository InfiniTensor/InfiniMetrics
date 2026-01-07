# TestInput 类使用指南

## 概述

`TestInput` 是一个数据类，用于表示测试框架中的测试输入数据。它封装了测试用例的所有信息，包括配置、运行ID、时间戳和指标定义。

## 位置

```python
from infinimetrics.input import TestInput
```

## 类结构

```python
@dataclass
class TestInput:
    testcase: str                           # 必填：测试用例名称
    run_id: Optional[str] = None            # 可选：运行标识符
    time: Optional[str] = None              # 可选：时间戳（自动生成）
    success: Optional[int] = None           # 可选：成功标志（0=成功）
    config: Dict[str, Any] = {}             # 可选：配置字典
    metrics: List[Dict[str, Any]] = []      # 可选：指标定义列表
```

## 使用方法

### 1. 创建最小化输入

只需要提供必填的 `testcase` 字段：

```python
from infinimetrics.input import TestInput

test_input = TestInput(testcase="infer.InfiniLM.Direct")
# time 字段会自动设置为当前时间
```

### 2. 创建完整输入

提供所有字段：

```python
test_input = TestInput(
    testcase="train.InfiniTrain.SFT",
    run_id="test.my-run.001",
    success=0,
    config={
        "operator": "Conv",
        "device": "cuda",
        "input_shape": [1, 64, 256, 256],
        "output_shape": [1, 128, 254, 254]
    },
    metrics=[
        {"name": "latency", "unit": "ms"},
        {"name": "memory_usage", "unit": "MB"}
    ]
)
```

### 3. 从字典创建

从JSON文件或字典加载：

```python
data = {
    "testcase": "infer.InfiniLM.Direct",
    "run_id": "infer.test.123",
    "config": {
        "model": "Qwen3-1.7B",
        "device": {"gpu_platform": "nvidia"}
    }
}

test_input = TestInput.from_dict(data)
```

### 4. 转换为字典

```python
test_input = TestInput(testcase="infer.Test")
data_dict = test_input.to_dict()
# 可以序列化为JSON
import json
json_str = json.dumps(data_dict, indent=2)
```

### 5. 配置辅助方法

```python
test_input = TestInput(
    testcase="test",
    config={"model": "Qwen3-1.7B"}
)

# 获取配置值
model = test_input.get_config_value("model")  # "Qwen3-1.7B"
missing = test_input.get_config_value("missing", "default")  # "default"

# 设置配置值
test_input.set_config_value("device", {"gpu_platform": "nvidia"})
```

## 输入文件格式示例

### 推理测试输入 (format_infer.json)

```json
{
  "run_id": "infer.infinilm.direct.test",
  "testcase": "infer.InfiniLM.Direct",
  "config": {
    "model": "Qwen3-1.7B",
    "model_path": "/home/model/Qwen3-1.7B",
    "device": {
      "gpu_platform": "nvidia",
      "device_ids": [0]
    },
    "infer_args": {
      "parallel": {"dp": 1, "tp": 1},
      "static_batch_size": 1,
      "prompt_token_num": 100,
      "output_token_num": 50
    },
    "output_dir": "./test_output",
    "warmup_iterations": 2,
    "measured_iterations": 5
  }
}
```

### 算子测试输入 (format_input.json)

```json
{
  "run_id": "train.infiniTrain.SFT.a8b4c9e1",
  "time": "2025-10-11 14:50:50",
  "testcase": "train.InfiniTrain.SFT",
  "success": 0,
  "config": {
    "model_source": "FM9G_70B",
    "operator": "Conv",
    "device": "cuda",
    "attributes": [...],
    "inputs": [...],
    "outputs": [...],
    "warmup_iterations": 100,
    "measured_iterations": 1000,
    "bench": true
  },
  "metrics": [
    {
      "name": "operator.latency",
      "type": "timeseries",
      "raw_data_url": "./operator/${run_id}_operator_latency.csv",
      "unit": "ms"
    },
    {
      "name": "operator.flops",
      "type": "timeseries",
      "unit": "TFLOPS"
    }
  ]
}
```

## 验证规则

1. **testcase 必须是非空字符串**
   ```python
   TestInput(testcase="")  # 抛出 ValueError
   TestInput(testcase=None)  # 抛出 ValueError
   ```

2. **time 字段自动生成**
   如果不提供 `time` 字段，会自动设置为当前时间（格式：`YYYY-MM-DD HH:MM:SS`）

3. **可选字段不会出现在输出中**
   使用 `to_dict()` 时，值为 `None` 的可选字段不会包含在输出字典中

## 常见用例

### 用例1：从JSON文件加载

```python
import json
from infinimetrics.input import TestInput

# 读取JSON文件
with open('test_input.json', 'r') as f:
    data = json.load(f)

# 创建TestInput对象
test_input = TestInput.from_dict(data)

# 使用
print(f"Running test: {test_input.testcase}")
```

### 用例2：批量创建测试输入

```python
testcases = [
    "infer.InfiniLM.Direct",
    "infer.InfiniLM.Service",
    "train.InfiniTrain.SFT"
]

inputs = [
    TestInput(testcase=tc, config={"device": "cuda"})
    for tc in testcases
]
```

### 用例3：动态修改配置

```python
test_input = TestInput(
    testcase="test",
    config={"model": "Qwen3-1.7B"}
)

# 根据条件修改配置
if use_gpu:
    test_input.set_config_value("device", {"gpu_platform": "nvidia"})
else:
    test_input.set_config_value("device", {"cpu_only": True})
```

## 与现有代码的集成

### 在 Dispatcher 中使用

```python
from infinimetrics.input import TestInput
from infinimetrics.dispatcher import Dispatcher

# 创建测试输入
test_input_obj = TestInput(
    testcase="infer.InfiniLM.Direct",
    config={"model": "Qwen3-1.7B", "output_dir": "./output"}
)

# 转换为字典并分发
dispatcher = Dispatcher()
result = dispatcher.dispatch(test_input_obj.to_dict())
```

### 在 Adapter 中使用

```python
class MyAdapter(BaseAdapter):
    def process(self, test_input: Dict[str, Any]) -> Dict[str, Any]:
        # 将test_input转换为TestInput对象
        test_input_obj = TestInput.from_dict(test_input)

        # 使用辅助方法访问配置
        model = test_input_obj.get_config_value("model")
        device = test_input_obj.get_config_value("device")

        # 执行测试...
        return {
            'success': 0,
            'data': {...},
            'metrics': [...]
        }
```

## 注意事项

1. **testcase 是必填字段**，创建时必须提供
2. **config 字段默认为空字典**，可以后续添加
3. **time 字段会自动生成**，除非明确提供
4. **to_dict() 只包含非None的可选字段**，保持输出简洁
5. **from_dict() 和 to_dict() 是可逆的**，可以往返转换

## 测试

运行测试：

```bash
python -m unittest tests.test_input -v
```

所有测试应该通过：

```
Ran 10 tests in 0.001s
OK
```
