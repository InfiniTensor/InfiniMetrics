# 结果文件结构说明

## 概述

为了优化存储和避免 summary 文件过大，我们采用**文件引用**而不是**数据嵌入**的方式存储测试结果。

## 文件结构

### 1. Executor 结果文件（详细）

每个测试执行后会生成一个独立的结果文件：

```
output/
├── infer_InfiniLM_Direct_20250107_123000_results.json
├── train_InfiniTrain_SFT_20250107_123005_results.json
└── ...
```

**文件命名格式**：`{testcase}_{timestamp}_results.json`

**文件内容**：
```json
{
  "run_id": "infer.infinilm.direct.abc123",
  "testcase": "infer.InfiniLM.Direct",
  "success": 1,
  "start_time": 1704601200.123,
  "duration": 123.45,
  "data": {
    "model": "Qwen3-1.7B",
    "inference_results": [...],
    "metrics_data": {...}
  },
  "metrics": [
    {"name": "execution.duration", "value": 123.45, "unit": "s"},
    {"name": "gpu.memory.peak", "value": 2048, "unit": "MB"},
    {"name": "throughput", "value": 10.5, "unit": "tokens/s"}
  ],
  "error": null
}
```

### 2. Dispatcher Summary 文件（轻量级）

Dispatcher 生成的 summary 只包含元信息和文件引用：

```
output/
└── dispatcher_summary_20250107_123010.json
```

**文件内容**：
```json
{
  "total_tests": 3,
  "successful_tests": 2,
  "failed_tests": 1,
  "results": [
    {
      "run_id": "test1",
      "testcase": "infer.InfiniLM.Direct",
      "success": 1,
      "duration": 123.45,
      "result_file": "/path/to/infer_InfiniLM_Direct_20250107_123000_results.json"
    },
    {
      "run_id": "test2",
      "testcase": "train.InfiniTrain.SFT",
      "success": 1,
      "duration": 234.56,
      "result_file": "/path/to/train_InfiniTrain_SFT_20250107_123005_results.json"
    },
    {
      "run_id": "test3",
      "testcase": "infer.InfiniLM.Batch",
      "success": 0,
      "duration": 10.0,
      "result_file": "/path/to/infer_InfiniLM_Batch_20250107_123008_results.json",
      "error": "Out of memory"
    }
  ],
  "timestamp": "2025-01-07T12:30:10.123456"
}
```

## 优势

### 1. 文件大小对比

**之前的设计**（数据嵌入）：
```json
{
  "total_tests": 100,
  "results": [
    {
      "testcase": "test1",
      "success": 1,
      "data": {...大量数据...},
      "metrics": [...大量指标...]
    },
    // ... 100 个测试的数据
  ]
}
```
❌ 问题：Summary 文件可能达到几百 MB 甚至 GB

**新的设计**（文件引用）：
```json
{
  "total_tests": 100,
  "results": [
    {
      "testcase": "test1",
      "success": 1,
      "result_file": "/path/to/test1.json"
    },
    // ... 只包含元信息
  ]
}
```
✅ 优点：Summary 文件只有几 KB，无论有多少测试

### 2. 使用便利性

**快速查看概览**：
```bash
# 只需要看 summary 文件
cat output/dispatcher_summary_*.json
# 能快速看到哪些测试成功/失败，耗时多少
```

**深入分析特定测试**：
```bash
# 根据 summary 中的 result_file 查看详细结果
cat output/infer_InfiniLM_Direct_20250107_123000_results.json
```

### 3. 数据管理

- ✅ 每个测试结果独立，便于单独删除/归档
- ✅ 避免数据重复（只在结果文件中存储一次）
- ✅ Summary 文件可以快速加载和解析
- ✅ 支持增量处理（不需要一次性加载所有数据）

## API 变化

### Executor.run() 返回值

**之前**：
```python
{
    'run_id': '...',
    'testcase': '...',
    'success': 1,
    'data': {...},          # 大量数据
    'metrics': [...],       # 所有指标
    'duration': 123.45
}
```

**现在**：
```python
{
    'run_id': '...',
    'testcase': '...',
    'success': 1,
    'duration': 123.45,
    'result_file': '/path/to/result.json',  # 文件引用
    'error': None
}
```

### Dispatcher 返回的 summary

**之前**：
```python
{
    'total_tests': 1,
    'results': [
        {
            'run_id': '...',
            'testcase': '...',
            'success': 1,
            'data': {...},        # 嵌入详细数据
            'metrics': [...],     # 嵌入所有指标
            'duration': 123.45
        }
    ]
}
```

**现在**：
```python
{
    'total_tests': 1,
    'results': [
        {
            'run_id': '...',
            'testcase': '...',
            'success': 1,
            'duration': 123.45,
            'result_file': '/path/to/result.json',  # 文件路径
            'error': None
        }
    ]
}
```

## 使用示例

### 1. 基本使用

```python
from dispatcher import Dispatcher

config = {'framework': 'infinilm', 'model_path': '/path/to/model', ...}
dispatcher = Dispatcher(config)

payload = {
    'run_id': 'test_001',
    'testcase': 'infer.InfiniLM.Direct',
    'config': config
}

result = dispatcher.dispatch(payload)

# result 是轻量级的 summary
print(f"Total tests: {result['total_tests']}")
print(f"Success: {result['successful_tests']}")
print(f"Failed: {result['failed_tests']}")

# 查看第一个测试的结果文件
first_test = result['results'][0]
print(f"Result file: {first_test['result_file']}")

# 如果需要详细数据，读取结果文件
import json
with open(first_test['result_file'], 'r') as f:
    detailed = json.load(f)
    print(f"Full data: {detailed['data']}")
    print(f"Metrics: {detailed['metrics']}")
```

### 2. 批量测试结果分析

```python
# 运行多个测试
results = []
for payload in payloads:
    result = dispatcher.dispatch(payload)
    results.append(result)

# 快速统计（不需要加载详细文件）
total_tests = sum(r['total_tests'] for r in results)
total_failed = sum(r['failed_tests'] for r in results)
total_duration = sum(
    sum(t['duration'] for t in r['results'])
    for r in results
)

print(f"Total: {total_tests}, Failed: {total_failed}, Duration: {total_duration:.2f}s")

# 只对失败的测试读取详细结果
for result in results:
    for test in result['results']:
        if test['success'] == 0:
            # 只读取失败测试的详细结果
            with open(test['result_file'], 'r') as f:
                detailed = json.load(f)
                print(f"Error in {test['testcase']}: {detailed.get('error')}")
```

## 迁移指南

如果你有旧代码依赖 Executor 返回的 `data` 或 `metrics` 字段：

### 旧代码
```python
result = executor.run()
data = result['data']  # ❌ 不再直接返回
metrics = result['metrics']  # ❌ 不再直接返回
```

### 新代码
```python
result = executor.run()
result_file = result['result_file']

import json
with open(result_file, 'r') as f:
    detailed = json.load(f)
    data = detailed['data']  # ✅ 从文件读取
    metrics = detailed['metrics']  # ✅ 从文件读取
```

或者简化为：
```python
result = executor.run()
detailed = json.load(open(result['result_file']))
data = detailed['data']
metrics = detailed['metrics']
```
