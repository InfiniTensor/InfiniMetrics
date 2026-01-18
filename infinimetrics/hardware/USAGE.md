# HardwareTestAdapter 使用说明

## 概述

`HardwareTestAdapter` 是 InfiniMetrics 中用于 CUDA 硬件性能测试的适配器，支持两种工作模式：

### 1. 正常模式 (Normal Mode) - 需要显卡
- 执行真实的 CUDA 基准测试
- 需要 NVIDIA GPU 和 CUDA 环境
- 运行 `cuda_perf_suite` 程序
- **自动编译**：如果 `cuda_perf_suite` 不存在，会自动编译

### 2. Mock 模式 (Mock Mode) - 无需显卡
- 从已有的测试输出文件中读取数据
- 适用于无 GPU 的开发/测试环境
- 读取 `outputs/` 目录中的 log 文件
- **跳过编译**：Mock 模式下不会尝试编译 CUDA 项目

---

## 生命周期方法

`HardwareTestAdapter` 遵循标准适配器生命周期：

```python
adapter = HardwareTestAdapter(...)
adapter.setup(config)      # 初始化资源（编译 CUDA 项目）
adapter.process(test_input) # 执行测试
adapter.teardown()         # 清理资源
```

### `setup(config)` 方法

`setup()` 方法会在测试前自动调用（由 Executor），负责：

**正常模式：**
1. 检查 `cuda_perf_suite` 可执行文件是否存在
2. 如果不存在，自动编译 CUDA 项目
3. 验证编译成功

**Mock 模式：**
- 跳过所有编译步骤

**配置选项：**
```python
config = {
    "skip_build": False,  # True=跳过编译, False=自动编译（默认）
    "output_dir": "./output"
}
```

---

## 使用方法

### Mock 模式（无 GPU）

```python
from infinimetrics.hardware.hardware_adapter import HardwareTestAdapter

# 创建适配器（启用 mock 模式）
adapter = HardwareTestAdapter(
    output_dir="./output",
    mock_mode=True,  # 启用 mock 模式
    input_outputs_dir="./infinimetrics/hardware/outputs"  # 指定输入目录
)

# 准备测试输入
test_input = {
    "testcase": "hardware.cudaunified.comprehensive",
    "run_id": "test_001",
    "config": {
        "test_type": "comprehensive",  # bandwidth, stream, cache, comprehensive
        "output_dir": "./output"
    }
}

# 执行测试
result = adapter.process(test_input)

# 查看结果
print(f"Result code: {result['result_code']}")
print(f"Metrics extracted: {len(result['metrics'])}")

for metric in result['metrics']:
    print(f"  - {metric['name']}: {metric['value']} {metric['unit']}")

# 获取转换后的文件路径
converted_files = result.get('converted_files', {})
print(f"Bandwidth CSV: {converted_files.get('bandwidth_csv')}")
print(f"Stream CSV: {converted_files.get('stream_csv')}")
print(f"Metrics JSON: {converted_files.get('metrics_json')}")
```

### 正常模式（有 GPU）

```python
from infinimetrics.hardware.hardware_adapter import HardwareTestAdapter

# 创建适配器（默认为正常模式）
adapter = HardwareTestAdapter(
    output_dir="./output",
    mock_mode=False  # 正常模式，执行真实测试
)

# 准备测试输入
test_input = {
    "testcase": "hardware.cudaunified.comprehensive",
    "run_id": "test_001",
    "config": {
        "test_type": "comprehensive",
        "device_id": 0,          # GPU 设备 ID
        "iterations": 10,        # 测试迭代次数
        "output_dir": "./output"
    }
}

# 执行测试（会运行真实的 CUDA benchmark）
result = adapter.process(test_input)
```

---

## 支持的测试类型

| test_type | 说明 | Log 文件 | 指标 |
|-----------|------|----------|------|
| `bandwidth` | 内存带宽测试 | `bandwidth_test.log` | Host↔Device, Device↔Device 带宽 |
| `stream` | STREAM 基准测试 | `stream_test.log` | Copy, Scale, Add, Triad 带宽 |
| `cache` | 缓存带宽测试 | `cache_test.log` | L1, L2 缓存带宽 |
| `comprehensive` | 综合测试 | `comprehensive_test.log` | 以上所有指标 |

---

## 输出文件

Mock 模式会生成以下文件：

### 1. CSV 数据文件
- `memory_bandwidth_{run_id}_{timestamp}.csv` - 内存带宽数据
- `stream_{run_id}_{timestamp}.csv` - STREAM 测试数据
- `cache_bandwidth_{run_id}_{timestamp}.csv` - 缓存带宽数据

### 2. JSON 指标汇总
- `{testcase}_{run_id}_{timestamp}_metrics.json` - 所有指标的 JSON 格式

### 3. 原始日志副本
- `{original_log}_{run_id}_{timestamp}.log` - 复制的原始 log 文件

---

## 文件结构示例

```
infinimetrics/hardware/
├── outputs/                           # Mock 模式的输入目录
│   ├── bandwidth_test.log            # 带宽测试输出
│   ├── stream_test.log               # STREAM 测试输出
│   ├── cache_test.log                # 缓存测试输出
│   └── comprehensive_test.log        # 综合测试输出
│
├── cuda-memory-benchmark/            # CUDA 测试程序
│   ├── build/
│   │   └── cuda_perf_suite           # 可执行文件
│   └── build.sh
│
└── hardware_adapter.py               # 适配器实现
```

---

## 环境变量支持

可以通过环境变量配置 mock 模式：

```bash
# 启用 mock 模式
export HARDWARE_MOCK_MODE=true

# 指定 mock 输出文件（可选）
export HARDWARE_MOCK_OUTPUT=/path/to/output.log
```

---

## 错误处理

### 常见错误

1. **`FileNotFoundError: Input outputs directory not found`**
   - 确保 `outputs/` 目录存在
   - 检查 `input_outputs_dir` 参数是否正确

2. **`Log file not found`**
   - 检查对应的 log 文件是否存在
   - 参考 `_get_log_files_for_test_type()` 中的文件映射

3. **`No metrics extracted`**
   - 检查 log 文件格式是否正确
   - 查看日志了解解析详情

---

## 示例输出

```json
{
  "result_code": 0,
  "time": "2026-01-19 00:33:14",
  "metrics": [
    {"name": "memory.bandwidth.host_to_device", "value": 367.2, "unit": "GB/s"},
    {"name": "memory.bandwidth.device_to_host", "value": 367.2, "unit": "GB/s"},
    {"name": "memory.bandwidth.device_to_device", "value": 367.2, "unit": "GB/s"},
    {"name": "stream.bandwidth.copy", "value": 437.36, "unit": "GB/s"},
    {"name": "stream.bandwidth.scale", "value": 446.41, "unit": "GB/s"},
    {"name": "stream.bandwidth.add", "value": 394.01, "unit": "GB/s"},
    {"name": "stream.bandwidth.triad", "value": 417.26, "unit": "GB/s"}
  ],
  "converted_files": {
    "bandwidth_csv": "test_output/memory_bandwidth_test_001_20260119_003314.csv",
    "stream_csv": "test_output/stream_test_001_20260119_003314.csv",
    "comprehensive_test_log": "test_output/comprehensive_test_test_001_20260119_003314.log",
    "metrics_json": "test_output/hardware_cudaunified_comprehensive_test_001_20260119_003314_metrics.json"
  }
}
```

---

## 测试脚本

运行测试脚本验证功能：

```bash
# 测试 mock 模式
python test_hardware_adapter_outputs.py
```

这将：
1. 读取 `outputs/` 目录中的所有 log 文件
2. 解析并提取性能指标
3. 生成 CSV 和 JSON 文件
4. 显示测试结果

---

## 注意事项

1. **Mock 模式仅用于开发/调试**
   - 不会执行真实的 GPU 测试
   - 数据来自预先保存的 log 文件

2. **正常模式需要完整的 CUDA 环境**
   - NVIDIA GPU 驱动
   - CUDA Toolkit
   - 已编译的 `cuda_perf_suite`

3. **输出文件会覆盖同名文件**
   - 使用时间戳避免冲突
   - 注意备份重要数据
