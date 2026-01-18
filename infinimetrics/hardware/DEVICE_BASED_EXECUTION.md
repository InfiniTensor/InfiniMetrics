# HardwareTestAdapter - Device-Based Execution

## 概述

`HardwareTestAdapter` 现在支持基于 **device 参数**的智能执行模式选择，自动决定是否编译 CUDA 项目和执行测试。

---

## 🎯 执行模式

### 决策逻辑

```
┌─────────────────────────────────────────────────────────────┐
│  1. mock_mode=True?                                        │
│     ├─ Yes → Mock Mode (always)                            │
│     └─ No  → Continue to step 2                            │
│                                                             │
│  2. Check device parameter:                                │
│     ├─ "cpu" or "mock" → Mock Mode                         │
│     ├─ "cuda" or "gpu" → Normal Mode                       │
│     └─ Not specified → Default "cuda" (Normal Mode)        │
└─────────────────────────────────────────────────────────────┘
```

### 执行模式对比

| Device | Mode | Setup (编译) | Process (执行) | 适用场景 |
|--------|------|--------------|----------------|----------|
| `cpu`, `CPU` | Mock | 跳过编译 | 读取 outputs/*.log | 无 GPU 开发 |
| `mock`, `MOCK` | Mock | 跳过编译 | 读取 outputs/*.log | 测试/CI |
| `cuda`, `CUDA` | Normal | 自动编译（如需要） | 执行 CUDA 测试 | 生产环境 |
| `gpu`, `GPU` | Normal | 自动编译（如需要） | 执行 CUDA 测试 | 生产环境 |
| 未指定 | Normal | 自动编译（如需要） | 执行 CUDA 测试 | 默认 |

---

## 💡 使用示例

### 1. CPU 设备（Mock 模式）

```python
adapter = HardwareTestAdapter(mock_mode=False)

test_input = {
    "testcase": "hardware.cudaunified.comprehensive",
    "run_id": "test_001",
    "config": {
        "device": "cpu",  # CPU 设备 → Mock 模式
        "test_type": "comprehensive",
        "output_dir": "./output"
    }
}

# 即使 mock_mode=False，device="cpu" 也会触发 Mock 模式
result = adapter.process(test_input)
```

### 2. CUDA 设备（正常模式）

```python
adapter = HardwareTestAdapter(mock_mode=False)

test_input = {
    "testcase": "hardware.cudaunified.comprehensive",
    "run_id": "test_002",
    "config": {
        "device": "cuda",  # CUDA 设备 → 正常模式
        "test_type": "comprehensive",
        "device_id": 0,
        "iterations": 10,
        "output_dir": "./output"
    }
}

# 会自动检查并编译 cuda_perf_suite（如需要）
# 然后执行真实的 CUDA 测试
result = adapter.process(test_input)
```

### 3. Mock 模式参数优先

```python
adapter = HardwareTestAdapter(mock_mode=True)  # Mock 模式

test_input = {
    "testcase": "hardware.cudaunified.comprehensive",
    "run_id": "test_003",
    "config": {
        "device": "cuda",  # 即使是 cuda 设备
        "output_dir": "./output"
    }
}

# mock_mode=True 优先级更高，强制使用 Mock 模式
result = adapter.process(test_input)
```

### 4. GPU 设备（正常模式）

```python
adapter = HardwareTestAdapter(mock_mode=False)

test_input = {
    "testcase": "hardware.cudaunified.bandwidth",
    "run_id": "test_004",
    "config": {
        "device": "gpu",  # GPU 设备 → 正常模式
        "test_type": "bandwidth",
        "output_dir": "./output"
    }
}

# 等同于 device="cuda"
result = adapter.process(test_input)
```

---

## 🔄 执行流程

### Mock 模式流程

```
adapter.setup(config)
  ├─ device in ["cpu", "mock"]?
  │   └─ Yes → 跳过 CUDA 编译 ✓
  ├─ mock_mode=True?
  │   └─ Yes → 跳过 CUDA 编译 ✓
  └─ skip_build=True?
      └─ Yes → 跳过 CUDA 编译 ✓

adapter.process(test_input)
  └─ 读取 outputs/ 目录中的 log 文件
      ├─ 解析性能指标
      ├─ 生成 CSV 文件
      └─ 生成 JSON 文件
```

### 正常模式流程

```
adapter.setup(config)
  ├─ device in ["cuda", "gpu"]?
  │   └─ Yes → 检查 cuda_perf_suite 是否存在
  │       ├─ 存在 → 跳过编译
  │       └─ 不存在 → 自动编译
  │           └─ 运行 build.sh
  │               └─ 验证可执行文件

adapter.process(test_input)
  └─ 执行真实的 CUDA 测试
      ├─ 构建命令
      ├─ 运行 cuda_perf_suite
      ├─ 解析输出
      └─ 保存结果
```

---

## 📝 配置参数

### 完整参数列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `device` | str | "cuda" | 设备类型：cpu, mock, cuda, gpu |
| `test_type` | str | "comprehensive" | 测试类型 |
| `device_id` | int | 0 | GPU 设备 ID |
| `iterations` | int | 10 | 测试迭代次数 |
| `output_dir` | str | "./output" | 输出目录 |
| `skip_build` | bool | False | 是否跳过编译 |

### 示例配置

```python
config = {
    # 设备相关
    "device": "cuda",           # 设备类型
    "device_id": 0,             # GPU ID（多 GPU 时使用）

    # 测试相关
    "test_type": "comprehensive", # 测试类型
    "iterations": 10,           # 迭代次数

    # 输出相关
    "output_dir": "./output",   # 输出目录

    # 编译相关
    "skip_build": False         # 跳过自动编译
}
```

---

## 🎬 实际场景

### 场景 1：开发环境（无 GPU）

```python
# 方式 1: 使用 device="cpu"
adapter = HardwareTestAdapter(mock_mode=False)
config = {"device": "cpu", "test_type": "comprehensive"}

# 方式 2: 使用 mock_mode=True
adapter = HardwareTestAdapter(mock_mode=True)
config = {"test_type": "comprehensive"}

# 两者都会触发 Mock 模式，无需编译，读取 outputs/*.log
```

### 场景 2：生产环境（有 GPU）

```python
# 首次运行：会自动编译
adapter = HardwareTestAdapter(mock_mode=False)
config = {
    "device": "cuda",
    "test_type": "comprehensive",
    "device_id": 0
}

# setup() 会：
# 1. 检查 cuda_perf_suite 是否存在
# 2. 不存在 → 自动编译
# 3. process() 执行真实测试

# 后续运行：跳过编译，直接执行
# adapter = HardwareTestAdapter(mock_mode=False)
# setup() 检测到 cuda_perf_suite 已存在，跳过编译
```

### 场景 3：CI/CD 环境

```python
# CI 中使用预编译的二进制文件 + Mock 数据
adapter = HardwareTestAdapter(
    mock_mode=False,
    cuda_perf_path="/opt/cuda_perf_suite"  # 预编译路径
)

config = {
    "device": "mock",          # 使用 mock 模式
    "skip_build": True,        # 跳过编译检查
    "test_type": "comprehensive",
    "input_outputs_dir": "/test/fixtures/outputs"
}
```

---

## ⚙️ 高级选项

### 优先级顺序

```
1. mock_mode=True (最高优先级)
   ↓ 如果为 False
2. device="cpu" 或 device="mock"
   ↓ 如果不是
3. device="cuda" 或 device="gpu"
```

### 自定义编译选项

```python
# 跳过自动编译（使用预编译版本）
adapter = HardwareTestAdapter(
    cuda_perf_path="/custom/path/to/cuda_perf_suite"
)

config = {
    "device": "cuda",
    "skip_build": True  # 跳过编译检查
}
```

---

## 🧪 测试

运行测试验证功能：

```bash
# 测试 device 参数
python test_device_based_execution.py

# 测试 setup 方法
python test_hardware_setup.py

# 测试 mock 模式
python test_hardware_adapter_outputs.py
```

---

## 🐛 故障排查

### 问题 1: 编译失败

**症状：**
```
RuntimeError: Failed to build CUDA project: nvcc not found
```

**解决方案：**
1. 使用 `device="cpu"` 或 `device="mock"`
2. 或安装 CUDA Toolkit
3. 或使用预编译的二进制文件

### 问题 2: 设备被忽略

**症状：**
设置了 `device="cpu"` 但仍然尝试编译

**检查：**
```python
# 确认 device 参数传递正确
config = {"device": "cpu"}
adapter.setup(config)  # device 需要在 config 中

# 或者直接在 test_input 中
test_input = {
    "config": {"device": "cpu"}
}
```

### 问题 3: 找不到输出文件

**症状：**
```
FileNotFoundError: Input outputs directory not found
```

**解决方案：**
```python
adapter = HardwareTestAdapter(
    input_outputs_dir="./path/to/outputs"  # 指定正确的路径
)
```

---

## 📚 相关文档

- [USAGE.md](USAGE.md) - 完整使用说明
- [SETUP_IMPLEMENTATION.md](SETUP_IMPLEMENTATION.md) - 实现细节
- [test_device_based_execution.py](../../test_device_based_execution.py) - 测试脚本

---

## 🎯 关键改进

1. ✅ **智能判断** - 根据 device 自动选择执行模式
2. ✅ **灵活配置** - 支持多种设备类型
3. ✅ **向后兼容** - 默认行为不变
4. ✅ **优先级清晰** - mock_mode > device > 默认
5. ✅ **开发友好** - CPU/Mock 设备无需 CUDA 环境

---

**最后更新**: 2026-01-19
**版本**: 2.0 (Device-Based)
**作者**: Claude Code Assistant
