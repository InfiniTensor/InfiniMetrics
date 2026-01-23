# InfiniMetrics

<div align="center">

**面向 InfiniTensor 的全面加速器评估框架**


</div>

---

## 🎯 项目概述

**InfiniMetrics** 是一个统一、模块化的测试框架，专为加速卡和软件栈的全面性能评估而设计。它提供了标准化的接口，用于在多个层次进行基准测试：

- **硬件层**：GPU 内存带宽、缓存性能、计算能力
- **算子层**：单个操作的性能（FLOPS、延迟）
- **推理层**：端到端模型推理吞吐量和延迟
- **通信层**：NCCL 集合操作和 GPU 间通信

### 核心特性

✨ **统一适配器接口** - 所有测试类型和框架的一致 API
🔧 **可扩展架构** - 易于添加新的测试类型、框架和指标
📊 **全面的指标系统** - 标量值、时间序列数据、自定义测量
🎛️ **框架无关** - 支持 InfiniLM、vLLM、InfiniCore 等
🚀 **生产就绪** - 健壮的错误处理、日志记录和结果聚合

---

## 📋 目录

- [快速开始](#快速开始)
- [安装指南](#安装指南)
- [项目架构](#项目架构)
- [支持的测试类型](#支持的测试类型)
- [配置指南](#配置指南)
- [输出和结果](#输出和结果)
- [使用示例](#使用示例)
- [开发指南](#开发指南)

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/InfiniTensor/InfiniMetrics.git
cd InfiniMetrics
git submodule update --init --recursive
```

### 2. 运行硬件基准测试

```bash
# 运行全面的硬件测试
python main.py format_input_comprehensive_hardware.json
```

---

## 💾 安装指南

### 前置要求

- **Python**：3.8 或更高版本
- **编译器**：GCC 11.3
- **CMake**：3.20+

### 依赖安装

```bash
# 核心 Python 依赖（按需安装）
pip install numpy torch  # 用于 InfiniLM 适配器
pip install vllm        # 用于 vLLM 适配器
pip install pandas       # 用于数据处理
```

### 编译硬件基准测试（可选）

如需使用硬件测试模块：

```bash
cd infinimetrics/hardware/cuda-memory-benchmark
bash build.sh
```

---

## 🏗️ 项目架构

```
InfiniMetrics/
├── main.py                          # 主入口文件
├── infinimetrics/
│   ├── adapter.py                  # 基础适配器接口
│   ├── dispatcher.py               # 测试编排调度器
│   ├── executor.py                 # 通用测试执行器
│   ├── input.py                    # 测试输入数据类
│   │
│   ├── common/                     # 共享工具
│   │   ├── constants.py            # 测试类型、指标、枚举
│   │   ├── metrics.py              # 指标定义
│   │   └── testcase_utils.py       # 测试用例工具
│   │
│   ├── hardware/                   # 硬件测试模块
│   │   └── cuda-memory-benchmark/  # CUDA 内存基准测试套件
│   │       ├── include/            # C++ 头文件
│   │       ├── src/                # CUDA/C++ 源代码
│   │       ├── CMakeLists.txt      # 构建配置
│   │       ├── build.sh            # 构建脚本
│   │       └── QUICKSTART.md       # 硬件测试快速开始指南
│   │
│   ├── operators/                  # 算子级测试
│   │   ├── infinicore_adapter.py   # InfiniCore 操作
│   │   └── flops_calculator.py     # FLOPS 计算器
│   │
│   ├── inference/                  # 推理评估
│   │   ├── adapters/
│   │   │   ├── infinilm_adapter.py # InfiniLM 适配器
│   │   │   └── vllm_adapter.py     # vLLM 适配器
│   │   ├── infer_main.py           # 推理入口点
│   │   └── utils/                  # 推理工具
│   │
│   ├── communication/              # 通信测试
│   │   └── nccl_adapter.py         # NCCL 适配器
│   │
│   └── utils/                      # 工具函数
│       └── input_loader.py         # 输入文件加载器
│
├── submodules/
    └── nccl-tests/                 # NCCL 测试套件（git 子模块）

```

### 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                          InfiniMetrics                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────┐        │
│  │  输入    │───▶│  调度器    │───▶│    执行器        │    │
│  │  文件    │    │             │    │                  │    │
│  └──────────┘    └─────────────┘    └────────┬─────────┘    │
│                                              │                   │
│                                    ┌─────────▼─────────┐    │
│                                    │                   │    │
│                       ┌──────────────┴──────────────┐    │
│                       │                              │    │
│  ┌────────────────────▼──────────────────────┐    │
│  │          适配器注册表                  │    │
│  │  ┌─────────┐  ┌──────────┐  ┌──────────┐  │    │
│  │  │ 硬件    │  │  算子     │  │  推理    │  │    │
│  │  │ 测试    │  │  测试     │  │  测试    │  │    │
│  │  └────┬────┘  └────┬─────┘  └────┬─────┘  │    │
│  └───────┼────────────┼─────────────┼──────────┘    │
│          │            │             │                   │
│  ┌───────▼───────────▼─────┬───────▼───────┐    │
│  │           具体适配器                   │    │
│  │  ┌─────────┐  ┌──────────┐  ┌──────┐  │    │
│  │  │ CUDA    │  │InfiniCore│  │vLLM  │  │    │
│  │  │ 内存    │  │  适配器  │  │适配器│  │    │
│  │  │ 基准    │  │          │  │      │  │    │
│  │  └─────────┘  └──────────┘  └──────┘  │    │
│  └────────────────────────────────────┘    │
│                                           │    │
│  ┌────────────────────────────────────┐    │
│  │         指标系统                    │    │
│  │  • 标量指标                        │    │
│  │  • 时间序列指标                    │    │
│  │  • 自定义指标定义                   │    │
│  └────────────────────────────────────┘    │
│                                           │    │
└───────────────────────────────────────────┘
```

---

## 🧪 支持的测试类型

### 1. 硬件测试 (`hardware.*`)

#### CUDA 内存基准测试

| 测试名称 | 描述 | 指标 |
|-----------|------|------|
| `hardware.mem_sweep_h2d` | 主机到设备扫描测试（64KB-1GB） | 带宽 (GB/s)、时间 (ms) |
| `hardware.mem_sweep_d2h` | 设备到主机扫描测试 | 带宽 (GB/s)、时间 (ms) |
| `hardware.mem_sweep_d2d` | 设备到设备扫描测试 | 带宽 (GB/s)、时间 (ms) |
| `hardware.mem_bw_h2d` | H2D 带宽测试（固定大小） | 带宽 (GB/s) |
| `hardware.mem_bw_d2h` | D2H 带宽测试（固定大小） | 带宽 (GB/s) |
| `hardware.mem_bw_d2d` | D2D 带宽测试（固定大小） | 带宽 (GB/s) |

#### STREAM 基准测试

| 测试名称 | 描述 | 字节数/元素 |
|-----------|------|-----------|
| `hardware.stream_copy` | 复制操作 | 2 |
| `hardware.stream_scale` | 缩放操作 | 2 |
| `hardware.stream_add` | 加法操作 | 3 |
| `hardware.stream_triad` | 三元组操作 | 3 |

#### GPU 缓存测试

| 测试名称 | 描述 |
|-----------|------|
| `hardware.gpu_cache_l1` | L1 缓存带宽 |
| `hardware.gpu_cache_l2` | L2 缓存带宽 |

### 2. 算子测试 (`operator.*`)

| 测试名称 | 框架 | 描述 |
|-----------|------|------|
| `operator.infinicore.*` | InfiniCore | 单个算子性能测试、FLOPS 计算 |

### 3. 推理测试 (`infer.*`)

| 测试名称 | 框架 | 指标 |
|-----------|------|------|
| `infer.infinilm.direct` | InfiniLM | 吞吐量 (tokens/s)、延迟 (ms)、内存使用 |
| `infer.infinilm.prefill` | InfiniLM | 预填充阶段指标 |
| `infer.vllm.*` | vLLM | 各种 vLLM 推理模式 |

### 4. 通信测试 (`comm.*`)

| 测试名称 | 框架 | 描述 |
|-----------|------|------|
| `comm.nccltest.*` | NCCL | NCCL 集合通信操作基准测试 |

---

## ⚙️ 配置指南

### 输入文件格式

测试规格以 JSON 格式提供：

```json
{
    "run_id": "唯一运行标识符",
    "testcase": "hardware.cudaUnified.Comprehensive",
    "config": {
        "device": "nvidia",
        "array_size": 67108864,
        "buffer_size_mb": 256,
        "output_dir": "./output"
    },
    "metrics": [
        {"name": "hardware.mem_bw_h2d"},
        {"name": "hardware.mem_bw_d2h"},
        {"name": "hardware.stream_triad"}
    ]
}
```

### 配置参数说明

| 参数 | 类型 | 描述 | 默认值 |
|------|------|------|--------|
| `run_id` | string | 唯一的测试运行标识符 | 必填 |
| `testcase` | string | 测试类型标识符 | 必填 |
| `config.device` | string | 加速器类型 (nvidia/amd/huawei/cambricon) | nvidia |
| `config.array_size` | int | STREAM 测试的数组大小 | 67108864 |
| `config.buffer_size_mb` | int | 内存测试的缓冲区大小 (MB) | 256 |
| `config.output_dir` | string | 输出目录路径 | ./output |

### 测试用例命名约定

格式：`<类别>.<框架>.<测试名称>`

- **类别**: `hardware`, `operator`, `infer`, `comm`
- **框架**: `cudaUnified`, `infinicore`, `infinilm`, `vllm`, `nccltest`
- **示例**:
  - `hardware.cudaUnified.Comprehensive`
  - `operator.infinicore.Conv2D`
  - `infer.infinilm.direct`
  - `comm.nccltest.AllReduce`

---

## 📊 输出和结果

### 输出目录结构

```
./summary_output/
├── dispatcher_summary_YYYYMMDD_HHMMSS.json    # 总体测试摘要
└── ./output/
    ├── hardware.cudaUnified.Comprehensive/
    │   ├── metrics.json                            # 测试指标
    │   ├── trace.json                              # 执行跟踪（如有）
    │   └── log.txt                                 # 详细日志
    └── ...
```

### 摽要格式

```json
{
  "total_tests": 3,
  "successful_tests": 2,
  "failed_tests": 1,
  "results": [
    {
      "run_id": "test_run_001",
      "testcase": "hardware.cudaUnified.Comprehensive",
      "result_code": 0,
      "result_file": "./output/hardware.cudaUnified.Comprehensive/metrics.json"
    }
  ],
  "timestamp": "2026-01-22T10:56:09.338373"
}
```

### 输出示例

#### 硬件基准测试输出

```
===================================================
内存拷贝带宽扫描测试
方向：主机到设备
内存类型：固定
===================================================

大小 (MB)      时间 (ms)  带宽 (GB/s)      CV (%)
------------------------------------------------------
     0.06            0.12             25.60        1.20
     0.12            0.24             25.80        0.90
     0.25            0.48             26.10        0.85
   256.00           98.45             26.20        1.10
   512.00          195.12             26.50        0.95
```

#### 推理基准测试输出

```json
{
  "throughput_tokens_per_sec": 1250.5,
  "latency_ms": 8.5,
  "memory_usage_mb": 2048,
  "model_name": "infinilm-7b",
  "batch_size": 32
}
```

---

## 📚 使用示例

### 示例 1：全面硬件基准测试

```bash
python main.py format_input_comprehensive_hardware.json
```

**测试内容：**
- 内存带宽 (H2D、D2H、D2D) 多种缓冲区大小
- STREAM 基准测试 (Copy、Scale、Add、Triad 操作)
- GPU 缓存性能 (L1、L2)

### 示例 2：多个测试配置

```bash
# 运行目录中的所有 JSON 配置
python main.py ./test_configs/
```

### 示例 3：推理评估

```bash
cd infinimetrics/inference
python infer_main.py --config config.json --model infinilm-7b
```

### 示例 4：详细输出

```bash
python main.py input.json --verbose --output ./results
```

---

## 🔧 开发指南

### 添加新的适配器

1. **创建适配器类**，继承 `BaseAdapter`：

```python
from infinimetrics.adapter import BaseAdapter

class MyCustomAdapter(BaseAdapter):
    def __init__(self, config):
        super().__init__(config)
        # 初始化你的适配器

    def setup(self):
        # 准备测试环境
        pass

    def process(self, test_input):
        # 执行测试并返回指标
        return {"my_metric": 42.0}

    def teardown(self):
        # 清理资源
        pass
```

2. **在 Dispatcher 中注册适配器**：

```python
# 在 dispatcher.py 中
self.adapter_registry = {
    ("operator", "myframework"): MyCustomAdapter,
    # ... 现有适配器 ...
}
```

3. **定义测试用例和指标**：

```json
{
    "run_id": "my_test",
    "testcase": "operator.myframework.MyTest",
    "config": {...},
    "metrics": [
        {"name": "my.metric"}
    ]
}
```

### 添加新指标

在 `infinimetrics/common/metrics.py` 中定义指标：

```python
class CustomMetric(Metric):
    def __init__(self, name: str, value: float, unit: str = ""):
        super().__init__(name, value, unit)
```

## 🤝 贡献

欢迎贡献！请随时提交 Pull Request。

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

---

## 📧 联系我们

如有问题或建议，请在 GitHub 上提交 issue 或联系 InfiniTensor 团队。

---

<div align="center">

**由 InfiniTensor 团队构建 ❤️**

[官网](https://infinitensor.org) | [文档](https://docs.infinitensor.org) | [GitHub](https://github.com/InfiniTensor)

</div>
