# InfiniMetrics 统一测试系统架构设计文档

**版本**: v1.0
**日期**: 2025-01-05
**作者**: InfiniTensor Team

---

## 1. 系统概述

### 1.1 背景与动机

InfiniMetrics 当前面临的主要挑战：

1. **架构不统一** - 算子测试、推理测试、训练测试各自独立，缺乏统一入口
2. **代码分散** - 测试脚本散落在各处，难以维护和复用
3. **扩展困难** - 添加新的测试类型或框架需要修改多处代码
4. **职责模糊** - Runner 和 Adapter 的边界不清晰，功能重叠

### 1.2 设计目标

构建一个**统一、简洁、可扩展**的深度学习测试框架：

- **统一入口** - 所有测试通过同一个调度器进行
- **清晰分层** - 每层有明确的职责，易于理解和维护
- **易于扩展** - 添加新功能无需修改核心代码
- **代码复用** - 通用逻辑在基类中实现，避免重复

### 1.3 适用场景

系统支持三种核心测试模式：

| 测试模式 | 典型场景 | 特点 |
|---------|---------|------|
| **算子测试** | 测试单个算子的性能和正确性 | 单次调用、无状态、快速反馈 |
| **推理测试** | 测试端到端模型推理性能 | 需要模型加载、多轮迭代、性能监控 |
| **训练测试** | 测试模型训练过程 | 长时间运行、监控loss曲线、checkpoint管理（未来） |

### 1.4 设计原则

我们遵循以下核心原则：

1. **简单 > 复杂** - 优先考虑易用性和可理解性
2. **清晰 > 巧妙** - 代码应该自解释，避免过度设计
3. **实用 > 完美** - 解决实际问题，不过度优化
4. **渐进 > 大爆炸** - 分阶段实施，逐步完善

---

## 2. 整体架构

### 2.1 架构分层

```
┌──────────────────────────────────────────────────────────────┐
│                         用户层                               │
│  CLI / Python API / REST API / Config File                  │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ↓
┌──────────────────────────────────────────────────────────────┐
│                      Dispatcher (调度层)                      │
│  • 接收测试请求                                             │
│  • 路由到合适的 Runner                                       │
│  • 统一错误处理                                             │
│  • 返回标准化结果                                           │
└────────────────────────────┬───────────────────────────────┘
                             │
                ┌────────────┼────────────┐
                ↓            ↓            ↓
        ┌───────────┐  ┌──────────┐  ┌──────────┐
        │  Operator │  │ Direct   │  │ Service  │
        │  Runner   │  │ Infer.   │  │ Infer.   │
        │           │  │ Runner   │  │ Runner   │
        └─────┬─────┘  └────┬─────┘  └────┬─────┘
              │              │              │
              └──────────────┼──────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────┐
│                    Runner (执行层)                            │
│  • 生命周期管理 (setup → execute → collect → teardown)       │
│  • 测试流程编排                                             │
│  • 结果聚合与统计                                           │
│  • 数据保存                                                 │
└────────────────────────────┬───────────────────────────────┘
                             │
                ┌────────────┼────────────┐
                ↓            ↓            ↓
        ┌───────────┐  ┌──────────┐  ┌──────────┐
        │   Prompt  │  │  Trace   │  │ Dataset  │
        │  Loader   │  │  Loader  │  │  Loader  │
        │           │  │          │  │          │
        └─────┬─────┘  └────┬─────┘  └────┬─────┘
              │              │              │
              └──────────────┼──────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────┐
│                    DataLoader (数据层)                        │
│  • 测试数据准备                                             │
│  • 支持多种数据源                                           │
│  • Batch 管理                                               │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ↓
┌──────────────────────────────────────────────────────────────┐
│                    Adapter (接口层)                          │
│  • 外部框架接口封装                                         │
│  • 模型生命周期管理                                         │
│  • 结果格式转换                                             │
└────────────────────────────┬───────────────────────────────┘
                             │
                ┌────────────┼────────────┐
                ↓            ↓            ↓
        ┌───────────┐  ┌──────────┐  ┌──────────┐
        │InfiniCore │  │ InfiniLM │  │   vLLM   │
        │  Adapter  │  │ Adapter  │  │ Adapter  │
        └───────────┘  └──────────┘  └──────────┘
                             │
                             ↓
                        ┌──────────┐
                        │  框架层   │
                        └──────────┘
```

### 2.2 职责划分

#### 2.2.1 Dispatcher - 统一调度器

**核心职责**：系统入口和路由

- **请求解析** - 解析不同格式的测试请求（JSON、YAML、CLI参数）
- **类型路由** - 根据测试类型（operator/inference/training）创建对应的 Runner
- **生命周期管理** - 调用 Runner 的完整生命周期（setup → execute → collect → teardown）
- **错误处理** - 统一的异常捕获和错误返回格式
- **结果封装** - 将不同 Runner 的结果封装成统一格式返回给用户

**不负责**：
- 具体测试逻辑（由 Runner 负责）
- 数据准备（由 DataLoader 负责）
- 框架调用（由 Adapter 负责）

**关键设计决策**：
- 使用工厂模式，根据 `test_type` 动态创建 Runner
- 支持 Runner 注册，便于扩展新的测试类型
- 提供简单的同步接口，隐藏异步复杂性（如 ServiceInferenceRunner）

#### 2.2.2 Runner - 测试执行器

**核心职责**：测试流程编排和结果管理

**生命周期方法**（所有 Runner 必须实现）：

1. **setup()** - 准备测试环境
   - 创建监控器（AcceleratorMonitor）
   - 加载模型（通过 adapter.setup()）
   - 准备测试数据（通过 data_loader.load()）
   - 启动服务（对于 ServiceInferenceRunner）

2. **execute()** - 执行测试
   - **OperatorRunner**：单次调用 adapter.process()
   - **DirectInferenceRunner**：Warmup 阶段 + Measurement 阶段，多次调用
   - **ServiceInferenceRunner**：异步执行 trace 测试
   - **TrainingRunner**：训练循环（未来）

3. **collect_metrics()** - 收集和计算指标
   - 停止监控，获取峰值内存
   - 调用 calculate_statistics() 计算统计数据
   - 调用 _add_special_metrics() 添加特殊指标（perplexity、accuracy等）
   - 返回标准化的指标字典

4. **teardown()** - 清理环境
   - 停止监控
   - 卸载模型（通过 adapter.teardown()）
   - 停止服务（对于 ServiceInferenceRunner）

**提供的公共方法**（基类实现，子类继承）：

- **prepare_output_dir()** - 创建输出目录
- **save_timeseries_data()** - 保存时间序列数据到 CSV 文件
- **calculate_statistics()** - 计算 avg/p95/p99 等统计值
- **dump_json()** - 导出 JSON 格式的结果文件

**可选钩子方法**（子类可重写）：

- **_add_special_metrics()** - 添加测试类型特有的指标
- **_calculate_perplexity()** - 计算困惑度（DirectInferenceRunner）
- **_load_test_data()** - 加载测试数据集（用于计算指标）

**关键设计决策**：
- 使用模板方法模式，定义标准流程，子类实现具体步骤
- 提供丰富的工具方法，减少子类重复代码
- 通过钩子方法提供扩展点，避免修改基类

#### 2.2.3 DataLoader - 数据加载器

**核心职责**：准备测试输入数据

**不同类型的 Loader**：

1. **PromptLoader** - 为推理测试准备 prompts
   - 支持从文件加载（JSON/JSONL/TXT）
   - 支持模板生成（预定义模板 + 主题）
   - 支持随机生成（随机词汇组合）
   - 支持固定 prompt（重复使用单个 prompt）
   - 支持 tokenizer 约束（控制 prompt 长度）

2. **TraceLoader** - 为服务测试加载 trace 文件
   - 加载 CSV 格式的 trace 文件
   - 验证 trace 数据（时间戳顺序、token数限制）
   - 转换为内部 RequestTrace 对象

3. **DatasetLoader** - 加载测试数据集
   - 用于计算 perplexity、accuracy 等指标
   - 支持多种格式（JSON/JSONL）
   - 限制数量，避免计算时间过长

**公共接口**：

- **load(count)** - 加载指定数量的数据项
- **create_batches(data, batch_size)** - 将数据分批

**关键设计决策**：
- 使用策略模式，支持多种数据源
- 支持降级策略（主要方法失败时使用 fallback）
- 可配置化，通过配置文件控制加载行为

#### 2.2.4 Adapter - 框架适配器

**核心职责**：封装外部框架的接口

**统一接口**：

1. **process(request) -> response** (必须实现)
   - 无状态 adapter（如 InfiniCoreAdapter）：单次调用返回结果
   - 有状态 adapter（如 InfiniLMAdapter）：在 setup/teardown 之间多次调用

2. **setup(config)** (可选实现)
   - 有状态 adapter 实现：加载模型、建立连接等
   - 无状态 adapter：不需要实现或使用默认实现

3. **teardown()** (可选实现)
   - 有状态 adapter 实现：卸载模型、关闭连接等
   - 无状态 adapter：不需要实现或使用默认实现

4. **validate()** (可选实现)
   - 验证配置是否正确
   - 返回错误列表，空列表表示验证通过

**两种模式**：

**模式一：无状态适配器**（如 InfiniCoreAdapter）

```
适用场景：算子测试
特点：
  - 不需要 setup/teardown
  - 每次调用独立
  - 输入包含完整测试信息

示例：
  adapter = InfiniCoreAdapter()
  result = adapter.process({
      "operator": "add",
      "inputs": [...],
      "metrics": ["latency", "accuracy"]
  })
```

**模式二：有状态适配器**（如 InfiniLMAdapter）

```
适用场景：推理测试
特点：
  - 需要 setup 加载模型
  - 可多次调用 process
  - 需要 teardown 清理

示例：
  adapter = InfiniLMAdapter(config)
  adapter.setup()  # 加载模型

  # 多次推理
  for batch in batches:
      result = adapter.process({"prompts": batch})

  adapter.teardown()  # 卸载模型
```

**关键设计决策**：
- 单一基类支持有状态和无状态两种模式
- 通过可选方法（setup/teardown）区分，不需要继承层次
- process 方法返回统一格式，便于 Runner 处理

---

## 3. 三种测试模式详解

### 3.1 算子测试模式

#### 3.1.1 应用场景

测试单个算子（如 matmul, conv2d, attention）的性能和正确性。

典型问题：
- 这个算子在不同输入规模下的延迟如何？
- 算子结果是否正确（与参考实现对比）？
- 算子的计算吞吐量（TFLOPS）和带宽利用率如何？

#### 3.1.2 测试流程

```
┌──────────────┐
│ 用户提供配置  │
│  - 算子类型   │
│  - 输入tensor │
│  - 输出tensor │
│  - 设备类型   │
└──────┬───────┘
       │
       ↓
┌──────────────┐
│Dispatcher    │
│→ 创建 OperatorRunner
└──────┬───────┘
       │
       ↓
┌──────────────┐
│OperatorRunner│
│ setup()      │ → 验证配置
│ execute()    │ → 调用 adapter.process()
│ collect()    │ → 提取结果
└──────┬───────┘
       │
       ↓
┌──────────────┐
│InfiniCore    │
│Adapter       │
│ process()    │ → 执行算子
│              │ → 计算指标
└──────┬───────┘
       │
       ↓
┌──────────────┐
│  返回结果    │
│  - latency   │
│  - accuracy  │
│  - tflops    │
└──────────────┘
```

#### 3.1.3 特点

- **轻量级**：不需要模型加载，单次调用
- **快速反馈**：测试时间通常在毫秒到秒级别
- **自包含**：输入数据在配置中，不需要额外的数据文件
- **同步执行**：调用即返回，无需异步

#### 3.1.4 适用情况

✅ **适合**：
- 算子性能优化验证
- 算子正确性回归测试
- 不同设备（CPU/GPU）性能对比
- 不同数据类型的性能对比

❌ **不适合**：
- 需要模型状态的场景
- 长时间运行的测试
- 需要并发处理的场景

### 3.2 推理测试模式

推理测试有两种子模式：直接推理和服务推理。

#### 3.2.1 直接推理 (Direct Inference)

**应用场景**：

测试模型的端到端推理性能，模拟实际使用场景。

典型问题：
- 模型推理的平均延迟是多少？
- 在不同 batch size 下的吞吐量如何？
- 模型的峰值内存占用是多少？
- 模型的困惑度（perplexity）如何？

**测试流程**：

```
阶段1: Setup
├─ 启动资源监控 (AcceleratorMonitor)
├─ 加载模型 (adapter.setup())
└─ 准备测试数据 (PromptLoader.load())
    └─ 生成 N = (warmup + measured) * batch_size 个 prompts

阶段2: Execute
├─ Warmup 阶段 (warmup_iterations 次调用)
│   └─ 目的：预热GPU、稳定性能
│   └─ 不记录数据
│
└─ Measurement 阶段 (measured_iterations 次调用)
    ├─ 每次调用返回：
    │   ├─ texts: 生成的文本
    │   ├─ latencies: 端到端延迟
    │   └─ ttfts: 首token延迟 (Time To First Token)
    └─ 收集所有数据到 result

阶段3: Collect Metrics
├─ 停止监控，获取峰值内存
├─ 计算统计数据 (avg/p95/p99)
├─ 添加特殊指标：
│   ├─ perplexity (如果有测试集)
│   └─ accuracy (placeholder)
└─ 返回指标字典

阶段4: Teardown
├─ 卸载模型 (adapter.teardown())
└─ 停止监控
```

**关键指标**：

| 指标 | 含义 | 计算方式 |
|------|------|---------|
| **avg_latency** | 平均延迟 | 所有latency的平均值 |
| **p95_latency** | 95分位延迟 | latency的95百分位数 |
| **avg_ttft** | 平均首token延迟 | 所有ttft的平均值 |
| **throughput** | 吞吐量 | tokens/sec = (batch_size * output_tokens) / (latency/1000) |
| **peak_memory** | 峰值内存 | 监控器获取的最大内存占用 |
| **perplexity** | 困惑度 | adapter.calculate_perplexity(test_data) |

**特点**：

- **有状态**：需要加载模型，占用GPU内存
- **多轮迭代**：Warmup + Measurement 两阶段
- **同步执行**：按顺序执行每个batch
- **详细监控**：记录每次调用的详细数据

**适用情况**：

✅ **适合**：
- 模型性能基准测试
- 不同配置的性能对比
- 优化前后的性能验证
- 生成详细的性能报告

❌ **不适合**：
- 测试并发性能
- 模拟真实用户请求分布
- 测试服务稳定性

#### 3.2.2 服务推理 (Service Inference)

**应用场景**：

测试推理服务的并发处理能力和性能。

典型问题：
- 服务在不同并发度下的QPS是多少？
- 服务在请求分布不均时的表现如何？
- 服务的端到端延迟分布如何？
- 服务的稳定性和成功率如何？

**测试流程**：

```
阶段1: Setup
├─ 启动资源监控
├─ 启动推理服务 (ServiceManager.start())
│   └─ 监听端口 (如8000)
└─ 加载 trace 文件 (TraceLoader.load())
    └─ trace 文件包含：
        ├─ arrival_timestamp_ms: 请求到达时间
        ├─ input_token_num: 输入token数
        └─ output_token_num: 输出token数

阶段2: Execute (异步执行)
├─ 创建 TraceClient
├─ 按照trace中的时间戳发送请求
│   ├─ 控制并发度 (concurrency)
│   └─ 等待响应
└─ 收集响应数据
    ├─ ttft (首token延迟)
    ├─ e2e_latency (端到端延迟)
    └─ success (是否成功)

阶段3: Collect Metrics
├─ 聚合 trace 统计
│   ├─ avg_ttft
│   ├─ avg_e2e_latency
│   ├─ throughput_tps (tokens/sec)
│   └─ success_rate
├─ 保存 trace 结果到 CSV
└─ 停止监控

阶段4: Teardown
├─ 停止推理服务 (ServiceManager.stop())
└─ 停止监控
```

**关键指标**：

| 指标 | 含义 | 说明 |
|------|------|------|
| **avg_ttft** | 平均首token延迟 | 服务响应速度的重要指标 |
| **avg_e2e_latency** | 平均端到端延迟 | 从发送请求到收到完整结果 |
| **throughput_tps** | 吞吐量 | 每秒处理的token数 |
| **requests_per_second** | QPS | 每秒处理的请求数 |
| **success_rate** | 成功率 | 成功请求的百分比 |
| **peak_memory** | 峰值内存 | 测试期间的内存占用 |

**特点**：

- **异步执行**：使用 asyncio 处理并发请求
- **时间驱动**：按照trace中的时间戳发送请求
- **并发控制**：限制同时进行的请求数量
- **真实模拟**：模拟真实的用户请求分布

**Trace 文件格式**：

```csv
arrival_timestamp_ms,input_token_num,output_token_num
0,128,256
100,256,512
200,64,128
300,512,1024
...
```

**适用情况**：

✅ **适合**：
- 测试服务并发能力
- 模拟真实用户请求分布
- 测试服务稳定性
- 生产环境性能验证

❌ **不适合**：
- 模型本身性能测试（应该用直接推理）
- 需要精确控制输入的场景

### 3.3 训练测试模式（未来）

**应用场景**：

测试模型训练过程的性能。

典型问题：
- 训练速度如何（samples/sec）？
- GPU利用率如何？
- 内存使用情况如何？
- Loss曲线是否正常？

**测试流程**（规划）：

```
阶段1: Setup
├─ 启动监控
├─ 加载训练数据集 (DatasetLoader)
└─ 初始化模型和优化器

阶段2: Execute
├─ 训练循环
│   ├─ for epoch in range(epochs):
│   │   ├─ for batch in dataloader:
│   │   │   ├─ 前向传播
│   │   │   ├─ 反向传播
│   │   │   ├─ 更新参数
│   │   │   └─ 记录 loss
└─ 保存 checkpoint

阶段3: Collect Metrics
├─ Loss 曲线
├─ 训练吞吐量
├─ GPU 内存使用
└─ 最终模型准确率

阶段4: Teardown
├─ 保存最终模型
└─ 释放资源
```

**关键指标**：

| 指标 | 含义 |
|------|------|
| **loss** | 损失函数值 |
| **throughput_samples** | 训练吞吐量（样本/秒）|
| **peak_memory** | 峰值内存占用 |
| **gpu_utilization** | GPU利用率 |
| **final_accuracy** | 最终准确率 |

**特点**：

- **长时间运行**：可能需要数小时或数天
- **监控loss曲线**：观察训练是否稳定
- **checkpoint管理**：定期保存模型状态

---

## 4. 数据流设计

### 4.1 直接推理测试的完整数据流

```
┌─────────────────────────────────────────────────────────────┐
│                        输入层                              │
│  • JSON 配置文件                                             │
│  • 或 YAML 配置文件                                           │
│  • 或 CLI 命令参数                                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                      Dispatcher                             │
│  解析配置，识别 test_type = "inference:direct"              │
│  创建 DirectInferenceRunner                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                  DirectInferenceRunner                       │
│                                                               │
│  setup():                                                     │
│  ├─ 创建 AcceleratorMonitor → 监控 GPU 内存、利用率           │
│  ├─ adapter.setup() → 加载模型到 GPU                         │
│  │   └─ InfiniLMAdapter: 加载权重、初始化 KV Cache          │
│  └─ PromptLoader.load(total_prompts) → 生成测试 prompts       │
│      └─ 支持多种模式：file/template/random/fixed             │
│                                                               │
│  execute():                                                    │
│  ├─ Warmup 阶段 (10 次迭代)                                   │
│  │   └─ adapter.process() → 推理，不记录数据                  │
│  │                                                             │
│  └─ Measurement 阶段 (100 次迭代)                              │
│      └─ 每次迭代：                                             │
│          ├─ adapter.process({                                │
│          │   "prompts": batch_of_8,                           │
│          │   "max_tokens": 256                                │
│          │ })                                                 │
│          │                                                     │
│          │ → InfiniLMAdapter:                                 │
│          │   ├─ Tokenize prompts                             │
│          │   ├─ 创建 InferTasks                             │
│          │   ├─ 执行推理循环                                 │
│          │   └─ Decode tokens                                │
│          │                                                     │
│          └─ 返回 result: {                                   │
│                "texts": [...],                                 │
│                "latencies": [...],                             │
│                "ttfts": [...]                                 │
│              }                                                │
│          └─ 收集数据：                                         │
│              ├─ result.add_latency(latency)                  │
│              ├─ result.add_ttft(ttft)                         │
│              └─ result.add_throughput(throughput)             │
│                                                               │
│  collect_metrics():                                            │
│  ├─ monitor.stop() → 获取峰值内存                             │
│  ├─ calculate_statistics() → 计算 avg/p95/p99               │
│  ├─ _add_special_metrics():                                   │
│  │   ├─ _calculate_perplexity() → 计算 PPL                   │
│  │   │   └─ DatasetLoader.load() → 加载测试集               │
│  │   │       └─ adapter.calculate_perplexity(test_data)      │
│  │   └─ 添加 accuracy (placeholder)                          │
│  └─ 返回指标字典                                               │
│                                                               │
│  teardown():                                                  │
│  ├─ adapter.teardown() → 卸载模型，释放内存                  │
│  └─ monitor.stop() → 停止监控                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                     输出层                                  │
│  • JSON 结果文件                                              │
│  • CSV 数据文件 (latency.csv, ttft.csv, throughput.csv)       │
│  • 控制台日志                                                 │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 服务推理测试的异步数据流

```
┌─────────────────────────────────────────────────────────────┐
│                  Trace 文件                                 │
│  • arrival_timestamp_ms: 请求到达时间                        │
│  • input_token_num: 输入token数                             │
│  • output_token_num: 输出token数                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│               ServiceInferenceRunner                          │
│                                                               │
│  setup():                                                     │
│  ├─ 创建监控                                                  │
│  ├─ ServiceManager.start() → 启动推理服务                   │
│  │   └─ 监听 http://localhost:8000                           │
│  └─ TraceLoader.load() → 加载并验证 trace                    │
│      └─ 检查时间戳顺序、token数限制                             │
│                                                               │
│  execute():                                                    │
│  └─ asyncio.run(_run_trace_async())                           │
│      └─ 异步执行：                                            │
│          ├─ 创建 TraceClient                                 │
│          ├─ 按照时间戳发送请求                                │
│          ├─ 控制并发度 (concurrency=8)                        │
│          ├─ 收集响应数据                                      │
│          └─ 等待所有请求完成                                  │
│                                                               │
│  collect_metrics():                                            │
│  ├─ 聚合 trace 统计                                           │
│  │   ├─ avg_ttft                                            │
│  │   ├─ avg_e2e_latency                                     │
│  │   ├─ throughput_tps                                      │
│  │   └─ success_rate                                        │
│  ├─ 保存 trace 结果到 CSV                                     │
│  └─ monitor.stop()                                           │
│                                                               │
│  teardown():                                                  │
│  ├─ ServiceManager.stop() → 停止推理服务                    │
│  └─ monitor.stop()                                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                     输出层                                  │
│  • trace_result.csv (每次请求的详细数据)                     │
│  • JSON 结果文件                                              │
│  • 统计指标                                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 扩展性设计

### 5.1 添加新的测试类型

**场景**：需要添加"微调测试"（Fine-tuning Test）

**步骤**：

1. **创建新的 Runner**

```
创建文件：runners/finetuning/finetuning_runner.py

继承：RunnerBase

实现：
  - setup(): 加载模型、数据集
  - execute(): 微调循环
  - collect_metrics(): 收集loss、准确率等
  - teardown(): 保存微调后的模型
```

2. **注册到 Dispatcher**

```python
Dispatcher.register_runner("finetuning", FinetuningRunner)
```

3. **使用**

```python
Dispatcher.run_test({
    "test_type": "finetuning",
    "model": "infinilm-7b",
    "dataset": "/data/qa_dataset",
    "epochs": 3,
    "learning_rate": 1e-5
})
```

**关键点**：
- ✅ 不需要修改 Dispatcher
- ✅ 不需要修改基类
- ✅ 不需要修改其他 Runner
- ✅ 只需实现标准接口

### 5.2 添加新的框架支持

**场景**：需要支持新的推理框架（如 TensorRT-LLM）

**步骤**：

1. **创建新的 Adapter**

```
创建文件：inference/adapters/tensorrt_adapter.py

继承：BaseAdapter

实现：
  - process(): 执行推理
  - setup(): 加载 TensorRT 引擎
  - teardown(): 卸载引擎
  - validate(): 验证引擎文件存在
```

2. **在配置中指定**

```python
Dispatcher.run_test({
    "test_type": "inference:direct",
    "framework": "tensorrt",  # 使用新框架
    "engine_path": "/path/to/engine.plan",
    ...
})
```

**关键点**：
- ✅ 不需要修改 Runner
- ✅ 不需要修改 DataLoader
- ✅ 只需实现统一接口
- ✅ 所有测试模式自动支持

### 5.3 添加新的数据源

**场景**：需要从数据库加载测试 prompts

**步骤**：

1. **创建新的 Loader**

```
创建文件：loaders/database_loader.py

继承：DataLoaderBase

实现：
  - load(count): 从数据库查询 prompts
  - create_batches(...): 分批
```

2. **在配置中使用**

```python
Dispatcher.run_test({
    "test_type": "inference:direct",
    "prompt_config": {
        "mode": "database",  # 使用新模式
        "host": "localhost",
        "database": "prompts",
        "table": "test_prompts"
    }
})
```

**关键点**：
- ✅ 不需要修改 Runner
- ✅ 不需要修改 Adapter
- ✅ 只需实现 Loader 接口

---

## 6. 实施策略

### 6.1 分阶段实施

**Phase 1：核心基础（Week 1-2）**

目标：建立统一的基础设施

- [ ] 创建 `core/runner_base.py` - Runner 基类
- [ ] 创建 `core/dispatcher.py` - 统一调度器
- [ ] 创建 `core/benchmark_result.py` - 结果容器
- [ ] 定义统一的配置格式
- [ ] 编写单元测试

**验收标准**：
- Dispatcher 能够路由到不同的 Runner
- Runner 基类定义了完整的生命周期
- 结果容器能够存储和导出数据

**Phase 2：数据加载器（Week 2-3）**

目标：提取数据准备逻辑

- [ ] 创建 `loaders/base_loader.py` - DataLoader 基类
- [ ] 实现 `loaders/prompt_loader.py` - Prompt 生成器
  - [ ] 支持从文件加载
  - [ ] 支持模板生成
  - [ ] 支持随机生成
  - [ ] 支持降级策略
- [ ] 实现 `loaders/trace_loader.py` - Trace 加载器
- [ ] 实现 `loaders/dataset_loader.py` - 数据集加载器
- [ ] 编写测试用例

**验收标准**：
- PromptLoader 能够生成符合要求的测试数据
- TraceLoader 能够正确加载和验证 trace 文件
- 支持多种数据源模式

**Phase 3：推理测试重构（Week 3-4）**

目标：将现有推理测试迁移到新架构

- [ ] 重构 `DirectInferRunner` → `DirectInferenceRunner`
  - [ ] 继承 `RunnerBase`
  - [ ] 使用 `PromptLoader`
  - [ ] 简化 `execute()` 方法
- [ ] 重构 `ServiceInferRunner` → `ServiceInferenceRunner`
  - [ ] 继承 `RunnerBase`
  - [ ] 使用 `TraceLoader`
  - [ ] 保持异步执行
- [ ] 更新配置文件格式
- [ ] 迁移测试用例
- [ ] 回归测试

**验收标准**：
- 新 Runner 能够通过所有现有测试
- 代码行数减少（目标：-30%）
- 测试结果与原来一致

**Phase 4：算子测试重构（Week 4）**

目标：将算子测试纳入统一架构

- [ ] 创建 `runners/operator/operator_runner.py`
- [ ] 集成 `InfiniCoreAdapter`
- [ ] 编写测试用例
- [ ] 性能测试

**验收标准**：
- OperatorRunner 能够执行算子测试
- 与现有测试方式兼容
- 支持所有指标计算

**Phase 5：清理和优化（Week 5）**

目标：清理旧代码，优化性能

- [ ] 删除旧的基类（`InferRunnerBase`, `InferAdapter`）
- [ ] 更新所有导入语句
- [ ] 代码审查和重构
- [ ] 性能优化
- [ ] 完善文档
- [ ] 编写使用示例

**验收标准**：
- 代码库无废弃代码
- 所有测试通过
- 文档完整

### 6.2 风险控制

**风险1：重构破坏现有功能**

- **缓解**：保持新旧架构并存
- **缓解**：逐步迁移，每步都有测试保护
- **缓解**：在 `InferAdapter` 中提供兼容性方法

**风险2：性能下降**

- **缓解**：性能测试对比新旧架构
- **缓解**：保持关键路径的性能
- **缓解**：必要时优化（如使用缓存）

**风险3：迁移工作量太大**

- **缓解**：分阶段进行，每个阶段都有明确目标
- **缓解**：优先重构最常用的功能
- **缓解**：保留旧代码作为参考

---

## 7. 配置格式规范

### 7.1 通用配置结构

所有测试配置都遵循统一结构：

```yaml
# 必填字段
test_type: string           # 测试类型
run_id: string              # 运行ID（用于标识结果）

# 可选字段
output_dir: string         # 输出目录
device: object              # 设备配置
  accelerator: string       # "cuda" | "cpu"
  device_ids: list[int]     # GPU设备ID列表
  cpu_only: boolean         # 是否仅用CPU
```

### 7.2 算子测试配置

```yaml
test_type: "operator"
run_id: "test_matmul_fp16_001"

# 算子配置
operator: "matmul"          # 算子名称
device: "cuda"

# 输入tensor配置
inputs:
  - name: "A"
    shape: [1024, 1024]
    dtype: "float16"
    init: "random"          # "random" | "zeros" | "ones"

  - name: "B"
    shape: [1024, 1024]
    dtype: "float16"
    init: "random"

# 输出tensor配置
outputs:
  - name: "C"
    shape: [1024, 1024]
    dtype: "float16"

# 算子属性（可选）
attributes: []

# 需要计算的指标
metrics:
  - "latency"              # 延迟（ms）
  - "accuracy"             # 准确性
  - "tflops"               # 计算量（TFLOPS）
  - "bandwidth"            # 带宽（GB/s）

# 容差配置
tolerance:
  atol: 1e-3                # 绝对误差
  rtol: 1e-3                # 相对误差
```

### 7.3 直接推理测试配置

```yaml
test_type: "inference:direct"
run_id: "infinilm_7b_bench_001"

# 框架配置
framework: "infinilm"       # "infinilm" | "vllm" | "tensorrt"
model: "infinilm-7b"
model_path: "/models/infinilm-7b"

# 设备配置
device:
  accelerator: "cuda"
  device_ids: [0]
  cpu_only: false

# 测试参数
warmup_iterations: 10       # Warmup迭代次数
measured_iterations: 100     # 测量迭代次数
batch_size: 8               # Batch大小

# 推理参数
infer_args:
  prompt_token_num: 128     # 输入token数
  output_token_num: 256     # 输出token数
  temperature: 0.8          # 温度参数
  top_p: 0.95              # Top-p采样
  top_k: 50                # Top-k采样
  max_seq_len: 2048        # 最大序列长度

# Prompt配置
prompt_config:
  mode: "template"          # "file" | "template" | "random" | "fixed"
  template_name: "ai_qa"    # 模板名称
  topic_name: "ai_ml"       # 主题名称
  # file_path: "/path/to/prompts.jsonl"  # mode=file时
  # fixed_prompt: "Test prompt"                 # mode=fixed时

# 测试数据集（用于计算perplexity等）
test_dataset: "/data/test.json"

# 输出配置
output_dir: "/results/test001"
```

### 7.4 服务推理测试配置

```yaml
test_type: "inference:service"
run_id: "service_bench_001"

# 框架配置
framework: "infinilm"
model: "infinilm-7b"
model_path: "/models/infinilm-7b"

# 服务配置
service_config:
  port: 8000                # 服务端口
  timeout_ms: 30000         # 超时时间

# Trace配置
request_trace: "/traces/trace.csv"

# 推理参数
infer_args:
  concurrency: 8            # 并发数
  max_seq_len: 4096         # 最大序列长度
  timeout_ms: 30000         # 请求超时

# 设备配置
device:
  accelerator: "cuda"
  device_ids: [0, 1]

# 输出配置
output_dir: "/results/service_test001"
```

---

## 8. 结果格式规范

### 8.1 统一结果结构

所有测试都返回统一格式的结果：

```yaml
# 必填字段
run_id: string              # 运行ID
test_type: string           # 测试类型
success: integer           # 0=成功, 1=失败
time: string               # 完成时间

# 可选字段
error_msg: string          # 错误信息（失败时）
metrics: list[object]      # 指标列表

# 每个指标的结构
metric:
  name: string             # 指标名称
  type: string             # "scalar" | "timeseries"
  value: float             # 值（scalar类型）
  unit: string             # 单位（可为null）
  raw_data_url: string     # 数据文件路径（timeseries类型）
```

### 8.2 算子测试结果示例

```yaml
run_id: "test_matmul_001"
test_type: "operator"
success: 0
time: "2025-01-05 12:00:00"

metrics:
  - name: "operator.latency"
    type: "scalar"
    value: 1.234
    unit: "ms"

  - name: "operator.tensor_accuracy"
    type: "scalar"
    value: "PASS"
    unit: null

  - name: "operator.tflops"
    type: "scalar"
    value: 123.456
    unit: "tflops"

  - name: "operator.bandwidth"
    type: "scalar"
    value: 456.789
    unit: "GB/s"
```

### 8.3 推理测试结果示例

```yaml
run_id: "infinilm_test_001"
test_type: "inference:direct"
success: 0
time: "2025-01-05 14:30:00"

metrics:
  # 标量指标
  - name: "infer.avg_latency"
    type: "scalar"
    value: 125.5
    unit: "ms"

  - name: "infer.p95_latency"
    type: "scalar"
    value: 150.2
    unit: "ms"

  - name: "infer.avg_ttft"
    type: "scalar"
    value: 45.3
    unit: "ms"

  - name: "infer.avg_throughput_tps"
    type: "scalar"
    value: 1234.5
    unit: "tokens/s/gpu"

  - name: "infer.ppl"
    type: "scalar"
    value: 12.34
    unit: null

  - name: "infer.accuracy"
    type: "scalar"
    value: 0.0
    unit: "placeholder"

  - name: "infer.peak_memory_usage"
    type: "scalar"
    value: 15.678
    unit: "GB"

  # 时间序列指标
  - name: "infer.compute_latency"
    type: "timeseries"
    raw_data_url: "./infer/inferilm_test_001_infer_latency.csv"
    unit: "ms"

  - name: "infer.ttft"
    type: "timeseries"
    raw_data_url: "./infer/inferilm_test_001_infer_ttft.csv"
    unit: "ms"

  - name: "infer.direct_throughput_tps"
    type: "timeseries"
    raw_data_url: "./infer/inferilm_test_001_infer_throughput.csv"
    unit: "tokens/s/gpu"
```

---

## 9. API 接口规范

### 9.1 Dispatcher API

**方法：`run_test(request)`**

**功能**：执行测试

**参数**：
- `request` (dict): 测试请求字典

**返回**：
- (dict): 测试结果字典

**异常**：
- `ValueError`: 无效的 test_type
- `RuntimeError`: 测试执行失败
- `FileNotFoundError`: 配置文件或模型文件不存在

**示例**：
```python
# 算子测试
result = Dispatcher.run_test({
    "test_type": "operator",
    "operator": "add",
    ...
})

# 推理测试
result = Dispatcher.run_test({
    "test_type": "inference:direct",
    "framework": "infinilm",
    ...
})
```

**方法：`register_runner(test_type, runner_class)`**

**功能**：注册新的 Runner 类型

**参数**：
- `test_type` (str): 测试类型标识
- `runner_class` (type): Runner 类

**示例**：
```python
Dispatcher.register_runner("training", TrainingRunner)
```

**方法：`list_supported_tests()`**

**功能**：列出支持的测试类型

**返回**：
- (list[str]): 支持的测试类型列表

**示例**：
```python
types = Dispatcher.list_supported_tests()
# ["operator", "inference:direct", "inference:service"]
```

### 9.2 RunnerBase API

**必须实现的方法**：

- `setup()` - 准备测试环境
- `execute()` - 执行测试
- `collect_metrics()` - 收集指标

**可选重写的方法**：

- `teardown()` - 清理环境
- `_add_special_metrics()` - 添加特殊指标

**提供的工具方法**（继承可用）：

- `prepare_output_dir()` - 创建输出目录
- `save_timeseries_data()` - 保存时间序列数据
- `calculate_statistics()` - 计算统计数据
- `dump_json()` - 导出JSON文件

### 9.3 BaseAdapter API

**必须实现的方法**：

- `process(request)` - 处理请求

**可选实现的方法**：

- `setup(config)` - 初始化资源
- `teardown()` - 清理资源
- `validate()` - 验证配置

**提供的工具方法**：

- `is_setup()` - 检查是否已初始化
- `ensure_setup()` - 确保已初始化

---

## 10. 关键设计决策

### 10.1 为什么选择三层架构？

**Dispatcher → Runner → Adapter**

| 层级 | 职责 | 好处 |
|------|------|------|
| **Dispatcher** | 路由、调度 | 统一入口，用户友好 |
| **Runner** | 流程编排 | 测试逻辑集中管理 |
| **Adapter** | 框架接口 | 隔离外部依赖 |

**为什么不用两层？**

如果只有 Dispatcher → Adapter：
- ❌ Dispatcher 需要知道每种测试的细节
- ❌ 无法统一测试流程（warmup/measurement等）
- ❌ 特殊逻辑（异步、并发）难以处理

### 10.2 为什么统一 BaseAdapter？

**单一基类支持有/无状态两种模式**

而不是：
- ❌ StatefulAdapter / StatelessAdapter 分层
- ❌ 会导致继承层次复杂
- ❌ 增加 adapter 数量

**优点**：
- ✅ 简单 - 只有一个基类
- ✅ 灵活 - 通过可选方法支持两种模式
- ✅ 易用 - 用户不需要关心状态

### 10.3 为什么提取 DataLoader？

**而不是让 Runner 自己生成数据**

| 方式 | 优点 | 缺点 |
|------|------|------|
| **提取 DataLoader** | 职责清晰、可复用、可测试 | 增加一层抽象 |
| **Runner 自己生成** | 简单直接 | 代码重复、难以测试 |

**决策**：提取 DataLoader
- 数据准备逻辑复杂（多种模式、降级策略）
- 不同 Runner 需要相同的数据（如 DirectInference 和 ServiceInference）
- 便于单元测试

### 10.4 为什么使用钩子方法？

**而不是在基类中实现所有功能**

例如 `_add_special_metrics()` 方法：

```python
# 基类提供钩子
def _add_special_metrics(self):
    """子类可重写"""
    pass

# DirectInferenceRunner 重写
def _add_special_metrics(self):
    # 添加 perplexity
    # 添加 accuracy
    pass

# OperatorRunner 不重写
# （因为不需要这些指标）
```

**优点**：
- ✅ 基类不包含特定逻辑
- ✅ 子类按需重写
- ✅ 符合开闭原则

---

## 11. 性能考虑

### 11.1 内存管理

**问题**：长时间运行的测试可能导致内存泄漏

**解决方案**：

1. **及时释放资源**
   - `teardown()` 中确保释放所有资源
   - 使用 `with` 语句管理文件、连接等

2. **限制数据收集**
   - 限制 CSV 文件的大小
   - 对时间序列数据进行抽样

3. **监控内存使用**
   - 使用 AcceleratorMonitor 实时监控
   - 超过阈值时告警

### 11.2 并发控制

**问题**：服务测试可能产生过多并发请求

**解决方案**：

1. **配置并发度**
   ```yaml
   infer_args:
     concurrency: 8  # 最大并发请求数
   ```

2. **使用 asyncio.Semaphore**
   ```python
   semaphore = asyncio.Semaphore(concurrency)

   async with semaphore:
       await client.send_request(...)
   ```

3. **超时控制**
   ```yaml
   service_config:
     timeout_ms: 30000  # 30秒超时
   ```

### 11.3 数据序列化

**问题**：大量结果数据（如时间序列）可能很慢

**解决方案**：

1. **使用 CSV 格式**
   - 比 JSON 更紧凑
   - 便于用 pandas 等工具分析

2. **增量写入**
   - 每次迭代后追加到 CSV
   - 而不是全部保存在内存中

3. **压缩存储**（可选）
   - 对 CSV 文件进行压缩
   - 节省磁盘空间

---

## 12. 测试策略

### 12.1 单元测试

**目标**：确保每个组件独立工作正常

**关键测试**：

- [x] `BaseAdapter` 的生命周期
- [x] `Dispatcher` 的路由逻辑
- [x] `PromptLoader` 的各种模式
- [x] `TraceLoader` 的验证逻辑
- [x] `RunnerBase` 的工具方法

### 12.2 集成测试

**目标**：确保组件之间协作正常

**关键测试**：

- [ ] Dispatcher → OperatorRunner → InfiniCoreAdapter
- [ ] Dispatcher → DirectInferenceRunner → InfiniLMAdapter
- [ ] Dispatcher → ServiceInferenceRunner → InfiniLMAdapter
- [ ] DataLoader → Runner 数据流

### 12.3 回归测试

**目标**：确保重构不破坏现有功能

**测试数据**：

- [ ] 现有的算子测试用例
- [ ] 现有的推理测试用例
- [ ] 现有的服务测试用例

**验收标准**：

- 新旧架构的测试结果一致（误差 < 1%）
- 所有测试用例通过
- 性能无明显下降

---

## 13. 文档和培训

### 13.1 用户文档

**需要编写的文档**：

1. **快速开始指南**
   - 如何运行第一个测试
   - 常见配置示例
   - 故障排查

2. **配置参考**
   - 所有配置选项说明
   - 不同测试模式的配置示例
   - 最佳实践

3. **API 文档**
   - Dispatcher API
   - Runner API
   - Adapter API

4. **开发者指南**
   - 如何添加新的测试类型
   - 如何添加新的框架支持
   - 代码架构说明

### 13.2 示例和教程

**需要提供的示例**：

1. **简单示例**
   - 算子测试：测试 matmul 性能
   - 推理测试：测试模型吞吐量
   - 服务测试：测试并发性能

2. **高级示例**
   - 使用自定义 prompt 生成
   - 使用自定义 trace
   - 添加新的指标计算

3. **完整示例**
   - 端到端的性能基准测试
   - A/B 测试（对比两个模型）
   - 回归测试套件

---

## 14. 总结

### 14.1 架构优势

1. **统一入口**
   - 所有测试通过 Dispatcher 进行
   - 用户无需关心底层差异
   - API 简单一致

2. **职责清晰**
   - Dispatcher 负责调度
   - Runner 负责流程编排
   - DataLoader 负责数据准备
   - Adapter 负责框架接口

3. **易于扩展**
   - 添加新测试类型：创建新 Runner
   - 添加新框架：创建新 Adapter
   - 添加新数据源：创建新 DataLoader
   - 都不需要修改核心代码

4. **代码复用**
   - 通用功能在基类中实现
   - 减少重复代码
   - 提高代码质量

5. **易于维护**
   - 模块化设计
   - 清晰的接口
   - 完整的文档

### 14.2 实施建议

1. **分阶段进行**
   - 不要一次性重写所有代码
   - 每个阶段都有明确目标
   - 保持向后兼容

2. **先基础后功能**
   - 先建立核心架构
   - 再逐步迁移功能
   - 最后清理旧代码

3. **测试驱动**
   - 编写充分的单元测试
   - 保证重构不破坏功能
   - 使用测试作为文档

4. **文档先行**
   - 先设计接口和配置格式
   - 再实现具体功能
   - 最后补充示例和教程

### 14.3 未来展望

**短期目标**（3个月）：
- ✅ 完成核心架构
- ✅ 迁移所有测试
- ✅ 清理旧代码

**中期目标**（6个月）：
- 添加训练测试支持
- 添加更多框架支持（vLLM, TensorRT-LLM等）
- 性能优化

**长期目标**（1年）：
- Web UI界面
- 分布式测试支持
- 集成到CI/CD流程
- 性能分析和可视化

---

**文档版本**: v1.0
**最后更新**: 2025-01-05
**维护者**: InfiniTensor Team

**变更历史**：
- v1.0 (2025-01-05): 初始版本
