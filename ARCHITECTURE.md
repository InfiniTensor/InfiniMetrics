# InfiniMetrics 统一测试系统架构设计

**版本**: v2.0 (精简版)
**日期**: 2025-01-05

---

## 一、核心设计

### 1.1 架构分层

```
用户层
  ↓
Dispatcher (调度) - 路由到合适的测试类型
  ↓
Runner (执行) - 编排测试流程，管理生命周期
  ↓
Adapter (接口) - 与具体框架交互
  ↓
DataLoader (数据) - 准备测试数据
```

**关键点**：
- **Dispatcher**：根据配置选择合适的 Runner（算子/推理/训练）
- **Runner**：统一的测试流程（setup → execute → collect → teardown）
- **Adapter**：单一接口 `process(request)`，可选 `setup/teardown`
- **DataLoader**：提取出的数据准备逻辑（prompts、traces、datasets）

### 1.2 统一接口

**BaseAdapter**（单一基类，适配所有场景）：
```python
class BaseAdapter(ABC):
    @abstractmethod
    def process(self, request: dict) -> dict:
        """必须实现 - 处理请求"""
        pass

    def setup(self, config: dict = None) -> None:
        """可选 - 初始化资源（有状态adapter重写）"""
        pass

    def teardown(self) -> None:
        """可选 - 清理资源（有状态adapter重写）"""
        pass
```

**设计优势**：
- **无状态 adapter**（如 InfiniCore）：只实现 `process()`
- **有状态 adapter**（如 InfiniLM）：实现 `setup + process + teardown`
- **统一接口**：所有 adapter 使用同一套接口，易于理解

### 1.3 Runner 生命周期

所有 Runner 遵循统一的生命周期：

```python
class RunnerBase:
    def setup(self):
        """初始化：创建 adapter、加载模型、启动监控"""
        pass

    def execute(self):
        """执行：运行测试（可能包含 warmup + measurement）"""
        pass

    def collect_metrics(self):
        """收集：停止监控、计算统计"""
        pass

    def teardown(self):
        """清理：卸载模型、保存结果"""
        pass
```

---

## 二、三种测试模式

### 2.1 算子测试（Operator Testing）

**场景**：测试单个算子的性能和正确性

**特点**：
- 单次调用，无状态
- 快速反馈（秒级）
- 关注算子延迟、tensor 正确性

**示例流程**：
```
1. 加载算子配置（operator、device、inputs/outputs）
2. 调用 adapter.process() 执行算子
3. 收集延迟和精度指标
```

**使用示例**：
```yaml
test_type: operator
framework: infinicore
config:
  operator: "add"
  device: "cuda"
  inputs: [...]
  outputs: [...]
```

### 2.2 直接推理测试（Direct Inference）

**场景**：测试端到端模型推理性能

**特点**：
- 需要加载模型（有状态）
- Warmup + Measurement 两阶段
- 关注 TTFT、TPOT、吞吐量

**示例流程**：
```
1. adapter.setup() - 加载模型
2. Warmup 阶段 - 预热模型（不计入统计）
3. Measurement 阶段 - 正式测试
4. 收集指标（延迟、token/s、perplexity等）
5. adapter.teardown() - 卸载模型
```

**使用示例**：
```yaml
test_type: direct_inference
framework: infinilm
config:
  model_path: "/path/to/model"
  prompts:
    source: "file"
    path: "prompts.txt"
  measurement:
    iterations: 100
    warmup: 10
```

### 2.3 服务推理测试（Service Inference）

**场景**：测试推理服务的并发性能

**特点**：
- 异步执行，并发请求
- 基于 trace 文件驱动
- 关注 QPS、尾延迟、资源利用率

**示例流程**：
```
1. 启动推理服务
2. 加载 trace 文件（请求序列）
3. 异步执行请求（模拟真实负载）
4. 收集服务级指标
5. 停止服务
```

**使用示例**：
```yaml
test_type: service_inference
framework: infinilm
config:
  model_path: "/path/to/model"
  trace:
    source: "file"
    path: "trace.json"
  service:
    host: "localhost"
    port: 8000
```

---

## 三、关键设计决策

### 3.1 为什么单一 BaseAdapter？

**问题**：算子测试和推理测试差异很大，如何统一？

**解决**：通过可选的 `setup/teardown` 方法支持两种模式
- **无状态**（算子）：只实现 `process()`
- **有状态**（推理）：实现 `setup + process + teardown`

**优势**：
- 接口统一，易于理解
- 灵活支持不同场景
- 避免多层继承的复杂度

### 3.2 为什么提取 DataLoader？

**问题**：测试数据准备逻辑分散在各个 Runner 中

**解决**：提取独立的数据准备组件

**职责**：
- 从文件/内存/生成器加载 prompts
- 从文件加载 trace
- 验证数据格式

**优势**：
- 数据准备逻辑复用
- Runner 更专注于流程编排
- 易于测试和维护

### 3.3 Runner vs Adapter 的职责边界？

**Runner**（流程编排）：
- 管理测试生命周期
- 编排测试流程（warmup → measurement）
- 收集和聚合指标
- 保存结果

**Adapter**（框架接口）：
- 与具体框架交互
- 执行单个操作（算子/推理）
- 不关心测试流程

**关键区别**：
- Runner 知道"何时做什么"（流程）
- Adapter 知道"如何做"（实现）

---

## 四、数据流

### 4.1 算子测试

```
Config → Dispatcher → OperatorRunner
                      ↓
                   DataLoader (准备算子配置)
                      ↓
                   InfiniCoreAdapter.process()
                      ↓
                   收集延迟和精度 → 返回结果
```

### 4.2 直接推理测试

```
Config → Dispatcher → DirectInferenceRunner
                      ↓
                   DataLoader (加载 prompts)
                      ↓
                   InfiniLMAdapter.setup() (加载模型)
                      ↓
                   Warmup 阶段 (adapter.process × N)
                      ↓
                   Measurement 阶段 (adapter.process × N + 监控)
                      ↓
                   收集指标 → InfiniLMAdapter.teardown()
                      ↓
                   返回结果 (TTFT, TPOT, 吞吐量等)
```

### 4.3 服务推理测试

```
Config → Dispatcher → ServiceInferenceRunner
                      ↓
                   DataLoader (加载 trace)
                      ↓
                   启动推理服务
                      ↓
                   异步执行 trace 中的请求
                      ↓
                   收集服务级指标 (QPS, 尾延迟等)
                      ↓
                   停止服务 → 返回结果
```

---

## 五、实施计划

### Phase 1: 统一 Adapter（已完成 ✅）
- [x] 实现 BaseAdapter（支持有/无状态）
- [x] 重构 InfiniCoreAdapter
- [x] 重构 InfiniLMAdapter
- [x] 验证向后兼容性

### Phase 2: 提取 DataLoader（进行中）
- [ ] 创建 DataLoader 基类
- [ ] 实现 PromptLoader（文件/内存/生成）
- [ ] 实现 TraceLoader（JSON格式）
- [ ] 从 Runner 中迁移数据准备逻辑

### Phase 3: 统一 Runner（下一步）
- [ ] 创建 RunnerBase（统一生命周期）
- [ ] 实现 OperatorRunner（轻量级）
- [ ] 重构 DirectInferenceRunner
- [ ] 重构 ServiceInferenceRunner

### Phase 4: 实现 Dispatcher
- [ ] 创建配置路由器
- [ ] 实现 CLI 入口
- [ ] 实现 Python API

### Phase 5: 清理和优化
- [ ] 删除旧代码
- [ ] 更新文档
- [ ] 性能优化

---

## 六、配置格式示例

### 6.1 算子测试配置

```yaml
test:
  type: operator
  name: "matmul-performance-test"

framework:
  name: infinicore
  config:
    operator: "matmul"
    device: "cuda:0"
    inputs:
      - {name: "A", shape: [1024, 1024], dtype: "float32"}
      - {name: "B", shape: [1024, 1024], dtype: "float32"}
    outputs:
      - {name: "C", shape: [1024, 1024], dtype: "float32"}

metrics:
  - {name: "latency", unit: "ms"}
  - {name: "tensor_accuracy"}
```

### 6.2 直接推理配置

```yaml
test:
  type: direct_inference
  name: "llm-inference-benchmark"

framework:
  name: infinilm
  config:
    model_path: "/models/llama-2-7b"
    device: {accelerator: "nvidia", device_ids: [0]}

data:
  prompts:
    source: "file"
    path: "prompts.txt"
    count: 100

execution:
  warmup: 10
  measurement: 100
  max_tokens: 512
  temperature: 0.8
  top_p: 0.95

metrics:
  - {name: "ttft", unit: "ms"}
  - {name: "tpot", unit: "ms/token"}
  - {name: "throughput", unit: "tokens/s"}
  - {name: "perplexity"}
```

### 6.3 服务推理配置

```yaml
test:
  type: service_inference
  name: "concurrent-inference-test"

framework:
  name: infinilm
  config:
    model_path: "/models/llama-2-7b"

service:
  host: "localhost"
  port: 8000
  max_concurrent: 10

data:
  trace:
    source: "file"
    path: "trace.json"

metrics:
  - {name: "qps", unit: "requests/s"}
  - {name: "latency_p50", unit: "ms"}
  - {name: "latency_p99", unit: "ms"}
```

---

## 七、扩展性设计

### 7.1 添加新的 Adapter

**场景**：支持 vLLM 框架

**步骤**：
1. 继承 `BaseAdapter`
2. 实现 `setup/load_model`（加载 vLLM 模型）
3. 实现 `process/generate`（执行推理）
4. 实现 `teardown/unload_model`（释放资源）

**代码量**：约 150-200 行（复用基类逻辑）

### 7.2 添加新的 Runner

**场景**：支持训练测试

**步骤**：
1. 继承 `RunnerBase`
2. 实现 `setup`（初始化训练环境）
3. 实现 `execute`（运行训练循环，监控 loss）
4. 实现 `collect_metrics`（收集训练指标）

**代码量**：约 200-300 行（复用生命周期管理）

### 7.3 添加新的数据源

**场景**：从数据库加载 prompts

**步骤**：
1. 实现 `DataLoader` 子类
2. 实现 `load()` 方法（连接数据库、查询数据）
3. 返回标准格式（list of prompts）

---

## 八、FAQ

**Q1: 为什么不分离 OperatorAdapter 和 ModelAdapter？**
A: 单一基类更简洁。通过可选的 `setup/teardown` 就能支持两种模式，避免过度设计。

**Q2: DataLoader 是否必要？**
A: 是的。当前 Runner 中有大量数据准备逻辑（prompts 生成、trace 加载等），提取后可以复用，减少重复代码。

**Q3: 如何处理异步测试（如 ServiceInference）？**
A: Runner 支持异步 `execute()` 方法。基类提供 `async_execute()` 钩子，子类可以重写。

**Q4: 如何支持自定义指标？**
A: Runner 提供 `_add_special_metrics()` 钩子方法，子类可以重写来添加特定指标（如 perplexity、accuracy）。

**Q5: 向后兼容性如何保证？**
A: Adapter 层保留旧接口方法（如 `load_model/generate/unload_model`），内部调用新接口（`setup/process/teardown`）。

---

## 九、总结

**核心价值**：
1. **统一接口** - 所有测试使用同一套框架
2. **清晰分层** - 职责明确，易于维护
3. **易于扩展** - 添加新功能无需修改核心代码
4. **代码复用** - 通用逻辑在基类中实现

**实施状态**：
- ✅ Phase 1 完成（统一 Adapter）
- 🔄 Phase 2 进行中（提取 DataLoader）
- ⏳ Phase 3-5 待实施

**下一步**：
完成 DataLoader 提取，然后开始统一 Runner。

---

**文档长度**：约 400 行（适合 30-45 分钟 review）
**目标读者**：架构师、核心开发者
**反馈方式**：请直接在文档上批注或发起 PR 讨论
