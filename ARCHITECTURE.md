# InfiniMetrics 统一测试系统架构设计

## 一、核心设计

### 1.1 架构分层

```
用户层 (配置文件)
  ↓
Dispatcher (编排) - 解析配置，编排测试任务
  ↓
Runner (执行) - 通用流程编排器
  ↓
DataLoader (数据) - 独立的数据准备组件
  ↓
Adapter (接口) - 使用注册表模式管理框架适配器
```

**关键点**：

- **Dispatcher**：测试编排器，负责解析配置、生成测试任务、管理多个 Runner 实例、汇总结果
- **Runner**：通用执行器，负责单次测试的生命周期管理（setup/execute/teardown）
- **DataLoader**：独立的数据准备组件，支持工厂模式创建
- **Adapter**：使用装饰器注册，统一接口 `process(request)` + `setup/teardown`（可选）

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

### 1.3 Dispatcher 编排模式

**Dispatcher 的职责**：

```python
class Dispatcher:
    """测试编排器"""

    def dispatch(self, config: Config):
        """解析配置，生成并执行测试任务"""
        # 1. 解析配置，生成测试任务列表
        tasks = self._parse_tasks(config)

        # 2. 为每个任务创建 Runner 实例
        runners = []
        for task_config in tasks:
            runner = Runner(task_config)
            runners.append(runner)

        # 3. 执行所有 Runner（可串行/并行）
        results = []
        for runner in runners:
            result = runner.run()
            results.append(result)

        # 4. 汇总所有测试结果
        return self._aggregate_results(results)

    def _parse_tasks(self, config: Config) -> List[Config]:
        """解析配置，生成任务列表"""
        if config["test_type"] == "parameter_sweep":
            # 参数扫描：生成多组参数组合
            return self._generate_sweep_tasks(config)
        elif config["test_type"] == "comparison":
            # 框架对比：为每个框架生成任务
            return self._generate_comparison_tasks(config)
        else:
            # 单次测试：直接返回配置
            return [config]
```

**Runner 的生命周期**：

```python
class Runner:
    """通用执行器"""

    def setup(self):
        """setup 流程：创建公共组件"""
        # 创建 adapter、data_loader、初始化 metrics
        self.adapter = AdapterFactory.create(self.config["framework"])
        self.data_loader = DataLoaderFactory.from_config(self.config["data"])
        self.metrics = {}

        # 框架特有的资源管理交给 adapter
        self.adapter.setup(self.config["model_config"])

        # 启动监控
        self.monitor = Monitor()
        self.monitor.start()

    def execute(self):
        """execute 流程：根据配置执行不同测试"""
        test_type = self.config["test_type"]

        if test_type == "operator":
            self._execute_operator_test()
        elif test_type == "direct_inference":
            self._execute_direct_inference()
        elif test_type == "service_inference":
            self._execute_service_inference()

    def teardown(self):
        """teardown 流程：清理资源"""
        # 停止监控
        if self.monitor:
            self.monitor.stop()

        # 框架特有的资源清理交给 adapter
        self.adapter.teardown()

        # 保存结果
        self._save_results()
```

**执行顺序**：

```
1. Dispatcher.dispatch(config)
   ↓
2. 解析配置，生成任务列表 [task1, task2, task3, ...]
   ↓
3. 为每个任务创建 Runner 实例
   ↓
4. 执行所有 Runner:
   Runner1: setup() → execute() → teardown()
   Runner2: setup() → execute() → teardown()
   Runner3: setup() → execute() → teardown()
   ↓
5. 汇总所有结果并返回
```

### 1.4 注册表模式

**Adapter 注册表**：

```python
class AdapterFactory:
    _adapters = {}

    @classmethod
    def register(cls, name: str):
        """装饰器：注册 Adapter"""
        def decorator(adapter_class):
            cls._adapters[name] = adapter_class
            return adapter_class
        return decorator

    @classmethod
    def create(cls, name: str):
        """创建 Adapter"""
        return cls._adapters[name](config)

# 使用装饰器注册
@AdapterFactory.register("infinilm")
class InfiniLMAdapter(BaseAdapter):
    pass

@AdapterFactory.register("vllm")
class VLLMAdapter(BaseAdapter):
    pass
```

**优势**：

- ✅ 添加新框架 Adapter 只需注册，无需修改工厂代码
- ✅ 消除大量 if-elif-else 判断
- ✅ 支持动态注册（运行时注册）
- ✅ 代码简洁，易于维护

## 二、测试模式与编排

### 2.1 单次测试 vs 多次测试编排

#### 场景1：单次测试（简单场景）
```yaml
# 单次直接推理测试
test_type: direct_inference
framework: infinilm
model: /path/to/model
data:
  source: file
  file_path: /path/to/prompts.json
```

**执行流程**：
```
Config → Dispatcher → 解析为 1 个任务
                     ↓
                  Runner (单次执行)
                     ↓
                  返回结果
```

#### 场景2：参数扫描（多次测试）
```yaml
# 扫描不同的 batch_size
test_type: parameter_sweep
framework: infinilm
model: /path/to/model
sweep_parameters:
  batch_size: [1, 2, 4, 8, 16]
  prompt_length: [128, 512]
```

**执行流程**：
```
Config → Dispatcher → 解析为 10 个任务 (5×2)
                     ↓
    ┌─────────────────┼─────────────────┐
    ↓                 ↓                 ↓
Runner(bs=1,pl=128) Runner(bs=2,pl=128) ... Runner(bs=16,pl=512)
    ↓                 ↓                 ↓
  结果1              结果2              结果10
    └─────────────────┼─────────────────┘
                      ↓
                 汇总结果 (对比图表)
```

#### 场景3：框架对比（A/B测试）
```yaml
# 对比不同框架的性能
test_type: comparison
frameworks: [infinilm, vllm, transformers]
test_config:
  test_type: direct_inference
  model: /path/to/model
  data: ...
```

**执行流程**：
```
Config → Dispatcher → 为每个框架生成任务
                     ↓
    ┌─────────┬─────────┬─────────┐
    ↓         ↓         ↓         ↓
InfiniLM   vLLM   Transformers
  Runner     Runner     Runner
    ↓         ↓         ↓
  结果1      结果2      结果3
    └─────────┼─────────┘
              ↓
         对比分析报告
```

### 2.2 典型测试类型

**核心思想**：通过 **配置** 而非 **代码** 区分不同的测试类型

**单一 Runner 的实现**：

```python
class Runner:
    """通用 Runner - 支持所有测试类型"""

    def __init__(self, config: Config):
        self.config = config
        self.adapter = AdapterFactory.create(config["framework"])
        self.data_loader = DataLoaderFactory.from_config(config["data"])

    def setup(self):
        """通用 setup 流程"""
        self.adapter.setup(self.config["model_config"])
        self.monitor = Monitor()
        self.monitor.start()

    def execute(self):
        """通用 execute 流程 - 根据配置决定行为"""
        test_type = self.config["test_type"]

        if test_type == "operator":
            self._execute_operator_test()
        elif test_type == "direct_inference":
            self._execute_direct_inference()
        elif test_type == "service_inference":
            self._execute_service_inference()

    def teardown(self):
        """通用 teardown 流程"""
        self.monitor.stop()
        self.adapter.teardown()
        self._save_results()
```

**配置示例**：

```yaml
# 算子测试配置
test_type: operator
framework: infinicore
data:
  source: file
  file_path: /path/to/operator_config.json

# 直接推理测试配置
test_type: direct_inference
framework: infinilm
data:
  source: file
  file_path: /path/to/prompts.json

# 服务推理测试配置
test_type: service_inference
framework: infinilm
data:
  source: trace
  trace_file: /path/to/trace.json
```

### 2.2 不同测试类型的实现差异

虽然 Runner 是统一的，但不同测试类型的执行逻辑不同：

**算子测试**：
- 加载算子配置
- 调用 `adapter.process()` 执行单次算子操作
- 收集延迟和精度指标

**直接推理测试**：
- 加载模型（通过 `adapter.setup()`）
- Warmup 阶段：多次调用 `adapter.process()` 预热
- Measurement 阶段：多次调用 `adapter.process()` 并收集指标
- 收集 TTFT、TPOT、吞吐量等指标

**服务推理测试**：
- 启动推理服务
- 加载 trace 文件
- 异步并发发送请求
- 收集 QPS、尾延迟等指标

**关键点**：所有差异都通过配置和 Adapter 的不同实现来体现，Runner 本身保持通用。

### 2.3 Dispatcher 编排策略

#### 编排策略1：参数扫描

**用途**：测试不同参数组合下的性能

**配置示例**：
```yaml
test_type: parameter_sweep
framework: infinilm
model: /path/to/model
test_type: direct_inference
sweep_parameters:
  batch_size: [1, 2, 4, 8]
  max_tokens: [128, 256, 512]
  temperature: [0.1, 0.7, 1.0]
```

**Dispatcher 行为**：
- 生成 4×3×3 = 36 个测试任务
- 每个任务使用不同的参数组合
- 串行或并行执行所有任务
- 汇总结果，生成性能对比报告

**结果示例**：
```json
{
  "best_config": {"batch_size": 8, "max_tokens": 256, "temperature": 0.7},
  "all_results": [
    {"config": {...}, "throughput": 100, "latency": 50},
    {"config": {...}, "throughput": 150, "latency": 40},
    ...
  ],
  "heatmap": "性能热力图数据"
}
```

#### 编排策略2：框架对比

**用途**：在相同配置下对比不同框架的性能

**配置示例**：
```yaml
test_type: comparison
frameworks: [infinilm, vllm, transformers]
test_config:
  test_type: direct_inference
  model: /path/to/model
  data: ...
  warmup_iterations: 10
  measured_iterations: 100
```

**Dispatcher 行为**：
- 为每个框架创建一个测试任务
- 确保所有任务使用相同的测试配置
- 串行执行（避免资源竞争）
- 生成对比分析报告

**结果示例**：
```json
{
  "comparison": {
    "infinilm": {"throughput": 150, "latency": 40, "memory": 10.5},
    "vllm": {"throughput": 180, "latency": 35, "memory": 12.0},
    "transformers": {"throughput": 100, "latency": 60, "memory": 8.0}
  },
  "ranking": ["vllm", "infinilm", "transformers"],
  "speedup": {
    "vllm_vs_infinilm": "1.2x",
    "vllm_vs_transformers": "1.8x"
  }
}
```

#### 编排策略3：混合测试流程

**用途**：按顺序执行多个测试步骤

**配置示例**：
```yaml
test_type: workflow
steps:
  - name: smoke_test
    config:
      test_type: operator
      operator: matmul
      iterations: 10

  - name: load_test
    config:
      test_type: service_inference
      trace: /path/to/trace.json
      concurrency: 10
      condition: "smoke_test.success == true"

  - name: full_test
    config:
      test_type: direct_inference
      iterations: 1000
      condition: "load_test.p95_latency < 1000"
```

**Dispatcher 行为**：
- 按顺序执行每个步骤
- 支持条件判断（基于前一步的结果）
- 任何步骤失败可以终止整个流程
- 生成完整的测试报告

#### 编排策略4：并发测试

**用途**：同时运行多个独立的测试任务

**配置示例**：
```yaml
test_type: concurrent
tasks:
  - test_type: direct_inference
    model: /path/to/model1
  - test_type: service_inference
    model: /path/to/model2
  - test_type: operator
    operator: conv2d
execution: parallel  # 并行执行
max_concurrency: 3
```

**Dispatcher 行为**：
- 同时启动多个 Runner 实例
- 控制并发数量（避免资源耗尽）
- 等待所有任务完成
- 汇总所有结果

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

### 3.2 为什么使用注册表模式？

**问题**：如何优雅地管理多个 Runner 和 Adapter？

**解决**：使用注册表模式 + 装饰器

**优势**：

| 传统工厂模式 | 注册表模式 |
|------------|-----------|
| 大量 if-elif-else | 一行装饰器注册 |
| 添加新类型需改工厂 | 只需注册，无需改代码 |
| 代码冗长 | 代码简洁 |
| 难以扩展 | 易于扩展 |

**对比**：

```python
# ❌ 传统工厂模式
def create_adapter(framework):
    if framework == "infinilm":
        return InfiniLMAdapter()
    elif framework == "vllm":
        return VLLMAdapter()
    elif framework == "transformers":
        return TransformersAdapter()
    # ... 越来越多

# ✅ 注册表模式
@AdapterFactory.register("infinilm")
class InfiniLMAdapter(BaseAdapter):
    pass

adapter = AdapterFactory.create("infinilm")
```

### 3.3 为什么需要 Dispatcher？

**问题**：为什么不能直接创建 Runner，而需要 Dispatcher 这一层？

**解决**：Dispatcher 负责测试编排，Runner 负责测试执行

**优势**：

- ✅ **职责分离**：Dispatcher 关注"做什么"（编排），Runner 关注"怎么做"（执行）
- ✅ **支持复杂测试**：参数扫描、A/B测试、混合流程等需要编排能力
- ✅ **资源管理**：统一管理多个 Runner 实例，控制并发
- ✅ **结果聚合**：自动汇总多个测试任务的结果

**对比**：

```python
# ❌ 没有 Dispatcher：用户需要手动管理多个 Runner
def run_parameter_sweep(config):
    results = []
    for batch_size in [1, 2, 4, 8]:
        for max_tokens in [128, 256, 512]:
            cfg = copy(config)
            cfg.batch_size = batch_size
            cfg.max_tokens = max_tokens
            runner = Runner(cfg)
            result = runner.run()
            results.append(result)
    # 手动汇总结果...
    return aggregate(results)

# ✅ 有 Dispatcher：自动处理编排
config = {
    "test_type": "parameter_sweep",
    "sweep_parameters": {
        "batch_size": [1, 2, 4, 8],
        "max_tokens": [128, 256, 512]
    }
}
dispatcher = Dispatcher()
results = dispatcher.dispatch(config)  # 自动生成任务并执行
```

### 3.4 Runner vs Adapter 的职责边界？

**Runner**（测试级别 - 流程编排）：

- **职责**：编排测试流程，管理测试级别的通用逻辑
- **setup() 做什么**：
  - 选择 DataLoader → 加载测试数据
  - 创建 Adapter → 调用 `adapter.setup()`
  - 启动 Monitor → 开始监控资源
  - 初始化 MetricsCollector → 准备收集指标
- **execute() 做什么**：
  - 执行测试逻辑（调用 adapter.process()）
  - 收集测试数据
- **teardown() 做什么**：
  - 调用 `adapter.teardown()` → 清理框架资源
  - 停止 Monitor → 停止监控
  - 保存结果 → 写入文件/数据库
  - 清理临时文件 → 测试产生的临时数据

**Adapter**（框架级别 - 资源管理）：

- **职责**：管理框架资源，与具体框架交互
- **setup(config) 做什么**：
  - 加载模型 → 从磁盘加载到内存/GPU
  - 初始化 tokenizer → 文本编码器
  - 分配 GPU 内存 → 为模型预留空间
  - 编译/优化模型 → 框架特定的优化
- **process(request) 做什么**：
  - 执行单个操作（算子/推理）
- **teardown() 做什么**：
  - 卸载模型 → 释放 GPU 内存
  - 关闭连接 → 数据库/网络连接
  - 清理缓存 → KV cache 等
  - 重置状态 → 恢复到初始状态

**关键区别**：

| | **Runner** | **Adapter** |
|---|---|---|
| **抽象层级** | 测试级别（Test Level） | 框架级别（Framework Level） |
| **核心职责** | 流程编排 | 资源管理 |
| **是否必须实现** | ✅ 必须实现 `execute()` | ✅ 必须实现 `process()` |
| **setup/teardown** | 固定实现（通用逻辑） | ⚠️ 可选重写（有状态/无状态） |
| **是否调用对方** | 调用 `adapter.setup()/teardown()` | 不调用 runner |

**类比**：

```
Runner = 服务员（协调一切）
Adapter = 厨师（做菜）

服务员 (Runner) 的职责：
  ✅ 准备菜单 (DataLoader)
  ✅ 叫厨师来上班 (创建 Adapter)
  ✅ 记录订单 (调用 adapter.process())
  ✅ 准备餐桌 (Monitor)
  ✅ 结算账单 (保存结果)

厨师 (Adapter) 的职责：
  ✅ 穿上厨师服、准备厨具 (setup)
  ✅ 按订单做菜 (process)
  ✅ 清理厨房、关火 (teardown)

关键点：
  - 服务员不关心厨师怎么做菜（框架实现细节）
  - 厨师不关心服务员怎么服务（测试流程）
  - 各司其职，通过接口交互
```

### 3.5 为什么提取 DataLoader？

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

**工厂模式**：

```python
class DataLoaderFactory:
    @staticmethod
    def from_config(config: dict):
        source = config.get("source")

        if source == "file":
            return FilePromptLoader(config["file_path"])
        elif source == "generate":
            return GeneratePromptLoader(config["count"], config["template"])
        elif source == "trace":
            return TraceLoader(config["trace_file"])

# 使用
data_loader = DataLoaderFactory.from_config(config["data"])
prompts = data_loader.load()
```

## 四、数据流

### 4.1 单次测试：算子测试

```
Config (test_type: operator)
           ↓
       Dispatcher (解析为 1 个任务)
           ↓
       Runner (通用执行器)
           ↓
       setup() ← 创建 InfiniCoreAdapter、DataLoader → 启动监控
           ↓
       execute() ← 根据 test_type 分发到 _execute_operator()
              → 加载算子配置 → adapter.process() 执行算子
           ↓
       teardown() ← 停止监控 → 保存结果
           ↓
       Dispatcher ← 返回结果
           ↓
       返回给用户 (延迟、精度)
```

### 4.2 多次测试：参数扫描

```
Config (test_type: parameter_sweep)
           ↓
       Dispatcher (解析配置，生成多个任务)
           ↓
    ┌────────┬────────┬────────┐
    ↓        ↓        ↓        ↓
任务1:    任务2:    任务3:   任务4:
bs=1      bs=2      bs=4     bs=8
    ↓        ↓        ↓        ↓
Runner1   Runner2   Runner3  Runner4
    ↓        ↓        ↓        ↓
  结果1     结果2     结果3    结果4
    └────────┴────────┴────────┘
              ↓
       Dispatcher (汇总结果)
              ↓
       生成对比报告
           ↓
       返回给用户 (性能对比图表)
```

**关键点**：
- Dispatcher 负责生成多个测试任务
- 每个 Runner 独立执行
- Dispatcher 汇总所有结果

### 4.3 多次测试：框架对比

```
Config (test_type: comparison, frameworks: [infinilm, vllm])
           ↓
       Dispatcher (为每个框架生成任务)
           ↓
    ┌──────────────┬──────────────┐
    ↓              ↓              ↓
任务1: InfiniLM  任务2: vLLM
    ↓              ↓
Runner1         Runner2
    ↓              ↓
  结果1          结果2
    └──────────────┴──────────────┘
              ↓
       Dispatcher (对比分析)
              ↓
       生成对比报告
           ↓
       返回给用户 (哪个框架更快、性价比等)
```

**关键点**：
- Dispatcher 确保所有框架使用相同的测试配置
- 可以串行执行（避免资源竞争）或并行执行
- 自动生成对比分析报告

## 五、扩展性设计

### 5.1 添加新的编排策略

**场景**：支持新的测试编排模式

**步骤**：

1. 在 Dispatcher 中添加新的任务解析逻辑
2. 定义新的配置格式
3. 实现任务生成和结果聚合逻辑

**代码示例**：

```python
class Dispatcher:
    def _parse_tasks(self, config: Config) -> List[Config]:
        """解析配置，生成任务列表"""
        if config["test_type"] == "parameter_sweep":
            return self._generate_sweep_tasks(config)
        elif config["test_type"] == "comparison":
            return self._generate_comparison_tasks(config)
        elif config["test_type"] == "adaptive":  # 新增
            return self._generate_adaptive_tasks(config)  # 新增
        else:
            return [config]

    def _generate_adaptive_tasks(self, config: Config) -> List[Config]:
        """自适应测试：根据前一个任务的结果调整参数"""
        tasks = []
        current_config = config["initial_config"]

        for iteration in range(config["max_iterations"]):
            tasks.append(current_config)

            # 根据预测结果生成下一个配置
            current_config = self._predict_next_config(
                current_config,
                expected_improvement=config["target_improvement"]
            )

        return tasks
```

**配置示例**：

```yaml
# 自适应测试：自动寻找最优参数
test_type: adaptive
max_iterations: 10
initial_config:
  batch_size: 4
  max_tokens: 256
target_improvement: 0.1  # 每次迭代期望提升 10%
```

### 5.2 添加新的 Adapter

**场景**：支持 vLLM 框架

**步骤**：

1. 继承 `BaseAdapter`
2. 使用装饰器注册：`@AdapterFactory.register("vllm")`
3. 实现 `setup`（加载 vLLM 模型）
4. 实现 `process`（执行推理）
5. 实现 `teardown`（释放资源）

**代码示例**：

```python
@AdapterFactory.register("vllm")
class VLLMAdapter(BaseAdapter):
    def setup(self, config: dict = None):
        """加载 vLLM 模型"""
        from vllm import LLM
        self.model = LLM(model=config["model_path"])

    def process(self, request: dict) -> dict:
        """执行推理"""
        output = self.model.generate(request["prompt"])
        return {"text": output}

    def teardown(self):
        """释放资源"""
        del self.model
```

### 5.3 添加新的测试类型

**场景**：支持训练测试

**步骤**：

1. 在 Runner 的 `execute()` 方法中添加新的分支
2. 定义新的配置格式
3. 实现对应的测试逻辑（私有方法）

**代码示例**：

```python
class Runner:
    def execute(self):
        """根据配置分发到不同的测试逻辑"""
        test_type = self.config["test_type"]

        if test_type == "operator":
            self._execute_operator_test()
        elif test_type == "direct_inference":
            self._execute_direct_inference()
        elif test_type == "service_inference":
            self._execute_service_inference()
        elif test_type == "training":  # 新增
            self._execute_training()  # 新增方法

    def _execute_training(self):
        """新增的训练测试逻辑"""
        self.adapter.setup(self.config["model"])
        train_data = self.data_loader.load()
        optimizer = create_optimizer(self.config["optimizer"])

        for epoch in range(self.config["epochs"]):
            for batch in train_data:
                loss = self.adapter.train_step(batch)
                self.metrics["loss"].append(loss)
```

**优势**：
- 添加新测试类型无需创建新的 Runner 类
- 只需在现有 Runner 中添加新的分支方法
- 不影响现有的测试类型

### 5.4 添加新的数据源

**场景**：从数据库加载 prompts

**步骤**：

1. 实现 `DataLoader` 子类
2. 在 `DataLoaderFactory.from_config` 中添加分支
3. 实现 `load()` 方法（连接数据库、查询数据）
4. 返回标准格式（list of prompts）

**代码示例**：

```python
class DatabasePromptLoader(DataLoader):
    def __init__(self, connection_string: str, query: str):
        self.connection_string = connection_string
        self.query = query

    def load(self) -> List[str]:
        import psycopg2
        conn = psycopg2.connect(self.connection_string)
        cursor = conn.cursor()
        cursor.execute(self.query)
        return [row[0] for row in cursor.fetchall()]

# 在工厂中添加
class DataLoaderFactory:
    @staticmethod
    def from_config(config: dict):
        source = config.get("source")

        if source == "file":
            return FilePromptLoader(config["file_path"])
        elif source == "database":  # 新增
            return DatabasePromptLoader(
                config["connection_string"],
                config["query"]
            )
```

## 六、FAQ

**Q1: 为什么不分离 OperatorAdapter 和 ModelAdapter？**
A: 单一基类更简洁。通过可选的 `setup/teardown` 就能支持两种模式，避免过度设计。

**Q2: 为什么使用注册表模式而不是工厂模式？**
A: 注册表模式更简洁、更易扩展。添加新类型只需一行装饰器注册，无需修改工厂代码。

**Q3: 为什么需要 Dispatcher 而不是直接创建 Runner？**
A: Dispatcher 提供测试编排能力，支持参数扫描、框架对比、混合流程等复杂场景。如果只是简单的单次测试，可以跳过 Dispatcher 直接创建 Runner，但使用 Dispatcher 可以获得更好的扩展性。

**Q4: 为什么 Runner 保持通用？**
A: Runner 保持通用可以复用于所有测试类型和框架。框架特定的逻辑全部交给 Adapter，测试类型特定的差异通过配置和私有方法体现。

**Q5: DataLoader 是否必要？**
A: 是的。当前 Runner 中有大量数据准备逻辑（prompts 生成、trace 加载等），提取后可以复用，减少重复代码。

**Q6: Runner 和 Adapter 的 setup/teardown 有什么区别？**
A:
- **Runner**：测试级别，编排流程（创建组件、启动监控、保存结果）
- **Adapter**：框架级别，管理资源（加载模型、分配内存、释放资源）
- Runner 调用 Adapter 的 setup/teardown，但不知道内部细节

**Q7: 如何处理异步测试（如 ServiceInference）？**
A: Runner 可以实现异步的 `execute()` 方法。基类的 `run()` 方法会自动检测并调用 `async_execute()`。

**Q8: 如何支持自定义指标？**
A: Runner 提供 `do_setup` 和 `do_teardown` 钩子，子类可以添加特定的指标收集逻辑。也可以在 `execute` 中收集自定义指标。

## 七、总结

**核心价值**：

1. **统一接口** - 所有测试使用同一套框架
2. **清晰分层** - Dispatcher（编排）→ Runner（执行）→ Adapter（适配）
3. **职责分离** - Dispatcher 管理测试编排，Runner 管理执行流程，Adapter 管理框架适配
4. **强大编排** - 支持参数扫描、框架对比、混合流程等复杂测试场景
5. **框架解耦** - 通过 Adapter 适配不同框架，Runner 保持通用
6. **易于扩展** - 添加新框架、新测试类型、新编排策略都互不影响
7. **配置驱动** - 通过配置文件控制测试行为，无需修改代码

**设计模式总结**：

| 设计模式 | 应用场景 | 优势 |
|---------|---------|------|
| **编排模式** | Dispatcher | 统一管理多个测试任务，支持复杂编排策略 |
| **模板方法模式** | Runner 生命周期 | Runner 保持通用，框架逻辑交给 Adapter |
| **注册表模式** | Adapter 管理 | 添加新框架无需修改代码，消除 if-elif-else |
| **工厂模式** | DataLoader | 统一创建接口，支持多种数据源 |
| **策略模式** | 测试类型选择 | 通过配置而非代码区分测试类型 |
