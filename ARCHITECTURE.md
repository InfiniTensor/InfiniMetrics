# InfiniMetrics 统一测试系统架构设计

## 一、核心设计

### 1.1 架构分层

```
用户层
  ↓
Dispatcher (调度) - 使用注册表模式路由到合适的测试类型
  ↓
Runner (执行) - 使用模板方法模式管理生命周期
  ↓
DataLoader (数据) - 独立的数据准备组件
  ↓
Adapter (接口) - 使用注册表模式管理框架适配器
```

**关键点**：

- **Dispatcher**：使用注册表模式，根据 `test_type` 选择 Runner，避免大量 if-elif-else
- **Runner**：使用模板方法 + 钩子（`before_setup/after_setup`），统一生命周期管理
- **DataLoader**：独立的数据准备组件，支持工厂模式创建
- **Adapter**：使用装饰器注册，统一接口 `process(request)`，可选 `setup/teardown`

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

### 1.3 Runner 生命周期（模板方法模式）

所有 Runner 使用模板方法模式，生命周期分为三个阶段：

```python
class RunnerBase:
    # 模板方法：定义流程骨架（子类不要重写）
    def setup(self):
        """setup 流程：before → do → after"""
        self.before_setup()   # 公共逻辑（基类实现）
        self.do_setup()       # 子类特定逻辑（子类重写）
        self.after_setup()    # 公共逻辑（基类实现）

    def teardown(self):
        """teardown 流程：before → do → after"""
        self.before_teardown()  # 公共逻辑（基类实现）
        self.do_teardown()      # 子类特定逻辑（子类重写）
        self.after_teardown()   # 公共逻辑（基类实现）

    # 钩子方法
    def before_setup(self):
        """公共逻辑：创建 adapter、data_loader、初始化 metrics"""
        self.adapter = AdapterFactory.create(self.config["framework"])
        self.data_loader = DataLoaderFactory.from_config(self.config["data"])
        self.metrics = {}

    @abstractmethod
    def do_setup(self):
        """子类特定逻辑：每个 Runner 自己实现"""
        pass

    def after_setup(self):
        """公共逻辑：启动监控"""
        self.monitor = Monitor()
        self.monitor.start()

    def before_teardown(self):
        """公共逻辑：停止监控、收集指标"""
        if self.monitor:
            self.monitor.stop()

    @abstractmethod
    def do_teardown(self):
        """子类特定逻辑：每个 Runner 自己实现"""
        pass

    def after_teardown(self):
        """公共逻辑：保存结果"""
        self._save_results()
```

**执行顺序**：

```
调用 runner.setup():
  1. before_setup()  ← 创建组件（公共逻辑）
  2. do_setup()      ← 子类特定逻辑
  3. after_setup()   ← 启动监控（公共逻辑）

调用 runner.teardown():
  1. before_teardown()  ← 停止监控（公共逻辑）
  2. do_teardown()      ← 子类特定逻辑
  3. after_teardown()   ← 保存结果（公共逻辑）
```

### 1.4 注册表模式

**Dispatcher 注册表**：

```python
# Runner 注册表
RUNNER_REGISTRY = {
    "operator": OperatorTestRunner,
    "direct_inference": DirectInferenceRunner,
    "service_inference": ServiceInferenceRunner,
}

# 简洁的 Dispatcher
class Dispatcher:
    def dispatch(self, config):
        test_type = config["test_type"]
        return RUNNER_REGISTRY[test_type](config)
```

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

- ✅ 添加新 Runner/Adapter 只需注册，无需修改工厂代码
- ✅ 消除大量 if-elif-else 判断
- ✅ 支持动态注册（运行时注册）
- ✅ 代码简洁，易于维护

## 二、三种测试模式

### 2.1 算子测试（Operator Testing）

**场景**：测试单个算子的性能和正确性

**特点**：

- 单次调用，无状态
- 快速反馈（秒级）
- 关注算子延迟、tensor 正确性

**示例流程**：

```
1. before_setup() - 创建 InfiniCoreAdapter
2. do_setup() - 加载算子配置
3. after_setup() - 启动监控
4. execute() - 调用 adapter.process() 执行算子
5. before_teardown() - 停止监控
6. do_teardown() - 无状态 adapter，无需清理
7. after_teardown() - 保存结果
```

**代码示例**：

```python
class OperatorTestRunner(RunnerBase):
    def do_setup(self):
        """只写自己的逻辑，公共逻辑由基类处理"""
        self.op_config = self.data_loader.load()

    def do_teardown(self):
        """无状态，无需清理"""
        pass
```

### 2.2 直接推理测试（Direct Inference）

**场景**：测试端到端模型推理性能

**特点**：

- 需要加载模型（有状态）
- Warmup + Measurement 两阶段
- 关注 TTFT、TPOT、吞吐量

**示例流程**：

```
1. before_setup() - 创建 InfiniLMAdapter、DataLoader
2. do_setup() - adapter.setup()（加载模型）、加载 prompts
3. after_setup() - 启动监控
4. execute() - Warmup + Measurement 两阶段
5. before_teardown() - 停止监控、收集指标
6. do_teardown() - adapter.teardown()（卸载模型）
7. after_teardown() - 保存结果
```

**代码示例**：

```python
class DirectInferenceRunner(RunnerBase):
    def do_setup(self):
        """只写自己的逻辑"""
        self.adapter.setup(self.config["model"])  # 加载模型
        self.prompts = self.data_loader.load()

    def do_teardown(self):
        """只写自己的逻辑"""
        self.adapter.teardown()  # 卸载模型
```

### 2.3 服务推理测试（Service Inference）

**场景**：测试推理服务的并发性能

**特点**：

- 异步执行，并发请求
- 基于 trace 文件驱动
- 关注 QPS、尾延迟、资源利用率

**示例流程**：

```
1. before_setup() - 创建 DataLoader（不需要 adapter）
2. do_setup() - 启动推理服务、加载 trace
3. after_setup() - 启动监控
4. execute() - 异步并发执行 trace 中的请求
5. before_teardown() - 停止监控
6. do_teardown() - 停止服务
7. after_teardown() - 保存结果
```

**代码示例**：

```python
class ServiceInferenceRunner(RunnerBase):
    def do_setup(self):
        """只写自己的逻辑"""
        self.service = InferenceService(self.config["service"])
        self.service.start()
        self.trace = self.data_loader.load()

    def do_teardown(self):
        """只写自己的逻辑"""
        self.service.stop()
```

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

### 3.3 为什么使用模板方法 + 钩子？

**问题**：如何避免每个 Runner 都重复公共逻辑？

**解决**：模板方法模式 + 钩子方法

**优势**：

- ✅ **消除重复**：公共逻辑只写一次（在基类的 before/after 钩子中）
- ✅ **职责清晰**：公共逻辑 vs 特定逻辑明确分离
- ✅ **强制流程**：保证流程一致性，子类无法忘记公共逻辑
- ✅ **易于维护**：修改公共逻辑只需改基类

**对比**：

```python
# ❌ 改进前：每个 Runner 都重复公共逻辑
class DirectInferenceRunner:
    def setup(self):
        # 公共逻辑（重复）
        self.adapter = create_adapter(...)
        self.data_loader = create_data_loader(...)
        self.monitor = Monitor()
        self.monitor.start()

        # 自己的逻辑
        self.adapter.setup(...)
        self.prompts = load_prompts(...)

class OperatorTestRunner:
    def setup(self):
        # 公共逻辑（重复）
        self.adapter = create_adapter(...)
        self.data_loader = create_data_loader(...)
        self.monitor = Monitor()
        self.monitor.start()

        # 自己的逻辑
        self.op_config = load_config(...)

# ✅ 改进后：公共逻辑在基类，子类只写自己的逻辑
class DirectInferenceRunner(RunnerBase):
    def do_setup(self):
        # 只写自己的逻辑
        self.adapter.setup(...)
        self.prompts = load_prompts(...)
```

### 3.4 Runner vs Adapter 的职责边界？

**Runner**（测试级别 - 流程编排）：

- **职责**：编排测试流程，管理测试级别的东西
- **setup 做什么**：
  - 选择 DataLoader → 加载测试数据
  - 创建 Adapter → 调用 `adapter.setup()`
  - 启动 Monitor → 开始监控资源
  - 初始化 MetricsCollector → 准备收集指标
- **teardown 做什么**：
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
- **process() 做什么**：
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
| **是否必须重写 setup/teardown** | ✅ 必须重写 `do_setup/do_teardown` | ⚠️ 可选重写（有状态/无状态） |
| **是否调用对方** | 调用 `adapter.setup()/teardown()` | 不调用 runner |

**类比**：

```
Runner = 服务员（协调一切）
Adapter = 厨师（做菜）

服务员 (Runner) 的 setup：
  ✅ 准备菜单 (DataLoader)
  ✅ 叫厨师来上班 (adapter.setup())
  ✅ 准备餐桌 (Monitor)
  ✅ 准备账单 (MetricsCollector)

厨师 (Adapter) 的 setup：
  ✅ 穿上厨师服
  ✅ 准备厨具
  ✅ 打开炉灶
  ✅ 准备食材

关键点：
  - 服务员不关心厨师怎么切菜、怎么炒菜
  - 厨师不关心服务员怎么点菜、怎么上菜
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

### 4.1 算子测试

```
Config → Dispatcher (根据 test_type 选择)
           ↓
       OperatorTestRunner
           ↓
       before_setup() ← 创建 InfiniCoreAdapter、DataLoader
           ↓
       do_setup() ← 加载算子配置
           ↓
       after_setup() ← 启动监控
           ↓
       execute() ← adapter.process() 执行算子
           ↓
       before_teardown() ← 停止监控
           ↓
       do_teardown() ← 无状态，无需清理
           ↓
       after_teardown() ← 保存结果
           ↓
       返回结果 (延迟、精度)
```

### 4.2 直接推理测试

```
Config → Dispatcher (根据 test_type 选择)
           ↓
       DirectInferenceRunner
           ↓
       before_setup() ← 创建 InfiniLMAdapter、DataLoader
           ↓
       do_setup() ← adapter.setup() (加载模型)、加载 prompts
           ↓
       after_setup() ← 启动监控
           ↓
       execute()
         ├─ Warmup 阶段 (adapter.process × N)
         └─ Measurement 阶段 (adapter.process × N + 监控)
           ↓
       before_teardown() ← 停止监控、收集指标
           ↓
       do_teardown() ← adapter.teardown() (卸载模型)
           ↓
       after_teardown() ← 保存结果
           ↓
       返回结果 (TTFT, TPOT, 吞吐量等)
```

### 4.3 服务推理测试

```
Config → Dispatcher (根据 test_type 选择)
           ↓
       ServiceInferenceRunner
           ↓
       before_setup() ← 创建 DataLoader (不需要 adapter)
           ↓
       do_setup() ← 启动推理服务、加载 trace
           ↓
       after_setup() ← 启动监控
           ↓
       execute() ← 异步并发执行 trace 中的请求
           ↓
       before_teardown() ← 停止监控
           ↓
       do_teardown() ← 停止服务
           ↓
       after_teardown() ← 保存结果
           ↓
       返回结果 (QPS, 尾延迟等)
```

## 五、扩展性设计

### 5.1 添加新的 Adapter

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

### 5.2 添加新的 Runner

**场景**：支持训练测试

**步骤**：

1. 继承 `RunnerBase`
2. 注册到 `RUNNER_REGISTRY`
3. 实现 `do_setup`（初始化训练环境）
4. 实现 `execute`（运行训练循环，监控 loss）
5. 实现 `do_teardown`（清理训练环境）

**代码示例**：

```python
# 1. 注册到注册表
RUNNER_REGISTRY["training"] = TrainingRunner

# 2. 实现 Runner
class TrainingRunner(RunnerBase):
    def do_setup(self):
        """初始化训练环境"""
        self.adapter.setup(self.config["model"])
        self.train_data = self.data_loader.load()
        self.optimizer = create_optimizer(self.config["optimizer"])

    def execute(self):
        """运行训练循环"""
        for epoch in range(self.config["epochs"]):
            for batch in self.train_data:
                loss = self.adapter.train_step(batch)
                self.metrics["loss"].append(loss)

    def do_teardown(self):
        """清理训练环境"""
        self.adapter.teardown()
```

### 5.3 添加新的数据源

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

**Q3: 为什么使用模板方法 + 钩子？**
A: 消除重复代码，保证流程一致性。公共逻辑在基类的 before/after 钩子中，子类只实现特定逻辑。

**Q4: DataLoader 是否必要？**
A: 是的。当前 Runner 中有大量数据准备逻辑（prompts 生成、trace 加载等），提取后可以复用，减少重复代码。

**Q5: Runner 和 Adapter 的 setup/teardown 有什么区别？**
A:
- **Runner**：测试级别，编排流程（创建组件、启动监控、保存结果）
- **Adapter**：框架级别，管理资源（加载模型、分配内存、释放资源）
- Runner 调用 Adapter 的 setup/teardown，但不知道内部细节

**Q6: 如何处理异步测试（如 ServiceInference）？**
A: Runner 可以实现异步的 `execute()` 方法。基类的 `run()` 方法会自动检测并调用 `async_execute()`。

**Q7: 如何支持自定义指标？**
A: Runner 提供 `do_setup` 和 `do_teardown` 钩子，子类可以添加特定的指标收集逻辑。也可以在 `execute` 中收集自定义指标。

## 七、总结

**核心价值**：

1. **统一接口** - 所有测试使用同一套框架
2. **清晰分层** - 职责明确，易于维护
3. **易于扩展** - 添加新功能无需修改核心代码（注册表模式 + 模板方法）
4. **代码复用** - 通用逻辑在基类中实现（before/after 钩子）
5. **消除重复** - 注册表模式避免大量 if-elif-else，模板方法避免公共逻辑重复

**设计模式总结**：

| 设计模式 | 应用场景 | 优势 |
|---------|---------|------|
| **注册表模式** | Dispatcher、Adapter | 添加新类型无需修改代码，消除 if-elif-else |
| **模板方法模式** | Runner 生命周期 | 公共逻辑统一管理，保证流程一致性 |
| **工厂模式** | DataLoader | 统一创建接口，支持多种数据源 |
| **策略模式** | 不同的测试类型 | 灵活切换测试策略 |
