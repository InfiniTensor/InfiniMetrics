# Runner/Adapter 重构设计方案

## 📊 问题 1：Runner 需要基类和子类吗？

### ✅ 答案：需要，但要用**模板方法模式**优雅实现

```
┌─────────────────────────────────────────────────────────────┐
│                    RunnerBase (模板方法)                      │
├─────────────────────────────────────────────────────────────┤
│  run() {              ← 模板方法：定义流程骨架（不要重写）      │
│    setup()                                                   │
│    execute()           ← 抽象方法：子类必须实现               │
│    collect_metrics()   ← 钩子方法：子类可选重写              │
│    teardown()                                                 │
│  }                                                           │
│                                                              │
│  _collect_common_metrics()  ← 通用方法（所有 Runner 共享）    │
│  save_results()                                              │
└─────────────────────────────────────────────────────────────┘
           ▲                    ▲                    ▲
           │                    │                    │
    ┌──────┴──────┐      ┌─────┴─────┐      ┌──────┴──────┐
    │ Operator    │      │ Direct    │      │ Service     │
    │ TestRunner  │      │ Inference │      │ Inference   │
    │             │      │ Runner    │      │ Runner      │
    ├─────────────┤      ├───────────┤      ├─────────────┤
    │ execute()   │      │ execute() │      │ execute()   │
    │ - 单次调用  │      │ - warmup  │      │ - 异步执行   │
    │ - 快速测试  │      │ - measure │      │ - trace驱动  │
    └─────────────┘      └───────────┘      └─────────────┘
```

**关键优势：**
- ✅ **共享流程**：生命周期管理、指标收集、结果保存
- ✅ **灵活实现**：每个子类实现自己的 `execute()` 逻辑
- ✅ **易于扩展**：添加新 Runner 只需实现 3 个方法

---

## 📊 问题 2：如何写得优雅？

### 💡 核心原则：**职责分离**

```
┌─────────────────────────────────────────────────────────────┐
│                      清晰的职责边界                           │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Dispatcher   │───→│   Runner     │───→│   Adapter    │
│ (路由分发)   │    │  (流程编排)  │    │  (框架交互)  │
├──────────────┤    ├──────────────┤    ├──────────────┤
│ dispatch()   │    │ setup()      │    │ setup()      │
│ - 选择Runner │    │ execute()    │    │ process()    │
│ - 选择Adapter│    │ teardown()   │    │ teardown()   │
└──────────────┘    └──────────────┘    └──────────────┘
       ↓                    ↓                    ↓
┌──────────────┐    ┌──────────────┐
│ DataLoader   │    │   Monitor    │
│ (数据准备)   │    │  (监控管理)  │
├──────────────┤    ├──────────────┤
│ load()       │    │ start()      │
│ - 文件加载   │    │ stop()       │
│ - 数据生成   │    │ collect()    │
└──────────────┘    └──────────────┘
```

---

## 📊 问题 3：现有代码如何归类？

### 🔄 重构对照表

#### **Runner 层的方法归类**

| 当前方法 | 位置 | 重构后位置 | 理由 |
|---------|------|-----------|------|
| `setup()` | DirectInferRunner | ✅ 保留在 Runner | 流程编排 |
| `execute()` | DirectInferRunner | ✅ 保留在 Runner | 流程编排 |
| `teardown()` | DirectInferRunner | ✅ 保留在 Runner | 流程编排 |
| `collect_metrics()` | DirectInferRunner | ✅ 保留/移到基类 | 通用逻辑 |
| `_load_prompts()` | DirectInferRunner | ❌ 移到 DataLoader | 数据准备 |
| `_load_trace()` | ServiceInferRunner | ❌ 移到 DataLoader | 数据准备 |
| `_generate_prompts()` | DirectInferRunner | ❌ 移到 DataLoader | 数据准备 |
| `_start_monitoring()` | DirectInferRunner | ⚠️ 移到 Monitor 或保留 | 可选独立 |
| `_inference()` | DirectInferRunner | ❌ 移到 Adapter | 框架交互 |
| `_calculate_ttft()` | DirectInferRunner | ✅ 保留在 Runner | 指标计算 |

#### **Adapter 层的方法归类**

| 当前方法 | 重构后 | 说明 |
|---------|--------|------|
| `generate()` | ❌ 改为 `process()` | 统一接口 |
| `load_model()` | ✅ 保留为 `setup()` | 有状态 adapter |
| `unload_model()` | ✅ 保留为 `teardown()` | 资源清理 |
| `calculate_perplexity()` | ✅ 保留 | 可选方法 |

---

## 🎯 具体重构示例

### **场景 1：DirectInferRunner 重构**

#### ❌ 重构前（职责混乱）
```python
class DirectInferRunner(InferRunnerBase):
    def setup(self):
        # 问题1：数据准备混在 Runner 中
        self.prompts = self._load_prompts_from_file()

        # 问题2：框架细节暴露
        self.model = JiugeForCausalLM.from_pretrained(...)
        self.tokenizer = AutoTokenizer.from_pretrained(...)

    def execute(self):
        # 问题3：推理逻辑混在流程中
        for prompt in self.prompts:
            input_ids = self.tokenizer.encode(prompt)
            output = self.model.generate(input_ids, ...)
            self._record_ttft(output)

    def _load_prompts_from_file(self):
        # 问题4：数据加载逻辑分散
        with open(self.config["prompt_file"]) as f:
            return json.load(f)
```

#### ✅ 重构后（职责清晰）
```python
class DirectInferenceRunner(RunnerBase):
    def setup(self):
        # ✅ 1. 使用 DataLoader（数据准备独立）
        data_loader = DataLoader.from_config(self.config["data"])
        self.prompts = data_loader.load()

        # ✅ 2. 使用 Adapter（框架细节封装）
        self.adapter = InfiniLMAdapter(self.config["model"])
        self.adapter.setup()  # 模型加载在 adapter 内部

        # ✅ 3. 启动监控（可选：使用 Monitor）
        self.monitor = Monitor()
        self.monitor.start()

    def execute(self):
        # ✅ Warmup + Measurement（清晰的流程）
        for i in range(self.config["warmup_iters"]):
            self._run_single_inference(warmup=True)

        for i in range(self.config["measure_iters"]):
            self._run_single_inference(warmup=False)

    def _run_single_inference(self, warmup: bool):
        # ✅ 调用 adapter（统一接口）
        result = self.adapter.process({
            "prompt": self.prompts[i],
            "max_tokens": self.config["max_tokens"]
        })

        # ✅ 记录指标
        if not warmup:
            self.record_metrics(result)
```

---

### **场景 2：Adapter 统一接口**

#### ❌ 重构前
```python
class InfiniLMAdapter(InferAdapter):
    def generate(self, prompt: str, max_length: int):
        # 旧的接口名
        pass

class vLLMAdapter(InferAdapter):
    def infer(self, prompt: str, max_tokens: int):
        # 不同的接口名
        pass
```

#### ✅ 重构后
```python
class InfiniLMAdapter(BaseAdapter):
    def process(self, request: dict) -> dict:
        """统一接口"""
        prompt = request["prompt"]
        max_tokens = request.get("max_tokens", 100)

        # 调用原来的逻辑
        output = self._generate_impl(prompt, max_tokens)

        return {
            "text": output.text,
            "tokens": output.tokens,
            "metadata": {...}
        }

    def setup(self, config: dict = None):
        """加载模型"""
        self.model = self._load_model(config)

    def teardown(self):
        """清理资源"""
        if self.model:
            self.model.unload()

class vLLMAdapter(BaseAdapter):
    def process(self, request: dict) -> dict:
        """统一接口"""
        # vLLM 的实现
        pass
```

---

### **场景 3：DataLoader 提取**

#### ❌ 重构前（在 Runner 中）
```python
class DirectInferRunner(InferRunnerBase):
    def _load_prompts(self):
        if self.config.get("source") == "file":
            with open(self.config["file"]) as f:
                return json.load(f)
        elif self.config.get("source") == "generate":
            return [f"Prompt {i}" for i in range(100)]
```

#### ✅ 重构后（独立的 DataLoader）
```python
# 1. 基类
class DataLoader(ABC):
    @abstractmethod
    def load(self) -> List[Any]:
        pass

# 2. 具体实现
class FilePromptLoader(DataLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[str]:
        with open(self.file_path) as f:
            return json.load(f)

class GeneratePromptLoader(DataLoader):
    def __init__(self, count: int, template: str):
        self.count = count
        self.template = template

    def load(self) -> List[str]:
        return [self.template.format(i=i) for i in range(self.count)]

# 3. 工厂
class DataLoaderFactory:
    @staticmethod
    def from_config(config: dict) -> DataLoader:
        if config["source"] == "file":
            return FilePromptLoader(config["file_path"])
        elif config["source"] == "generate":
            return GeneratePromptLoader(config["count"], config["template"])

# 4. 使用
runner = DirectInferenceRunner(config)
data_loader = DataLoaderFactory.from_config(config["data"])
prompts = data_loader.load()
```

---

## 🎨 最终架构图

```
┌──────────────────────────────────────────────────────────────┐
│                         Dispatcher                            │
│                      (统一调度入口)                             │
└───────────────────────────┬──────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
        ┌─────▼─────┐ ┌────▼────┐ ┌─────▼─────┐
        │  Operator │ │ Direct  │ │  Service  │
        │   Runner  │ │Inference│ │ Inference │
        └─────┬─────┘ └────┬────┘ └─────┬─────┘
              │             │             │
              │        ┌────▼────┐       │
              │        │DataLoader│      │
              │        │(prompts) │      │
              │        └────┬────┘       │
              │             │             │
        ┌─────▼─────┐ ┌────▼────┐ ┌─────▼─────┐
        │InfiniCore │ │InfiniLM │ │ vLLM/...  │
        │  Adapter  │ │ Adapter │ │  Adapter  │
        └───────────┘ └─────────┘ └───────────┘
```

---

## ✅ 重构收益

### 1. **代码清晰度**
- ❌ 重构前：Runner 包含数据、流程、框架交互，职责混乱
- ✅ 重构后：每个类职责单一，代码清晰

### 2. **可维护性**
- ❌ 重构前：修改数据加载逻辑需要改 Runner
- ✅ 重构后：修改 DataLoader 不影响 Runner

### 3. **可扩展性**
- ❌ 重构前：添加新框架需要修改多处
- ✅ 重构后：添加新 Adapter 只需实现 `process()`

### 4. **可测试性**
- ❌ 重构前：测试 Runner 需要 mock 框架细节
- ✅ 重构后：可以独立测试 DataLoader、Adapter、Runner

---

## 🚀 实施建议

### **渐进式重构步骤**

**Phase 1：提取 DataLoader**
```
1. 创建 DataLoader 基类和具体实现
2. 在现有 Runner 中使用 DataLoader
3. 删除 Runner 中的数据加载代码
```

**Phase 2：统一 Adapter 接口**
```
1. 将 generate() 重命名为 process()
2. 确保所有 Adapter 实现相同接口
3. 添加 setup()/teardown() 支持有状态
```

**Phase 3：重构 Runner**
```
1. 创建 RunnerBase 基类（模板方法）
2. 将现有 Runner 改为继承 RunnerBase
3. 移除框架交互代码，调用 adapter.process()
4. 移除监控代码，使用 Monitor 类
```

**Phase 4：添加 Dispatcher**
```
1. 创建 Dispatcher 类
2. 根据配置路由到合适的 Runner
3. 统一入口：dispatcher.dispatch(config)
```

### **优先级建议**
```
高优先级：
  ✅ 提取 DataLoader（复用性高，影响大）
  ✅ 统一 Adapter 接口（设计文档要求）

中优先级：
  ⚠️ 重构 Runner 基类（需要调整现有代码）

低优先级：
  💡 添加 Dispatcher（可以最后实现）
  💡 提取 Monitor（可选，视情况而定）
```

---

## 📝 总结

### **问题 1：需要基类和子类吗？**
- ✅ 需要，用**模板方法模式**优雅实现
- ✅ 共享流程在基类，差异逻辑在子类

### **问题 2：如何写得优雅？**
- ✅ **职责分离**：Dispatcher → Runner → DataLoader → Adapter
- ✅ 每个类只做一件事，做好一件事

### **问题 3：方法归类规则**
| 方法类型 | 应该在哪里 |
|---------|-----------|
| 流程编排 | Runner (setup/execute/teardown) |
| 框架交互 | Adapter (process/setup/teardown) |
| 数据加载 | DataLoader (load) |
| 指标收集 | Runner (collect_metrics) |
| 路由分发 | Dispatcher (dispatch) |

---

**创建时间**: 2026-01-06
**作者**: Claude (Sonnet 4.5)
