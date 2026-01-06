# 三个核心问题的解答

## ❓ 问题 1：dispatcher.py 应该怎么写？

### ✅ 答案：使用**工厂模式**，根据配置路由到合适的 Runner

```python
class Dispatcher:
    def dispatch(self, config: dict) -> RunnerBase:
        test_type = config.get("test_type")

        if test_type == "operator":
            return self._create_operator_runner(config)
        elif test_type == "direct_inference":
            return self._create_direct_inference_runner(config)
        elif test_type == "service_inference":
            return self._create_service_inference_runner(config)
        else:
            raise ValueError(f"Unknown test type: {test_type}")

    def _create_direct_inference_runner(self, config):
        runner = DirectInferenceRunner(config)
        runner.adapter = self._create_adapter(config["framework"])
        runner.data_loader = self._create_data_loader(config["data"])
        return runner
```

**核心职责**：
- ✅ 根据 `test_type` 选择 Runner
- ✅ 根据 `framework` 选择 Adapter
- ✅ 根据 `data.source` 选择 DataLoader
- ✅ 提供统一的入口

**使用方式**：
```python
dispatcher = Dispatcher(config)
runner = dispatcher.dispatch()
runner.run()
```

---

## ❓ 问题 2：Runner 的子类 setup 和 teardown 是不是都要重写？

### ✅ 答案：**是的，必须重写！**

因为每个 Runner 的初始化和清理逻辑完全不同：

### **示例对比**

```python
class DirectInferenceRunner(RunnerBase):
    def setup(self):
        # ✅ 需要加载模型
        self.adapter = InfiniLMAdapter()
        self.adapter.setup()  # 加载模型到 GPU
        self.prompts = DataLoader.load_prompts(...)

    def teardown(self):
        # ✅ 需要卸载模型
        if self.adapter:
            self.adapter.teardown()  # 释放 GPU 内存


class ServiceInferenceRunner(RunnerBase):
    def setup(self):
        # ✅ 需要启动服务（不是加载模型）
        self.service = InferenceService(...)
        self.service.start()  # 启动 HTTP/gRPC 服务
        self.trace = DataLoader.load_trace(...)

    def teardown(self):
        # ✅ 需要停止服务（不是卸载模型）
        if self.service:
            self.service.stop()  # 停止服务


class OperatorTestRunner(RunnerBase):
    def setup(self):
        # ✅ 不需要加载模型（无状态）
        self.adapter = InfiniCoreAdapter()
        # 注意：不需要调用 adapter.setup()
        self.op_config = DataLoader.load_operator_config(...)

    def teardown(self):
        # ✅ 不需要清理（无状态）
        # 注意：不需要调用 adapter.teardown()
        pass
```

**为什么必须重写？**

| Runner 类型 | setup 逻辑 | teardown 逻辑 | 是否相同？ |
|------------|-----------|--------------|----------|
| DirectInference | 加载模型到 GPU | 卸载模型，释放内存 | ❌ 不同 |
| ServiceInference | 启动推理服务 | 停止服务 | ❌ 不同 |
| OperatorTest | 加载算子配置 | 什么都不做 | ❌ 不同 |

**结论**：每个 Runner 的 setup/teardown 逻辑都不同，必须重写！

---

## ❓ 问题 3：Runner 的 setup/teardown 和 Adapter 的有什么区别？

### ✅ 答案：**职责不同，层级不同**

### **职责对比表**

| | **Runner** | **Adapter** |
|---|---|---|
| **抽象层级** | 测试级别（Test Level） | 框架级别（Framework Level） |
| **核心职责** | 流程编排 | 资源管理 |
| **setup 做什么** | - 选择 DataLoader<br>- 创建 Adapter<br>- 启动 Monitor<br>- 调用 adapter.setup() | - 加载模型到 GPU<br>- 初始化 tokenizer<br>- 分配内存<br>- 框架特定的初始化 |
| **teardown 做什么** | - 停止 Monitor<br>- 调用 adapter.teardown()<br>- 保存结果<br>- 清理临时文件 | - 卸载模型<br>- 释放 GPU 内存<br>- 清理缓存<br>- 关闭连接 |
| **是否调用对方** | 调用 adapter.setup()/teardown() | 不调用 runner |
| **是否必须重写** | ✅ 必须重写（每个 Runner 不同） | ⚠️ 可选重写（有状态/无状态） |

### **代码示例：清晰的职责分离**

```python
# ============================================
# Runner 的 setup/teardown
# ============================================

class DirectInferenceRunner(RunnerBase):
    def setup(self):
        """
        ✅ 职责：编排测试流程（测试级别）
        """
        # 1. 数据准备（选择 DataLoader）
        data_loader = DataLoader.from_config(self.config["data"])
        self.prompts = data_loader.load()

        # 2. 创建 Adapter（框架细节封装）
        self.adapter = InfiniLMAdapter(self.config["model"])

        # 3. 调用 Adapter 的 setup（让 adapter 管理框架资源）
        self.adapter.setup(self.config["model"])

        # 4. 启动监控（测试级别）
        self.monitor = Monitor()
        self.monitor.start()

    def teardown(self):
        """
        ✅ 职责：清理测试资源（测试级别）
        """
        # 1. 停止监控
        self.monitor.stop()

        # 2. 调用 Adapter 的 teardown（让 adapter 清理框架资源）
        if self.adapter:
            self.adapter.teardown()

        # 3. 保存结果
        self.save_results()


# ============================================
# Adapter 的 setup/teardown
# ============================================

class InfiniLMAdapter(BaseAdapter):
    def setup(self, config: dict = None):
        """
        ✅ 职责：管理框架资源（框架级别）
        """
        # 1. 加载模型（框架细节）
        self.model = JiugeForCausalLM.from_pretrained(
            config["model_path"]
        )

        # 2. 初始化 tokenizer（框架细节）
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["tokenizer_path"]
        )

        # 3. 分配 GPU（框架细节）
        import torch
        self.device = torch.device("cuda")
        self.model.to(self.device)

    def process(self, request: dict) -> dict:
        """执行推理"""
        inputs = self.tokenizer(request["prompt"], return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids.to(self.device),
            max_length=request.get("max_tokens", 100)
        )
        return {"text": self.tokenizer.decode(outputs[0])}

    def teardown(self):
        """
        ✅ 职责：清理框架资源（框架级别）
        """
        # 1. 清理 GPU 缓存（框架细节）
        import torch
        torch.cuda.empty_cache()

        # 2. 删除模型（框架细节）
        del self.model
        del self.tokenizer


# ============================================
# 无状态 Adapter：不需要 setup/teardown
# ============================================

class InfiniCoreAdapter(BaseAdapter):
    def setup(self, config: dict = None):
        """无状态，不需要加载模型"""
        pass  # ← 什么都不做

    def process(self, request: dict) -> dict:
        """直接调用算子（无状态）"""
        return InfiniCore.run_operator(
            request["operator"],
            request["inputs"]
        )

    def teardown(self):
        """无状态，不需要清理"""
        pass  # ← 什么都不做
```

### **类比说明**

**餐厅服务类比**：

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

服务员 (Runner) 的 teardown：
  ✅ 叫厨师下班 (adapter.teardown())
  ✅ 收拾餐桌
  ✅ 结账
  ✅ 打扫卫生

厨师 (Adapter) 的 teardown：
  ✅ 关闭炉灶
  ✅ 清洗厨具
  ✅ 脱下厨师服
  ✅ 清理厨房

关键点：
  - 服务员不关心厨师怎么切菜、怎么炒菜
  - 厨师不关心服务员怎么点菜、怎么上菜
  - 各司其职，通过接口交互
```

---

## 📊 总结对比图

```
┌─────────────────────────────────────────────────────────────┐
│                    Dispatcher (统一入口)                      │
│                   dispatch(config) → Runner                  │
└─────────────────────────────┬───────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
        ┌───────▼──────┐ ┌───▼─────┐ ┌────▼────────┐
        │  Operator    │ │ Direct  │ │  Service    │
        │  TestRunner  │ │Inference│ │  Inference  │
        │              │ │ Runner  │ │  Runner     │
        ├──────────────┤ ├─────────┤ ├─────────────┤
        │ setup()      │ │setup()  │ │ setup()     │
        │ - 加载配置   │ │-加载模型│ │ - 启动服务  │
        │ - 创建adapter│ │-加载prompts│ - 加载trace│
        │ (必须重写)   │ │ (必须重写)│ │ (必须重写)  │
        └──────┬───────┘ └───┬─────┘ └────┬────────┘
               │             │            │
        ┌──────▼─────────────▼────────────▼────────┐
        │           Adapter (框架接口)              │
        ├──────────────────────────────────────────┤
        │ setup() (可选重写)                       │
        │  - 有状态：加载模型 (InfiniLM)           │
        │  - 无状态：pass (InfiniCore)            │
        │                                          │
        │ process() (必须实现)                     │
        │  - 执行算子/推理                         │
        │                                          │
        │ teardown() (可选重写)                    │
        │  - 有状态：卸载模型 (InfiniLM)           │
        │  - 无状态：pass (InfiniCore)            │
        └──────────────────────────────────────────┘
```

---

## ✅ 关键要点

### **1. Dispatcher**
- 使用**工厂模式**，根据配置创建 Runner
- 不关心测试流程，只负责路由

### **2. Runner 的 setup/teardown**
- ✅ **必须重写**（每个 Runner 逻辑不同）
- 职责：编排测试流程（测试级别）
- 调用 adapter 的 setup/teardown

### **3. Adapter 的 setup/teardown**
- ⚠️ **可选重写**（有状态 vs 无状态）
- 职责：管理框架资源（框架级别）
- 不调用 runner 的任何方法

### **4. 职责分离**
```
Dispatcher → "选谁做" (选择 Runner)
Runner      → "怎么做" (编排流程)
Adapter     → "具体做" (执行操作)
DataLoader  → "用什么数据" (准备数据)
```

---

**创建时间**: 2026-01-06
**作者**: Claude (Sonnet 4.5)
