"""
Runner vs Adapter 的 setup/teardown 职责对比

核心区别：
- Runner：测试流程编排（管理测试级别的东西）
- Adapter：框架资源管理（管理框架级别的东西）
"""

# ============================================================
# 一、职责对比表
# ============================================================

"""
┌─────────────────────────────────────────────────────────────┐
│                  Runner 的 setup/teardown                    │
├─────────────────────────────────────────────────────────────┤
│ 职责：编排测试流程（测试级别）                                 │
├─────────────────────────────────────────────────────────────┤
│ setup() 做什么：                                              │
│   1. 选择 DataLoader → 加载测试数据 (prompts/trace)          │
│   2. 创建 Adapter → 调用 adapter.setup()                     │
│   3. 启动 Monitor → 开始监控资源                             │
│   4. 初始化 MetricsCollector → 准备收集指标                  │
│                                                              │
│ teardown() 做什么：                                           │
│   1. 调用 adapter.teardown() → 清理框架资源                  │
│   2. 停止 Monitor → 停止监控                                 │
│   3. 保存结果 → 写入文件/数据库                               │
│   4. 清理临时文件 → 测试产生的临时数据                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  Adapter 的 setup/teardown                   │
├─────────────────────────────────────────────────────────────┤
│ 职责：管理框架资源（框架级别）                                 │
├─────────────────────────────────────────────────────────────┤
│ setup(config) 做什么：                                        │
│   1. 加载模型 → 从磁盘加载到内存/GPU                         │
│   2. 初始化 tokenizer → 文本编码器                           │
│   3. 分配 GPU 内存 → 为模型预留空间                          │
│   4. 编译/优化模型 → 框架特定的优化                           │
│                                                              │
│ teardown() 做什么：                                           │
│   1. 卸载模型 → 释放 GPU 内存                               │
│   2. 关闭连接 → 数据库/网络连接                              │
│   3. 清理缓存 → KV cache 等                                  │
│   4. 重置状态 → 恢复到初始状态                                │
└─────────────────────────────────────────────────────────────┘
"""

# ============================================================
# 二、代码对比：清晰的职责分离
# ============================================================

# ❌ 错误示例：职责混乱（Runner 管理框架细节）
class DirectInferRunner_WRONG:
    def setup(self):
        # ❌ 问题：直接操作框架细节
        self.model = JiugeForCausalLM.from_pretrained(
            self.config["model_path"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["tokenizer_path"]
        )

        # ❌ 问题：手动管理 GPU
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def teardown(self):
        # ❌ 问题：手动清理框架资源
        import torch
        torch.cuda.empty_cache()
        del self.model
        del self.tokenizer


# ✅ 正确示例：职责清晰（Runner 编排，Adapter 管理）
class DirectInferenceRunner_RIGHT(RunnerBase):
    def setup(self):
        """
        ✅ Runner 的 setup：只做编排，不关心框架细节
        """
        # 1. 数据准备（使用 DataLoader）
        data_loader = DataLoaderFactory.from_config(self.config["data"])
        self.prompts = data_loader.load()

        # 2. 创建 Adapter（框架细节封装）
        self.adapter = InfiniLMAdapter(self.config["model"])

        # 3. 调用 Adapter 的 setup（让 adapter 管理框架资源）
        self.adapter.setup(self.config["model"])

        # 4. 启动监控
        self.monitor = Monitor()
        self.monitor.start()

    def teardown(self):
        """
        ✅ Runner 的 teardown：只做编排，不关心框架细节
        """
        # 1. 停止监控
        self.monitor.stop()

        # 2. 调用 Adapter 的 teardown（让 adapter 清理框架资源）
        if self.adapter:
            self.adapter.teardown()

        # 3. 保存结果
        self.save_results()


class InfiniLMAdapter_RIGHT(BaseAdapter):
    """
    ✅ Adapter 的 setup/teardown：管理框架资源
    """
    def setup(self, config: dict = None):
        """
        ✅ Adapter 的 setup：管理框架级别的资源
        """
        # 1. 加载模型（框架细节）
        from infini_lm import JiugeForCausalLM
        self.model = JiugeForCausalLM.from_pretrained(
            config["model_path"]
        )

        # 2. 初始化 tokenizer（框架细节）
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["tokenizer_path"]
        )

        # 3. 分配 GPU（框架细节）
        import torch
        self.device = torch.device("cuda")
        self.model.to(self.device)

    def process(self, request: dict) -> dict:
        """
        ✅ Adapter 的 process：执行推理
        """
        prompt = request["prompt"]
        max_tokens = request.get("max_tokens", 100)

        # 框架特定的推理逻辑
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids.to(self.device),
            max_length=max_tokens
        )

        return {
            "text": self.tokenizer.decode(outputs[0]),
            "tokens": outputs[0].tolist()
        }

    def teardown(self):
        """
        ✅ Adapter 的 teardown：清理框架级别的资源
        """
        # 1. 清理 GPU 缓存（框架细节）
        import torch
        torch.cuda.empty_cache()

        # 2. 删除模型（框架细节）
        del self.model
        del self.tokenizer


# ============================================================
# 三、类比说明
# ============================================================

"""
类比：餐厅服务

Runner = 服务员（协调一切）
Adapter = 厨师（做菜）

服务员 (Runner) 的 setup：
  - 准备菜单 (DataLoader)
  - 叫厨师来上班 (adapter.setup())
  - 准备餐桌 (Monitor)
  - 准备账单 (MetricsCollector)

厨师 (Adapter) 的 setup：
  - 穿上厨师服
  - 准备厨具
  - 打开炉灶
  - 准备食材

服务员 (Runner) 的 teardown：
  - 叫厨师下班 (adapter.teardown())
  - 收拾餐桌
  - 结账
  - 打扫卫生

厨师 (Adapter) 的 teardown：
  - 关闭炉灶
  - 清洗厨具
  - 脱下厨师服
  - 清理厨房

关键点：
  - 服务员不关心厨师怎么切菜、怎么炒菜
  - 厨师不关心服务员怎么点菜、怎么上菜
  - 各司其职，通过接口交互
"""


# ============================================================
# 四、不同的 Runner，不同的 setup/teardown
# ============================================================

class DirectInferenceRunner(RunnerBase):
    """
    直接推理测试
    特点：需要加载模型到内存
    """
    def setup(self):
        # ✅ 1. 加载 prompts
        self.prompts = DataLoader.load_prompts(self.config["data"])

        # ✅ 2. 创建并加载模型（有状态 adapter）
        self.adapter = InfiniLMAdapter(self.config["model"])
        self.adapter.setup(self.config["model"])  # ← 关键：加载模型

        # ✅ 3. 启动监控
        self.monitor = Monitor()
        self.monitor.start()

    def execute(self):
        # Warmup + Measurement
        pass

    def teardown(self):
        # ✅ 1. 停止监控
        self.monitor.stop()

        # ✅ 2. 卸载模型（释放 GPU 内存）
        if self.adapter:
            self.adapter.teardown()  # ← 关键：卸载模型


class ServiceInferenceRunner(RunnerBase):
    """
    服务推理测试
    特点：需要启动推理服务
    """
    def setup(self):
        # ✅ 1. 加载 trace
        self.trace = DataLoader.load_trace(self.config["data"])

        # ✅ 2. 启动推理服务（不是直接加载模型）
        self.service = InferenceService(self.config["service"])
        self.service.start()  # ← 关键：启动服务

        # ✅ 3. 创建 adapter（连接到服务）
        self.adapter = ServiceAdapter(self.service)

        # ✅ 4. 启动监控
        self.monitor = ServiceMonitor()
        self.monitor.start()

    async def execute(self):
        # 异步并发执行 trace
        pass

    def teardown(self):
        # ✅ 1. 停止监控
        self.monitor.stop()

        # ✅ 2. 停止服务（不是卸载模型）
        if self.service:
            self.service.stop()  # ← 关键：停止服务


class OperatorTestRunner(RunnerBase):
    """
    算子测试
    特点：不需要加载模型，无状态
    """
    def setup(self):
        # ✅ 1. 加载算子配置
        self.op_config = DataLoader.load_operator_config(
            self.config["data"]
        )

        # ✅ 2. 创建 adapter（无状态，不需要 setup）
        self.adapter = InfiniCoreAdapter()
        # ← 注意：不需要调用 adapter.setup()

        # ✅ 3. 启动监控
        self.monitor = Monitor()
        self.monitor.start()

    def execute(self):
        # 单次调用算子
        pass

    def teardown(self):
        # ✅ 1. 停止监控
        self.monitor.stop()

        # ✅ 2. adapter 无状态，不需要 teardown
        # ← 注意：不需要调用 adapter.teardown()


# ============================================================
# 五、不同的 Adapter，不同的 setup/teardown
# ============================================================

class InfiniLMAdapter(BaseAdapter):
    """
    有状态 Adapter：需要加载模型
    """
    def setup(self, config: dict = None):
        """加载模型到 GPU"""
        self.model = load_model(config["model_path"])
        self.tokenizer = load_tokenizer(config["tokenizer_path"])

    def process(self, request: dict) -> dict:
        """使用加载好的模型推理"""
        return self.model.infer(request["prompt"])

    def teardown(self):
        """释放 GPU 内存"""
        del self.model
        torch.cuda.empty_cache()


class InfiniCoreAdapter(BaseAdapter):
    """
    无状态 Adapter：不需要加载模型
    """
    def setup(self, config: dict = None):
        """什么都不做"""
        pass  # ← 关键：无状态，不需要 setup

    def process(self, request: dict) -> dict:
        """直接调用算子"""
        return InfiniCore.run_operator(request["operator"], request["inputs"])

    def teardown(self):
        """什么都不做"""
        pass  # ← 关键：无状态，不需要 teardown


# ============================================================
# 六、总结
# ============================================================

"""
┌─────────────────┬───────────────────┬───────────────────┐
│      层级        │   Runner          │    Adapter        │
├─────────────────┼───────────────────┼───────────────────┤
│ 抽象级别        │ 测试级别          │ 框架级别          │
│ 主要职责        │ 流程编排          │ 资源管理          │
│                 │ - 选择组件        │ - 加载模型        │
│                 │ - 协调交互        │ - 初始化框架      │
│                 │ - 管理测试流程    │ - 执行操作        │
│                 │ - 收集指标        │ - 清理资源        │
├─────────────────┼───────────────────┼───────────────────┤
│ setup()         │ 必须重写          │ 可选重写          │
│                 │ - 每个 Runner     │ - 有状态：重写    │
│                 │   逻辑不同        │ - 无状态：pass    │
├─────────────────┼───────────────────┼───────────────────┤
│ teardown()      │ 必须重写          │ 可选重写          │
│                 │ - 每个 Runner     │ - 有状态：重写    │
│                 │   逻辑不同        │ - 无状态：pass    │
├─────────────────┼───────────────────┼───────────────────┤
│ 是否调用对方    │ 调用 adapter      │ 不调用 runner     │
│ 的方法          │ .setup()          │                   │
│                 │ .teardown()       │                   │
└─────────────────┴───────────────────┴───────────────────┘

关键原则：
  1. Runner 编排一切，但不懂细节
  2. Adapter 懂细节，但不关心流程
  3. Runner 通过统一的接口调用 Adapter
  4. 职责清晰，易于维护和测试
"""
