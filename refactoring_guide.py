"""
重构指南：现有代码如何归类到新架构

场景：将现有的 InfiniLM Runner/Adapter 重构到设计文档的架构
"""

# ============================================================
# 一、Runner 层的重构：方法归类
# ============================================================

"""
当前问题：Runner 中有哪些类型的方法？

1. ✅ 流程编排方法 → 保留在 Runner
   - setup(), execute(), teardown()
   - collect_metrics()

2. ❌ 数据准备方法 → 移到 DataLoader
   - _load_prompts()
   - _load_trace()
   - _generate_prompts()

3. ❌ 框架交互方法 → 移到 Adapter
   - _load_model()
   - _inference()
   - _calculate_perplexity()

4. ✅ 监控方法 → 保留在 Runner 或提取 Monitor
   - _start_monitoring()
   - _stop_monitoring()

5. ✅ 指标计算方法 → 保留在 Runner
   - _calculate_ttft()
   - _calculate_throughput()
"""

# ============================================================
# 示例：重构前的 DirectInferRunner
# ============================================================

class DirectInferRunner_BEFORE(InferRunnerBase):
    """
    重构前：职责不清晰
    """

    def setup(self):
        # ❌ 问题1：数据准备逻辑混在 Runner 中
        self.prompts = []
        if self.config.get("prompt_source") == "file":
            with open(self.config["prompt_file"]) as f:
                self.prompts = json.load(f)
        elif self.config.get("prompt_source") == "generate":
            self.prompts = self._generate_random_prompts(100)

        # ❌ 问题2：直接操作框架细节
        self.model = JiugeForCausalLM.from_pretrained(
            self.config["model_path"]
        )

        # ✅ OK：流程控制
        self._start_monitoring()

    def execute(self):
        # ❌ 问题3：测试流程和推理逻辑混在一起
        for prompt in self.prompts:
            # 框架细节
            input_ids = self.tokenizer.encode(prompt)
            output = self.model.generate(
                input_ids,
                max_length=self.config["max_length"],
                temperature=self.config["temperature"]
            )

            # 指标记录
            self._record_ttft(output)
            self._record_tppt(output)


# ============================================================
# 示例：重构后的 DirectInferenceRunner
# ============================================================

from infinimetrics.common.data_loader import DataLoader
from infinimetrics.inference.adapters import InfiniLMAdapter

class DirectInferenceRunner_AFTER(RunnerBase):
    """
    重构后：职责清晰
    """

    def setup(self):
        """
        只做流程编排，不关心数据如何加载、模型如何加载
        """
        # ✅ 1. 使用 DataLoader 加载数据
        self.data_loader = DataLoader.from_config(self.config["data_config"])
        self.prompts = self.data_loader.load()

        # ✅ 2. 使用 Adapter（框架细节封装）
        self.adapter = InfiniLMAdapter(self.config["model_config"])
        self.adapter.load_model()  # 框架细节在 adapter 内部

        # ✅ 3. 启动监控
        self._start_monitoring()

    def execute(self):
        """
        只做流程编排，不关心推理细节
        """
        # Warmup
        for i in range(self.config["warmup_iterations"]):
            self._run_inference(warmup=True)

        # Measurement
        for i in range(self.config["measure_iterations"]):
            self._run_inference(warmup=False)

    def _run_inference(self, warmup: bool):
        """
        ✅ 单次推理：只关注流程，不关心框架细节
        """
        prompt = self.prompts[i % len(self.prompts)]

        # 调用 adapter（统一接口）
        result = self.adapter.generate(
            prompt,
            max_length=self.config["max_length"]
        )

        # 记录指标
        if not warmup:
            self._record_metrics(result)


# ============================================================
# 二、Adapter 层的重构：统一接口
# ============================================================

"""
当前问题：
- InfiniLMAdapter 有 generate() 方法
- 设计文档要求 process() 方法

解决方案：使用别名或适配器模式
"""

# ============================================================
# 方案A：别名（推荐，简单）
# ============================================================

class InfiniLMAdapter(BaseAdapter):
    """
    统一接口：使用 process() 方法
    """

    def setup(self, config: dict = None):
        """有状态 adapter：需要加载模型"""
        self.model = self._load_model(config)

    def process(self, request: dict) -> dict:
        """
        统一接口：process()
        内部可以调用原来的 generate() 逻辑
        """
        prompt = request["prompt"]
        max_tokens = request.get("max_tokens", 100)

        # 原来的推理逻辑
        output = self.model.generate(prompt, max_tokens=max_tokens)

        return {
            "text": output.text,
            "tokens": output.tokens,
            "ttft": output.ttft,
            "tpot": output.tpot
        }

    def teardown(self):
        """清理资源"""
        if self.model:
            self.model.unload()

    # ====================================
    # 内部辅助方法（不暴露给 Runner）
    # ====================================
    def _load_model(self, config: dict):
        """私有方法：框架细节"""
        return JiugeForCausalLM.from_pretrained(config["model_path"])


# ============================================================
# 方案B：适配器模式（如果需要兼容旧代码）
# ============================================================

class InfiniLMAdapter(BaseAdapter):
    """
    使用适配器模式兼容旧接口
    """

    def process(self, request: dict) -> dict:
        """
        新接口：process()
        """
        return self._adapt_result(
            self.generate(request["prompt"])
        )

    def generate(self, prompt: str):
        """
        旧接口：保留以兼容
        """
        # 原来的实现
        pass

    def _adapt_result(self, old_result) -> dict:
        """将旧结果格式转换为新格式"""
        return {
            "text": old_result.text,
            "metadata": {...}
        }


# ============================================================
# 三、DataLoader 的提取
# ============================================================

"""
从 Runner 中提取数据准备逻辑
"""

# ============================================================
# 重构前：在 DirectInferRunner 中
# ============================================================

class DirectInferRunner_BEFORE(InferRunnerBase):
    def _load_prompts(self):
        """❌ 数据准备逻辑混在 Runner 中"""
        if self.config.get("prompt_source") == "file":
            with open(self.config["prompt_file"]) as f:
                return json.load(f)
        elif self.config.get("prompt_source") == "generate":
            return self._generate_random_prompts(100)
        else:
            return ["default prompt"]


# ============================================================
# 重构后：独立的 DataLoader
# ============================================================

class DataLoader(ABC):
    """
    数据加载器基类
    """

    @abstractmethod
    def load(self) -> List[Any]:
        """加载数据"""
        pass


class FilePromptLoader(DataLoader):
    """从文件加载 prompts"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[str]:
        with open(self.file_path) as f:
            return json.load(f)


class GeneratePromptLoader(DataLoader):
    """生成 prompts"""

    def __init__(self, count: int, template: str):
        self.count = count
        self.template = template

    def load(self) -> List[str]:
        return [self.template.format(i=i) for i in range(self.count)]


class DataLoaderFactory:
    """工厂模式：根据配置创建 DataLoader"""

    @staticmethod
    def from_config(config: dict) -> DataLoader:
        source = config.get("source", "file")

        if source == "file":
            return FilePromptLoader(config["file_path"])
        elif source == "generate":
            return GeneratePromptLoader(
                config["count"],
                config["template"]
            )
        else:
            raise ValueError(f"Unknown source: {source}")


# ============================================================
# 重构后的 Runner：使用 DataLoader
# ============================================================

class DirectInferenceRunner_AFTER(RunnerBase):
    def setup(self):
        """✅ 使用 DataLoader，逻辑清晰"""
        data_loader = DataLoaderFactory.from_config(self.config["data"])
        self.prompts = data_loader.load()


# ============================================================
# 四、完整的重构流程
# ============================================================

"""
重构步骤总结：

Step 1: 提取 DataLoader
-----------------------
从 Runner 中找出所有数据加载逻辑：
- _load_prompts() → FilePromptLoader
- _load_trace() → TraceLoader
- _generate_prompts() → GeneratePromptLoader

Step 2: 统一 Adapter 接口
-------------------------
将 generate() 改名为 process()：
- InfiniLMAdapter.process()
- vLLMAdapter.process()
- 所有 adapter 使用相同接口

Step 3: 简化 Runner
--------------------
Runner 只保留：
- 流程编排（setup/execute/teardown）
- 指标收集（collect_metrics）
- 调用 adapter 和 dataloader

Step 4: 添加 Dispatcher
------------------------
统一入口：
- 根据 config["test_type"] 选择 Runner
- 根据 config["framework"] 选择 Adapter
"""


# ============================================================
# 五、Dispatcher 的实现示例
# ============================================================

class Dispatcher:
    """
    统一调度器：根据配置创建合适的 Runner
    """

    def dispatch(self, config: dict) -> RunnerBase:
        """
        根据配置分发到合适的 Runner
        """
        test_type = config.get("test_type")

        if test_type == "operator":
            return self._create_operator_runner(config)
        elif test_type == "direct_inference":
            return self._create_direct_inference_runner(config)
        elif test_type == "service_inference":
            return self._create_service_inference_runner(config)
        else:
            raise ValueError(f"Unknown test type: {test_type}")

    def _create_operator_runner(self, config: dict) -> OperatorTestRunner:
        """创建算子测试 Runner"""
        runner = OperatorTestRunner(config)
        runner.adapter = self._create_adapter(config["framework"])
        return runner

    def _create_direct_inference_runner(self, config: dict) -> DirectInferenceRunner:
        """创建直接推理 Runner"""
        runner = DirectInferenceRunner(config)
        runner.adapter = self._create_adapter(config["framework"])
        return runner

    def _create_service_inference_runner(self, config: dict) -> ServiceInferenceRunner:
        """创建服务推理 Runner"""
        runner = ServiceInferenceRunner(config)
        runner.service = self._create_service(config["service_config"])
        return runner

    def _create_adapter(self, framework: str) -> BaseAdapter:
        """根据框架创建 Adapter"""
        if framework == "infinilm":
            return InfiniLMAdapter()
        elif framework == "vllm":
            return vLLMAdapter()
        else:
            raise ValueError(f"Unknown framework: {framework}")


# ============================================================
# 六、最终的使用方式
# ============================================================

def main():
    """
    统一入口：所有测试通过 Dispatcher
    """
    # 1. 加载配置
    config = load_config("config.json")

    # 2. 分发到合适的 Runner
    dispatcher = Dispatcher()
    runner = dispatcher.dispatch(config)

    # 3. 运行测试
    runner.run()

    # 4. 保存结果
    runner.save_results()


# ============================================================
# 七、重构对照表
# ============================================================

"""
| 重构前 | 重构后 | 说明 |
|--------|--------|------|
| DirectInferRunner.generate() | InfiniLMAdapter.process() | 框架交互移到 Adapter |
| DirectInferRunner._load_prompts() | FilePromptLoader.load() | 数据加载移到 DataLoader |
| DirectInferRunner._start_monitoring() | Monitor.start() | 监控逻辑独立 |
| DirectInferRunner.execute() | DirectInferenceRunner.execute() | 保留流程编排 |
| DirectInferRunner.collect_metrics() | RunnerBase.collect_metrics() | 基类提供默认实现 |
| infer_main.py | Dispatcher.dispatch() | 统一入口 |

| 方法类型 | 应该在哪里 | 示例 |
|----------|------------|------|
| 流程编排 | Runner | setup(), execute(), teardown() |
| 框架交互 | Adapter | process(), setup(), teardown() |
| 数据加载 | DataLoader | load() |
| 指标收集 | Runner | collect_metrics() |
| 监控管理 | Monitor | start(), stop() |
| 路由分发 | Dispatcher | dispatch() |
"""
