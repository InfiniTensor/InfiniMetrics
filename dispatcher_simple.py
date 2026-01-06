"""
Dispatcher 简洁版：避免重复代码

核心思想：
1. Runner 自己负责创建 adapter 和 data_loader
2. Dispatcher 只负责根据 test_type 选择 Runner
3. 减少重复代码
"""

from typing import Dict, Any


# ============================================================
# 方案 1：最简洁 - Dispatcher 只选 Runner，不负责创建组件
# ============================================================

class Dispatcher:
    """
    简洁版 Dispatcher：只负责选择 Runner

    职责：
    1. 根据 test_type 选择合适的 Runner 类
    2. 创建 Runner 实例
    3. 其他逻辑由 Runner 自己负责
    """

    # 测试类型到 Runner 类的映射
    RUNNER_CLASSES = {
        "operator": "infinimetrics.operators.runners:OperatorTestRunner",
        "direct_inference": "infinimetrics.inference.runners:DirectInferenceRunner",
        "service_inference": "infinimetrics.inference.runners:ServiceInferenceRunner",
    }

    def dispatch(self, config: Dict[str, Any]):
        """
        根据配置分发到合适的 Runner
        """
        test_type = config.get("test_type")

        # 动态导入并创建 Runner
        runner_class = self._get_runner_class(test_type)
        return runner_class(config)

    def _get_runner_class(self, test_type: str):
        """动态导入 Runner 类"""
        if test_type not in self.RUNNER_CLASSES:
            raise ValueError(f"Unknown test type: {test_type}")

        module_path, class_name = self.RUNNER_CLASSES[test_type].split(":")

        # 动态导入
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)


# ============================================================
# 方案 2：Runner 自己负责创建 adapter 和 data_loader
# ============================================================

class RunnerBase:
    """
    Runner 基类：负责自己的初始化
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # ✅ 在 __init__ 中自动创建 adapter 和 data_loader
        self.adapter = self._create_adapter()
        self.data_loader = self._create_data_loader()

    def _create_adapter(self):
        """子类可以重写这个方法来指定使用哪个 adapter"""
        framework = self.config.get("framework")
        if not framework:
            return None

        if framework == "infinilm":
            from infinimetrics.inference.adapters import InfiniLMAdapter
            return InfiniLMAdapter()
        elif framework == "vllm":
            from infinimetrics.inference.adapters import VLLMAdapter
            return VLLMAdapter()
        elif framework == "infinicore":
            from infinimetrics.operators.adapters import InfiniCoreAdapter
            return InfiniCoreAdapter()
        else:
            raise ValueError(f"Unknown framework: {framework}")

    def _create_data_loader(self):
        """创建 DataLoader（通用逻辑）"""
        from infinimetrics.common.data_loader import DataLoaderFactory

        data_config = self.config.get("data")
        if not data_config:
            return None

        return DataLoaderFactory.from_config(data_config)


# ============================================================
# 具体的 Runner：只需要实现核心逻辑
# ============================================================

class DirectInferenceRunner(RunnerBase):
    """
    直接推理 Runner：简洁版
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)  # ← 自动创建 adapter 和 data_loader

        # ✅ adapter 已经创建好了，直接用
        self.adapter.setup(self.config.get("model"))

        # ✅ data_loader 已经创建好了，直接加载数据
        self.prompts = self.data_loader.load()

    def setup(self):
        """初始化监控等"""
        self.monitor = Monitor()
        self.monitor.start()

    def execute(self):
        """执行推理"""
        for prompt in self.prompts:
            result = self.adapter.process({"prompt": prompt})
            # 处理结果...

    def teardown(self):
        """清理"""
        self.monitor.stop()
        self.adapter.teardown()


class OperatorTestRunner(RunnerBase):
    """
    算子测试 Runner：简洁版
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)  # ← 自动创建 adapter 和 data_loader

        # ✅ 加载算子配置
        self.op_config = self.data_loader.load()

    def setup(self):
        """算子测试可能不需要额外的 setup"""
        pass

    def execute(self):
        """执行算子测试"""
        result = self.adapter.process(self.op_config)
        # 处理结果...

    def teardown(self):
        """无状态 adapter，不需要 teardown"""
        pass


# ============================================================
# 方案 3：更简洁 - 使用 register 装饰器
# ============================================================

class SimpleDispatcher:
    """
    超简洁版 Dispatcher：使用装饰器注册 Runner
    """

    # Runner 注册表
    _runners = {}

    @classmethod
    def register(cls, test_type: str):
        """
        装饰器：用于注册 Runner

        使用示例：
        @SimpleDispatcher.register("direct_inference")
        class DirectInferenceRunner(RunnerBase):
            pass
        """
        def decorator(runner_class):
            cls._runners[test_type] = runner_class
            return runner_class
        return decorator

    @classmethod
    def dispatch(cls, config: Dict[str, Any]):
        """分发到合适的 Runner"""
        test_type = config.get("test_type")

        if test_type not in cls._runners:
            raise ValueError(f"Unknown test type: {test_type}")

        return cls._runners[test_type](config)


# 使用装饰器注册 Runner
@SimpleDispatcher.register("operator")
class OperatorTestRunner(RunnerBase):
    pass


@SimpleDispatcher.register("direct_inference")
class DirectInferenceRunner(RunnerBase):
    pass


@SimpleDispatcher.register("service_inference")
class ServiceInferenceRunner(RunnerBase):
    pass


# ============================================================
# 方案 4：最简洁 - 直接用字典映射
# ============================================================

# 定义 Runner 映射（在 runners 模块中）
RUNNER_REGISTRY = {
    "operator": OperatorTestRunner,
    "direct_inference": DirectInferenceRunner,
    "service_inference": ServiceInferenceRunner,
}


class UltraSimpleDispatcher:
    """
    极简版 Dispatcher：最简单的实现
    """

    def dispatch(self, config: Dict[str, Any]):
        test_type = config.get("test_type")

        if test_type not in RUNNER_REGISTRY:
            raise ValueError(f"Unknown test type: {test_type}")

        return RUNNER_REGISTRY[test_type](config)


# ============================================================
# 使用示例
# ============================================================

def example_usage():
    """所有方案的使用方式都一样"""
    config = {
        "test_type": "direct_inference",
        "framework": "infinilm",
        "model": {...},
        "data": {...}
    }

    # 方案 1, 2, 4
    dispatcher = Dispatcher()
    runner = dispatcher.dispatch(config)
    runner.run()

    # 方案 3
    runner = SimpleDispatcher.dispatch(config)
    runner.run()


# ============================================================
# 对比：啰嗦版 vs 简洁版
# ============================================================

"""
❌ 原来的啰嗦版（我之前写的）：

class Dispatcher:
    def _create_operator_runner(self):
        runner = OperatorTestRunner(self.config)
        runner.adapter = self._create_adapter(framework)      # 重复
        runner.data_loader = self._create_data_loader(data)    # 重复
        return runner

    def _create_direct_inference_runner(self):
        runner = DirectInferenceRunner(self.config)
        runner.adapter = self._create_adapter(framework)      # 重复
        runner.data_loader = self._create_data_loader(data)    # 重复
        return runner

    def _create_service_inference_runner(self):
        runner = ServiceInferenceRunner(self.config)
        runner.service = self._create_service(service)         # 稍有不同
        runner.data_loader = self._create_data_loader(data)    # 重复
        return runner


✅ 简洁版（推荐）：

class Dispatcher:
    RUNNER_CLASSES = {
        "operator": OperatorTestRunner,
        "direct_inference": DirectInferenceRunner,
        "service_inference": ServiceInferenceRunner,
    }

    def dispatch(self, config):
        test_type = config["test_type"]
        return self.RUNNER_CLASSES[test_type](config)  # 一行搞定！

# Runner 自己负责创建组件
class DirectInferenceRunner(RunnerBase):
    def __init__(self, config):
        super().__init__(config)  # 自动创建 adapter 和 data_loader
        # 直接使用，无需重复代码
"""


# ============================================================
# 如果真的需要区分 service_inference（不需要 adapter）
# ============================================================

class RunnerBase:
    """基类：有 adapter"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adapter = self._create_adapter()
        self.data_loader = self._create_data_loader()

    def _create_adapter(self):
        """创建 adapter"""
        framework = self.config.get("framework")
        if framework == "infinilm":
            from infinimetrics.inference.adapters import InfiniLMAdapter
            return InfiniLMAdapter()
        # ...


class ServiceRunnerBase(RunnerBase):
    """Service Runner 基类：不需要 adapter，需要 service"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.service = self._create_service()  # 创建 service 而不是 adapter
        self.data_loader = self._create_data_loader()

    def _create_service(self):
        """创建 service"""
        from infinimetrics.inference.service import InferenceService
        return InferenceService(self.config["service"])


# 或者更简单：在 Runner 内部判断
class DirectInferenceRunner(RunnerBase):
    def _create_adapter(self):
        # 有 adapter
        return InfiniLMAdapter()


class ServiceInferenceRunner(RunnerBase):
    def _create_adapter(self):
        # 不需要 adapter，返回 None
        return None

    def _create_service(self):
        # 创建 service
        return InferenceService(self.config["service"])


# ============================================================
# 总结：推荐方案
# ============================================================

"""
推荐使用 **方案 1 + 方案 2** 的组合：

1. Dispatcher 只负责选择 Runner（使用字典映射）
2. Runner.__init__ 自动创建 adapter 和 data_loader
3. 减少重复代码，职责清晰

代码量对比：
- 原版：~60 行（大量重复）
- 简洁版：~20 行（无重复）

核心改进：
- ❌ 删除：_create_*_runner() 重复代码
- ✅ 添加：Runner.__init__ 自动创建组件
- ✅ 结果：代码更少，逻辑更清晰
"""

if __name__ == "__main__":
    # 使用示例
    config = {
        "test_type": "direct_inference",
        "framework": "infinilm",
        "model": {"model_path": "/path/to/model"},
        "data": {"source": "file", "file_path": "prompts.json"}
    }

    dispatcher = Dispatcher()
    runner = dispatcher.dispatch(config)
    runner.run()
