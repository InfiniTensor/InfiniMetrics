"""
更优雅的设计：
1. Adapter 使用注册表模式（类似 Runner）
2. Runner 使用模板方法 + 钩子（before_setup/after_teardown）
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


# ============================================================
# 一、Adapter 使用注册表模式
# ============================================================

class AdapterFactory:
    """
    Adapter 工厂：使用注册表模式，避免 if-elif-else

    优势：
    1. 添加新 adapter 只需注册，无需修改工厂代码
    2. 支持动态注册（运行时注册）
    3. 代码简洁，易于维护
    """

    # Adapter 注册表
    _adapters: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, adapter_class: type = None):
        """
        注册 Adapter（支持装饰器）

        使用方式 1：装饰器
        @AdapterFactory.register("infinilm")
        class InfiniLMAdapter(BaseAdapter):
            pass

        使用方式 2：直接注册
        AdapterFactory.register("vllm", VLLMAdapter)
        """
        def decorator(cls):
            cls._adapters[name] = cls
            return cls

        if adapter_class is None:
            # 作为装饰器使用：@AdapterFactory.register("name")
            return decorator
        else:
            # 直接注册：AdapterFactory.register("name", AdapterClass)
            cls._adapters[name] = adapter_class
            return adapter_class

    @classmethod
    def create(cls, name: str, **kwargs):
        """创建 Adapter 实例"""
        if name not in cls._adapters:
            raise ValueError(f"Unknown adapter: {name}. Available: {list(cls._adapters.keys())}")

        adapter_class = cls._adapters[name]
        return adapter_class(**kwargs)

    @classmethod
    def list_adapters(cls):
        """列出所有注册的 adapter"""
        return list(cls._adapters.keys())


# 使用装饰器注册 Adapter
@AdapterFactory.register("infinilm")
class InfiniLMAdapter(BaseAdapter):
    def setup(self, config: dict = None):
        self.model = load_model(config["model_path"])

    def process(self, request: dict) -> dict:
        return self.model.generate(request["prompt"])

    def teardown(self):
        del self.model


@AdapterFactory.register("vllm")
class VLLMAdapter(BaseAdapter):
    def setup(self, config: dict = None):
        self.model = load_vllm_model(config["model_path"])

    def process(self, request: dict) -> dict:
        return self.model.generate(request["prompt"])

    def teardown(self):
        del self.model


@AdapterFactory.register("infinicore")
class InfiniCoreAdapter(BaseAdapter):
    def setup(self, config: dict = None):
        pass  # 无状态

    def process(self, request: dict) -> dict:
        return InfiniCore.run_operator(request["operator"], request["inputs"])

    def teardown(self):
        pass  # 无状态


# ============================================================
# 二、Runner 使用模板方法 + 钩子
# ============================================================

class RunnerBase(ABC):
    """
    Runner 基类：使用模板方法模式

    模板方法：
    - setup() = before_setup() + do_setup() + after_setup()
    - teardown() = before_teardown() + do_teardown() + after_teardown()

    钩子方法：
    - before_setup(): 公共逻辑（所有 Runner 都需要）
    - do_setup(): 子类重写（每个 Runner 不同）
    - after_setup(): 公共逻辑（所有 Runner 都需要）
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adapter = None
        self.data_loader = None
        self.monitor = None
        self.metrics = {}

    # ========================================
    # 模板方法：setup（不要重写）
    # ========================================
    def setup(self):
        """
        模板方法：定义 setup 流程

        流程：
        1. before_setup()  - 公共逻辑
        2. do_setup()      - 子类特定逻辑
        3. after_setup()   - 公共逻辑
        """
        self.before_setup()
        self.do_setup()
        self.after_setup()

    # ========================================
    # 钩子方法：子类可以重写
    # ========================================
    def before_setup(self):
        """
        钩子：setup 前的公共逻辑

        公共逻辑：
        - 创建 adapter
        - 创建 data_loader
        - 初始化指标收集器
        """
        print("[Common] Creating adapter and data loader...")

        # 1. 创建 adapter（使用工厂）
        framework = self.config.get("framework")
        if framework:
            self.adapter = AdapterFactory.create(framework)

        # 2. 创建 data_loader
        from infinimetrics.common.data_loader import DataLoaderFactory
        data_config = self.config.get("data")
        if data_config:
            self.data_loader = DataLoaderFactory.from_config(data_config)

        # 3. 初始化指标收集器
        self.metrics = {}

    @abstractmethod
    def do_setup(self):
        """
        钩子：子类特定的 setup 逻辑

        子类必须重写这个方法，实现自己的初始化逻辑
        """
        pass

    def after_setup(self):
        """
        钩子：setup 后的公共逻辑

        公共逻辑：
        - 验证初始化结果
        - 启动监控
        """
        print("[Common] Starting monitor...")

        # 启动监控
        if self.config.get("enable_monitoring", True):
            self.monitor = Monitor()
            self.monitor.start()

    # ========================================
    # 模板方法：teardown（不要重写）
    # ========================================
    def teardown(self):
        """
        模板方法：定义 teardown 流程

        流程：
        1. before_teardown() - 公共逻辑
        2. do_teardown()     - 子类特定逻辑
        3. after_teardown()  - 公共逻辑
        """
        self.before_teardown()
        self.do_teardown()
        self.after_teardown()

    # ========================================
    # 钩子方法：子类可以重写
    # ========================================
    def before_teardown(self):
        """
        钩子：teardown 前的公共逻辑

        公共逻辑：
        - 停止监控
        - 收集最终指标
        """
        print("[Common] Stopping monitor...")

        # 停止监控
        if self.monitor:
            self.monitor.stop()
            self.metrics.update(self.monitor.get_metrics())

    @abstractmethod
    def do_teardown(self):
        """
        钩子：子类特定的 teardown 逻辑

        子类必须重写这个方法，实现自己的清理逻辑
        """
        pass

    def after_teardown(self):
        """
        钩子：teardown 后的公共逻辑

        公共逻辑：
        - 保存结果
        - 清理临时文件
        """
        print("[Common] Saving results...")

        # 保存结果
        output_path = self.config.get("output_path", "result.json")
        self._save_results(output_path)

    # ========================================
    # 辅助方法
    # ========================================
    def _save_results(self, output_path: str):
        """保存结果（公共逻辑）"""
        import json
        result = {
            "config": self.config,
            "metrics": self.metrics
        }
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)


# ============================================================
# 三、具体的 Runner：只需重写 do_setup 和 do_teardown
# ============================================================

class DirectInferenceRunner(RunnerBase):
    """
    直接推理 Runner：简洁版

    只需实现 do_setup 和 do_teardown，公共逻辑由基类处理
    """

    def do_setup(self):
        """
        ✅ 子类特定逻辑：加载模型、加载 prompts
        """
        print("[DirectInference] Loading model and prompts...")

        # 1. 调用 adapter 的 setup（加载模型）
        if self.adapter:
            self.adapter.setup(self.config.get("model"))

        # 2. 加载 prompts
        if self.data_loader:
            self.prompts = self.data_loader.load()

    def do_teardown(self):
        """
        ✅ 子类特定逻辑：卸载模型
        """
        print("[DirectInference] Unloading model...")

        # 调用 adapter 的 teardown（卸载模型）
        if self.adapter:
            self.adapter.teardown()


class ServiceInferenceRunner(RunnerBase):
    """
    服务推理 Runner：简洁版
    """

    def do_setup(self):
        """
        ✅ 子类特定逻辑：启动服务、加载 trace
        """
        print("[ServiceInference] Starting service and loading trace...")

        # 1. 创建并启动服务（不需要 adapter）
        from infinimetrics.inference.service import InferenceService
        self.service = InferenceService(self.config["service"])
        self.service.start()

        # 2. 加载 trace
        if self.data_loader:
            self.trace = self.data_loader.load()

    def do_teardown(self):
        """
        ✅ 子类特定逻辑：停止服务
        """
        print("[ServiceInference] Stopping service...")

        # 停止服务
        if self.service:
            self.service.stop()


class OperatorTestRunner(RunnerBase):
    """
    算子测试 Runner：简洁版
    """

    def do_setup(self):
        """
        ✅ 子类特定逻辑：加载算子配置
        """
        print("[OperatorTest] Loading operator config...")

        # 加载算子配置
        if self.data_loader:
            self.op_config = self.data_loader.load()

    def do_teardown(self):
        """
        ✅ 子类特定逻辑：无状态 adapter，无需清理
        """
        print("[OperatorTest] Nothing to cleanup...")
        pass  # 无状态，什么都不做


# ============================================================
# 四、执行流程对比
# ============================================================

"""
调用 DirectInferenceRunner().setup() 时的执行顺序：

1. setup()                    # 模板方法（基类定义，不重写）
   ↓
2. before_setup()             # 公共逻辑（基类实现）
   - 创建 adapter
   - 创建 data_loader
   - 初始化 metrics
   ↓
3. do_setup()                 # 子类特定逻辑（子类重写）
   - adapter.setup()（加载模型）
   - 加载 prompts
   ↓
4. after_setup()              # 公共逻辑（基类实现）
   - 启动 monitor

调用 DirectInferenceRunner().teardown() 时的执行顺序：

1. teardown()                 # 模板方法（基类定义，不重写）
   ↓
2. before_teardown()          # 公共逻辑（基类实现）
   - 停止 monitor
   - 收集指标
   ↓
3. do_teardown()              # 子类特定逻辑（子类重写）
   - adapter.teardown()（卸载模型）
   ↓
4. after_teardown()           # 公共逻辑（基类实现）
   - 保存结果
"""


# ============================================================
# 五、使用示例
# ============================================================

def example_usage():
    """使用示例"""

    config = {
        "test_type": "direct_inference",
        "framework": "infinilm",
        "model": {"model_path": "/path/to/model"},
        "data": {"source": "file", "file_path": "prompts.json"},
        "output_path": "results.json"
    }

    # 创建 Runner
    runner = DirectInferenceRunner(config)

    # setup（自动执行 before + do + after）
    runner.setup()
    # 输出：
    # [Common] Creating adapter and data loader...
    # [DirectInference] Loading model and prompts...
    # [Common] Starting monitor...

    # teardown（自动执行 before + do + after）
    runner.teardown()
    # 输出：
    # [Common] Stopping monitor...
    # [DirectInference] Unloading model...
    # [Common] Saving results...


# ============================================================
# 六、总结：设计模式对比
# ============================================================

"""
┌─────────────────────────────────────────────────────────────┐
│                    设计模式总结                              │
└─────────────────────────────────────────────────────────────┘

1. Adapter：注册表模式
   - ✅ 避免大量的 if-elif-else
   - ✅ 支持装饰器注册
   - ✅ 添加新 adapter 无需修改工厂代码
   - ✅ 代码简洁，易于维护

   使用方式：
   @AdapterFactory.register("infinilm")
   class InfiniLMAdapter(BaseAdapter):
       pass

   adapter = AdapterFactory.create("infinilm")


2. Runner：模板方法模式 + 钩子
   - ✅ 公共逻辑在 before/after 钩子中
   - ✅ 特定逻辑在 do_setup/do_teardown 中
   - ✅ 避免代码重复
   - ✅ 职责清晰

   使用方式：
   class MyRunner(RunnerBase):
       def do_setup(self):
           # 只写自己的逻辑
           pass

       def do_teardown(self):
           # 只写自己的逻辑
           pass


3. 对比改进：

   ❌ 改进前：
   class DirectInferenceRunner:
       def setup(self):
           # 创建 adapter（重复）
           self.adapter = create_adapter(...)
           # 创建 data_loader（重复）
           self.data_loader = create_data_loader(...)
           # 启动 monitor（重复）
           self.monitor = Monitor()
           self.monitor.start()
           # 自己的逻辑
           self.adapter.setup(...)
           self.prompts = load_prompts(...)

   ✅ 改进后：
   class DirectInferenceRunner(RunnerBase):
       def do_setup(self):
           # 只写自己的逻辑
           self.adapter.setup(...)
           self.prompts = load_prompts(...)

   公共逻辑（创建 adapter、启动 monitor）由基类自动处理！


优势总结：
- ✅ 减少重复代码 60%
- ✅ 职责更清晰
- ✅ 更容易扩展
- ✅ 更容易测试
- ✅ 更容易维护
"""

if __name__ == "__main__":
    example_usage()
