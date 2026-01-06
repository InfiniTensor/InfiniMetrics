"""
Dispatcher 实现：统一调度入口

职责：
1. 根据配置选择合适的 Runner
2. 可选：根据配置选择合适的 Adapter
3. 提供统一的测试入口
"""

from typing import Dict, Any
from abc import ABC, abstractmethod


# ============================================================
# 一、Dispatcher 设计
# ============================================================

class Dispatcher:
    """
    统一调度器：根据配置路由到合适的 Runner

    设计模式：
    - 工厂模式：根据 test_type 创建 Runner
    - 策略模式：不同的 Runner 使用不同的执行策略
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def dispatch(self) -> 'RunnerBase':
        """
        根据配置分发到合适的 Runner

        配置示例：
        {
            "test_type": "direct_inference",  # 测试类型
            "framework": "infinilm",          # 框架类型
            ...
        }
        """
        test_type = self.config.get("test_type")

        # 根据 test_type 选择 Runner
        if test_type == "operator":
            return self._create_operator_runner()
        elif test_type == "direct_inference":
            return self._create_direct_inference_runner()
        elif test_type == "service_inference":
            return self._create_service_inference_runner()
        else:
            raise ValueError(f"Unknown test type: {test_type}")

    # ========================================
    # 创建不同的 Runner
    # ========================================

    def _create_operator_runner(self) -> 'OperatorTestRunner':
        """创建算子测试 Runner"""
        from infinimetrics.operators.runners import OperatorTestRunner

        # ✅ 1. 创建 Runner
        runner = OperatorTestRunner(self.config)

        # ✅ 2. 创建并注入 Adapter（如果需要）
        framework = self.config.get("framework", "infinicore")
        runner.adapter = self._create_adapter(framework)

        # ✅ 3. 创建并注入 DataLoader
        runner.data_loader = self._create_data_loader(self.config.get("data"))

        return runner

    def _create_direct_inference_runner(self) -> 'DirectInferenceRunner':
        """创建直接推理 Runner"""
        from infinimetrics.inference.runners import DirectInferenceRunner

        # ✅ 1. 创建 Runner
        runner = DirectInferenceRunner(self.config)

        # ✅ 2. 创建并注入 Adapter
        framework = self.config.get("framework", "infinilm")
        runner.adapter = self._create_adapter(framework)

        # ✅ 3. 创建并注入 DataLoader
        runner.data_loader = self._create_data_loader(self.config.get("data"))

        return runner

    def _create_service_inference_runner(self) -> 'ServiceInferenceRunner':
        """创建服务推理 Runner"""
        from infinimetrics.inference.runners import ServiceInferenceRunner

        # ✅ 1. 创建 Runner
        runner = ServiceInferenceRunner(self.config)

        # ✅ 2. 创建并注入 Service（不是 Adapter）
        runner.service = self._create_service(self.config.get("service"))

        # ✅ 3. 创建并注入 DataLoader
        runner.data_loader = self._create_data_loader(self.config.get("data"))

        return runner

    # ========================================
    # 创建不同的 Adapter
    # ========================================

    def _create_adapter(self, framework: str) -> 'BaseAdapter':
        """
        根据框架创建 Adapter

        框架示例：infinilm, vllm, transformers, infinicore
        """
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

    # ========================================
    # 创建不同的 DataLoader
    # ========================================

    def _create_data_loader(self, data_config: Dict[str, Any]):
        """
        根据配置创建 DataLoader

        数据源示例：file, generate, trace
        """
        from infinimetrics.common.data_loader import DataLoaderFactory

        return DataLoaderFactory.from_config(data_config)

    # ========================================
    # 创建不同的 Service
    # ========================================

    def _create_service(self, service_config: Dict[str, Any]):
        """创建推理服务"""
        from infinimetrics.inference.service import InferenceServiceFactory

        service_type = service_config.get("type", "infinilm")
        return InferenceServiceFactory.create(service_type, service_config)


# ============================================================
# 二、使用示例
# ============================================================

def example_usage():
    """
    统一入口的使用示例
    """
    # 1. 加载配置
    config = {
        "test_type": "direct_inference",
        "framework": "infinilm",
        "model": {
            "model_path": "/path/to/model",
            "tokenizer_path": "/path/to/tokenizer"
        },
        "data": {
            "source": "file",
            "file_path": "/path/to/prompts.json"
        },
        "warmup_iterations": 3,
        "measure_iterations": 10
    }

    # 2. 创建 Dispatcher
    dispatcher = Dispatcher(config)

    # 3. 分发到合适的 Runner
    runner = dispatcher.dispatch()

    # 4. 运行测试（统一的接口）
    runner.run()

    # 5. 获取结果
    results = runner.get_results()


# ============================================================
# 三、配置示例
# ============================================================

# 示例 1：算子测试
operator_test_config = {
    "test_type": "operator",
    "framework": "infinicore",
    "data": {
        "source": "file",
        "file_path": "configs/operator_matmul.json"
    },
    "iterations": 100
}

# 示例 2：直接推理测试
direct_inference_config = {
    "test_type": "direct_inference",
    "framework": "infinilm",
    "model": {
        "model_path": "/models/jiuge-7b",
        "tokenizer_path": "/models/jiuge-7b"
    },
    "data": {
        "source": "file",
        "file_path": "data/prompts.json"
    },
    "warmup_iterations": 3,
    "measure_iterations": 10,
    "output_path": "results/direct_inference.json"
}

# 示例 3：服务推理测试
service_inference_config = {
    "test_type": "service_inference",
    "service": {
        "type": "infinilm",
        "host": "localhost",
        "port": 8080,
        "model_path": "/models/jiuge-7b"
    },
    "data": {
        "source": "trace",
        "trace_file": "traces/production_trace.json"
    },
    "concurrency": 10,
    "output_path": "results/service_inference.json"
}


# ============================================================
# 四、高级用法：带验证的 Dispatcher
# ============================================================

class ValidatingDispatcher(Dispatcher):
    """
    带验证的 Dispatcher：在创建 Runner 之前验证配置
    """

    def dispatch(self) -> 'RunnerBase':
        """分发之前先验证配置"""
        # ✅ 1. 验证配置
        self._validate_config()

        # ✅ 2. 分发到 Runner
        runner = super().dispatch()

        # ✅ 3. 验证 Runner
        self._validate_runner(runner)

        return runner

    def _validate_config(self):
        """验证配置"""
        required_fields = ["test_type"]

        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field: {field}")

        # 根据测试类型验证特定字段
        test_type = self.config["test_type"]

        if test_type == "direct_inference":
            if "framework" not in self.config:
                raise ValueError("direct_inference requires 'framework'")
            if "model" not in self.config:
                raise ValueError("direct_inference requires 'model'")

        elif test_type == "service_inference":
            if "service" not in self.config:
                raise ValueError("service_inference requires 'service'")

    def _validate_runner(self, runner: 'RunnerBase'):
        """验证 Runner 创建是否成功"""
        if runner is None:
            raise RuntimeError("Failed to create runner")

        # 验证 adapter 是否创建成功
        if hasattr(runner, 'adapter') and runner.adapter is None:
            raise RuntimeError("Failed to create adapter")


# ============================================================
# 五、高级用法：带缓存的 Dispatcher
# ============================================================

class CachingDispatcher(Dispatcher):
    """
    带缓存的 Dispatcher：复用已创建的 Runner/Adapter
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._runner_cache = {}
        self._adapter_cache = {}

    def _create_adapter(self, framework: str) -> 'BaseAdapter':
        """复用已创建的 Adapter"""
        if framework not in self._adapter_cache:
            self._adapter_cache[framework] = super()._create_adapter(framework)

        return self._adapter_cache[framework]

    def clear_cache(self):
        """清理缓存"""
        for adapter in self._adapter_cache.values():
            adapter.teardown()

        self._adapter_cache.clear()
        self._runner_cache.clear()


# ============================================================
# 六、CLI 集成
# ============================================================

import argparse
import json

def main():
    """
    命令行入口
    """
    parser = argparse.ArgumentParser(description="InfiniMetrics Test Runner")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to test configuration file"
    )

    parser.add_argument(
        "--test-type",
        type=str,
        choices=["operator", "direct_inference", "service_inference"],
        help="Override test type from config"
    )

    args = parser.parse_args()

    # 1. 加载配置
    with open(args.config) as f:
        config = json.load(f)

    # 2. 覆盖测试类型（如果指定）
    if args.test_type:
        config["test_type"] = args.test_type

    # 3. 创建 Dispatcher
    dispatcher = Dispatcher(config)

    # 4. 分发并运行
    runner = dispatcher.dispatch()
    runner.run()

    print(f"Test completed. Results saved to {config.get('output_path', 'result.json')}")


if __name__ == "__main__":
    main()


# ============================================================
# 七、总结
# ============================================================

"""
Dispatcher 的设计要点：

1. **职责单一**
   - 只负责根据配置创建合适的 Runner
   - 不关心测试流程，不关心框架细节

2. **工厂模式**
   - 根据 test_type 创建不同的 Runner
   - 根据 framework 创建不同的 Adapter
   - 根据 data source 创建不同的 DataLoader

3. **可扩展性**
   - 添加新测试类型：添加新的 _create_xxx_runner() 方法
   - 添加新框架：添加新的分支到 _create_adapter()
   - 添加新数据源：扩展 DataLoaderFactory

4. **可选增强**
   - ValidatingDispatcher：配置验证
   - CachingDispatcher：缓存复用
   - LoggingDispatcher：日志记录

5. **统一入口**
   - 所有测试通过同一个入口
   - 配置文件驱动
   - 易于集成到 CLI/API

使用方式：
```bash
# 命令行
python -m infinimetrics.main --config config.json

# 代码中
dispatcher = Dispatcher(config)
runner = dispatcher.dispatch()
runner.run()
```
"""
