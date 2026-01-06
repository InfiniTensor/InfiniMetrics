"""
优雅的 Runner 设计示例
使用模板方法模式 + 钩子方法
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import json

class RunnerBase(ABC):
    """
    所有 Runner 的基类
    使用模板方法模式，定义测试流程骨架
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adapter = None
        self.metrics = {}
        self.metadata = {}

    # ========================================
    # 模板方法：定义流程骨架（不要重写）
    # ========================================
    def run(self):
        """
        主流程：模板方法
        定义了测试的标准流程，子类不要重写这个方法
        """
        try:
            # 1. 初始化阶段
            self.logger.info("=== Test Started ===")
            self.setup()

            # 2. 执行阶段（由子类实现具体逻辑）
            self.execute()

            # 3. 收集阶段
            self.collect_metrics()

            # 4. 清理阶段
            self.teardown()

            # 5. 保存结果
            self.save_results()

            self.logger.info("=== Test Completed ===")

        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            self.teardown()  # 确保资源清理
            raise

    # ========================================
    # 抽象方法：子类必须实现
    # ========================================
    @abstractmethod
    def setup(self):
        """
        初始化：创建 adapter、加载数据、启动监控
        """
        pass

    @abstractmethod
    def execute(self):
        """
        执行：运行测试（核心逻辑，由子类实现）
        """
        pass

    @abstractmethod
    def teardown(self):
        """
        清理：释放资源
        """
        pass

    # ========================================
    # 钩子方法：子类可选重写
    # ========================================
    def collect_metrics(self):
        """
        收集指标：默认实现，子类可以扩展
        """
        self.logger.info("Collecting metrics...")

        # 基础指标收集（所有 Runner 都需要）
        self._collect_common_metrics()

        # 子类特定指标（钩子）
        self._collect_special_metrics()

    def _collect_special_metrics(self):
        """
        钩子方法：子类可以重写来添加特定指标
        """
        pass  # 默认不做任何事

    def save_results(self):
        """
        保存结果：默认实现
        """
        result = {
            "metadata": self.metadata,
            "metrics": self.metrics
        }

        output_path = self.config.get("output_path", "result.json")
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        self.logger.info(f"Results saved to {output_path}")

    # ========================================
    # 通用方法：所有 Runner 共享
    # ========================================
    def _collect_common_metrics(self):
        """
        通用指标收集（所有 Runner 共享）
        """
        # 例如：CPU/GPU 利用率、内存使用等
        pass

    def _start_monitoring(self):
        """启动监控"""
        pass

    def _stop_monitoring(self):
        """停止监控"""
        pass


# ========================================
# 子类示例：直接推理 Runner
# ========================================
class DirectInferenceRunner(RunnerBase):
    """
    直接推理测试 Runner
    特点：同步执行，warmup + measurement
    """

    def setup(self):
        """初始化：加载模型"""
        self.logger.info("Setting up DirectInferenceRunner...")

        # 1. 创建 adapter
        from infinimetrics.inference.adapters import InfiniLMAdapter
        self.adapter = InfiniLMAdapter(self.config["model_config"])

        # 2. 加载模型
        self.adapter.load_model()

        # 3. 加载 prompts（未来：使用 DataLoader）
        self.prompts = self._load_prompts()

        # 4. 启动监控
        self._start_monitoring()

    def execute(self):
        """
        执行：warmup + measurement 两阶段
        这是直接推理特有的逻辑
        """
        self.logger.info("Executing direct inference test...")

        # 1. Warmup 阶段
        warmup_iters = self.config.get("warmup_iterations", 3)
        for i in range(warmup_iters):
            self._execute_single_request(warmup=True)

        # 2. Measurement 阶段
        measure_iters = self.config.get("measure_iterations", 10)
        for i in range(measure_iters):
            self._execute_single_request(warmup=False)

    def _execute_single_request(self, warmup: bool):
        """执行单个请求"""
        prompt = self.prompts[i % len(self.prompts)]
        result = self.adapter.generate(prompt)

        if not warmup:
            # 记录指标
            self._record_inference_metrics(result)

    def _collect_special_metrics(self):
        """
        收集直接推理特有指标：TTFT, TPOT 等
        """
        self.metrics.update({
            "ttft_p50": self._calculate_percentile(self.ttft_list, 50),
            "ttft_p95": self._calculate_percentile(self.ttft_list, 95),
            "tpot": self._calculate_tpot(),
            "throughput_tokens_per_sec": self._calculate_throughput()
        })

    def teardown(self):
        """清理：卸载模型"""
        if self.adapter:
            self.adapter.unload_model()
        self._stop_monitoring()


# ========================================
# 子类示例：服务推理 Runner
# ========================================
class ServiceInferenceRunner(RunnerBase):
    """
    服务推理测试 Runner
    特点：异步执行，trace 驱动
    """

    def setup(self):
        """初始化：启动服务"""
        self.logger.info("Setting up ServiceInferenceRunner...")

        # 1. 启动推理服务
        self.service = self._start_inference_service()

        # 2. 加载 trace（未来：使用 DataLoader）
        self.trace = self._load_trace()

        # 3. 启动监控
        self._start_monitoring()

    async def execute(self):
        """
        执行：异步并发请求
        这是服务推理特有的逻辑
        """
        self.logger.info("Executing service inference test...")

        # 异步执行 trace 中的请求
        tasks = []
        for request in self.trace:
            task = self._execute_async_request(request)
            tasks.append(task)

        # 并发执行
        await asyncio.gather(*tasks)

    async def _execute_async_request(self, request: Dict):
        """执行异步请求"""
        result = await self.service.generate(request["prompt"])
        self._record_service_metrics(result)

    def _collect_special_metrics(self):
        """
        收集服务推理特有指标：QPS, 尾延迟等
        """
        self.metrics.update({
            "qps": self._calculate_qps(),
            "latency_p50": self._calculate_percentile(self.latency_list, 50),
            "latency_p99": self._calculate_percentile(self.latency_list, 99),
            "concurrent_requests": self.config.get("concurrency", 1)
        })

    def teardown(self):
        """清理：停止服务"""
        if self.service:
            self.service.stop()
        self._stop_monitoring()


# ========================================
# 子类示例：算子测试 Runner
# ========================================
class OperatorTestRunner(RunnerBase):
    """
    算子测试 Runner
    特点：单次调用，快速测试
    """

    def setup(self):
        """初始化：创建 adapter"""
        self.logger.info("Setting up OperatorTestRunner...")

        # 1. 创建 adapter（无状态，不需要 load_model）
        from infinimetrics.operators.adapters import InfiniCoreAdapter
        self.adapter = InfiniCoreAdapter()

        # 2. 加载算子配置
        self.op_config = self._load_operator_config()

    def execute(self):
        """
        执行：单次调用算子
        """
        self.logger.info("Executing operator test...")

        iterations = self.config.get("iterations", 100)
        for i in range(iterations):
            result = self.adapter.process(self.op_config)
            self._record_operator_metrics(result)

    def _collect_special_metrics(self):
        """
        收集算子测试特有指标：延迟、正确性
        """
        self.metrics.update({
            "latency_mean": self._calculate_mean(self.latency_list),
            "latency_std": self._calculate_std(self.latency_list),
            "output_correct": self._verify_output()
        })

    def teardown(self):
        """
        清理：算子测试无状态，可能不需要清理
        """
        pass


# ========================================
# 使用示例
# ========================================
def main():
    config = {
        "test_type": "direct_inference",
        "model_config": {...},
        "warmup_iterations": 3,
        "measure_iterations": 10,
        "output_path": "results.json"
    }

    # 根据 test_type 选择 Runner
    runner = DirectInferenceRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
