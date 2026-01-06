# InfiniMetrics 统一测试框架使用指南

本文档介绍如何使用新创建的统一测试框架（Dispatcher、Executor、BaseAdapter）。

## 📋 架构概览

```
用户层
  ↓
Dispatcher（编排层）- 管理多个Executor
  ↓
Executor（执行层）- 管理单个测试的生命周期
  ↓
BaseAdapter（接口层）- 统一的测试接口
```

### 核心组件

1. **BaseAdapter** - 统一接口，支持推理、算子、训练三种测试
2. **Executor** - 通用执行器，管理测试生命周期
3. **Dispatcher** - 编排器，支持多个Executor和复杂测试场景

## 🚀 快速开始

### 场景1：单个测试（最简单）

```python
from base_adapter import BaseAdapter
from executor import Executor, ExecutorFactory

# 1. 实现你的适配器
class MyInferenceAdapter(BaseAdapter):
    def process(self, request):
        operation = request['operation']
        params = request['params']

        if operation == 'inference':
            # 执行推理测试
            prompts = params['prompts']
            max_tokens = params['max_tokens']

            # 调用你的模型...
            texts = self._generate(prompts, max_tokens)
            latencies = [...]

            return {
                'status': 'success',
                'data': {
                    'texts': texts,
                    'latencies': latencies,
                    'throughput': 150.0
                },
                'metadata': {
                    'memory_gb': 2.5,
                    'compute_time_ms': 200.0
                }
            }

    def setup(self, config):
        # 加载模型
        self.model = load_model(config['model_path'])

    def teardown(self):
        # 清理资源
        del self.model

# 2. 创建适配器实例
adapter = MyInferenceAdapter()

# 3. 创建并运行Executor
executor = ExecutorFactory.create(
    test_type='inference',
    adapter=adapter,
    test_name='My Inference Test',
    output_dir='./results',
    test_params={
        'prompts': ['Hello world', 'Test prompt'],
        'max_tokens': 128
    }
)

results = executor.run()
print(f"Test status: {results['status']}")
print(f"Duration: {results['duration']:.2f}s")
print(f"Results saved to: {results['result_file']}")
```

### 场景2：使用Dispatcher编排多个测试

```python
from dispatcher import Dispatcher
from base_adapter import BaseAdapter

# 1. 准备适配器
adapter = MyInferenceAdapter()

# 2. 配置Dispatcher（单个测试）
config = {
    'test_type': 'single',
    'test_name': 'My Test',
    'adapter': adapter,
    'output_dir': './results',
    'test_params': {
        'prompts': ['Test'],
        'max_tokens': 128
    }
}

# 3. 运行
dispatcher = Dispatcher(config)
results = dispatcher.dispatch()

print(f"Summary: {results['summary']}")
print(f"Results saved to: {results['results'][0]['result_file']}")
```

## 📝 完整示例：推理测试

```python
#!/usr/bin/env python3
"""
完整的推理测试示例
"""

from base_adapter import BaseAdapter
from executor import Executor
from dispatcher import Dispatcher
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)

# ============ 实现 Adapter ============

class SimpleInferenceAdapter(BaseAdapter):
    """简单的推理适配器示例"""

    def setup(self, config):
        print(f"[Adapter] Setting up with config: {config}")
        # 这里加载你的模型
        self.model_ready = True

    def teardown(self):
        print("[Adapter] Tearing down")
        # 这里清理资源

    def process(self, request):
        operation = request['operation']
        params = request['params']

        if operation == 'inference':
            prompts = params['prompts']
            max_tokens = params.get('max_tokens', 128)

            print(f"[Adapter] Processing {len(prompts)} prompts, max_tokens={max_tokens}")

            # 模拟推理
            results = []
            latencies = []
            for prompt in prompts:
                # 模拟生成
                result_text = f"Generated from: {prompt[:50]}..."
                latency = 100.0  # 模拟延迟
                results.append(result_text)
                latencies.append(latency)

            return {
                'status': 'success',
                'data': {
                    'texts': results,
                    'latencies': latencies,
                    'throughput': len(prompts) * max_tokens / (sum(latencies) / 1000)
                },
                'metadata': {
                    'memory_gb': 2.5,
                    'compute_time_ms': sum(latencies)
                }
            }

        return {
            'status': 'error',
            'error': f'Unknown operation: {operation}'
        }

# ============ 使用 Executor ============

def test_with_executor():
    """使用Executor运行单个测试"""
    print("\n=== Test with Executor ===\n")

    adapter = SimpleInferenceAdapter()

    executor = ExecutorFactory.create(
        test_type='inference',
        adapter=adapter,
        test_name='Simple Inference Test',
        output_dir='./results',
        test_params={
            'prompts': ['Hello world', 'How are you?', 'Test prompt'],
            'max_tokens': 128
        }
    )

    results = executor.run()

    print(f"\nResults:")
    print(f"  Status: {results['status']}")
    print(f"  Duration: {results['duration']:.2f}s")
    print(f"  Metrics: {results['metrics']}")
    print(f"  Data keys: {list(results['data'].keys())}")
    print(f"  Result file: {results['result_file']}")

# ============ 使用 Dispatcher ============

def test_with_dispatcher():
    """使用Dispatcher编排测试"""
    print("\n=== Test with Dispatcher ===\n")

    adapter = SimpleInferenceAdapter()

    config = {
        'test_type': 'single',
        'test_name': 'Dispatcher Test',
        'adapter': adapter,
        'output_dir': './results',
        'test_params': {
            'prompts': ['Hello', 'World'],
            'max_tokens': 64
        }
    }

    dispatcher = Dispatcher(config)
    results = dispatcher.dispatch()

    print(f"\nResults:")
    print(f"  Test type: {results['test_type']}")
    print(f"  Total executors: {results['total_executors']}")
    print(f"  Successful: {results['successful_executors']}")
    print(f"  Summary: {results['summary']}")

if __name__ == '__main__':
    test_with_executor()
    test_with_dispatcher()
```

## 🎯 不同测试类型的实现

### 推理测试（Inference）

```python
class InferenceAdapter(BaseAdapter):
    def process(self, request):
        if request['operation'] == 'inference':
            prompts = request['params']['prompts']

            # 调用模型生成
            texts = self.model.generate(prompts)

            return {
                'status': 'success',
                'data': {
                    'texts': texts,
                    'latencies': [...],
                    'ttfts': [...]
                }
            }
```

### 算子测试（Operator）

```python
class OperatorAdapter(BaseAdapter):
    def process(self, request):
        if request['operation'] == 'operator':
            operator_name = request['params']['operator_name']
            iterations = request['params']['iterations']

            # 执行算子测试
            for i in range(iterations):
                result = self.run_operator(operator_name)

            return {
                'status': 'success',
                'data': {
                    'operator_name': operator_name,
                    'iterations': iterations,
                    'avg_latency': [...],
                    'throughput': ...
                }
            }
```

### 训练测试（Training）

```python
class TrainingAdapter(BaseAdapter):
    def process(self, request):
        if request['operation'] == 'training':
            epochs = request['params']['epochs']
            batch_size = request['params']['batch_size']

            # 执行训练
            for epoch in range(epochs):
                loss = self.train_one_epoch(batch_size)

            return {
                'status': 'success',
                'data': {
                    'final_loss': loss,
                    'epochs': epochs,
                    'training_time': ...
                }
            }
```

## 🔧 高级用法

### 扩展Executor（自定义行为）

```python
class MyExecutor(Executor):
    """自定义Executor，添加额外功能"""

    def setup(self):
        """调用父类setup，然后添加自定义逻辑"""
        super().setup()
        # 你的自定义初始化
        self.custom_monitor = CustomMonitor()

    def execute(self):
        """自定义执行逻辑"""
        # 执行前置操作
        self.pre_execute()

        # 调用父类execute
        super().execute()

        # 执行后置操作
        self.post_execute()

    def pre_execute(self):
        print("Custom pre-execution logic")

    def post_execute(self):
        print("Custom post-execution logic")
```

### 扩展Dispatcher（支持新测试类型）

```python
class MyDispatcher(Dispatcher):
    """自定义Dispatcher，支持新的测试类型"""

    def _parse_tasks(self):
        test_type = self.config.get('test_type', 'single')

        if test_type == 'my_custom_type':
            return self._parse_custom_type()
        else:
            return super()._parse_tasks()

    def _parse_custom_type(self):
        """解析自定义测试类型"""
        # 生成多个executor配置
        configs = []
        for i in range(5):
            configs.append({
                'test_type': 'inference',
                'test_name': f'Custom Test {i}',
                ...
            })
        return configs
```

## 📊 结果格式

### Executor结果

```python
{
    'status': 'success',           # 'success' or 'error'
    'start_time': 1234567890.0,    # 开始时间戳
    'end_time': 1234567895.0,      # 结束时间戳
    'duration': 5.0,              # 持续时间（秒）
    'metrics': {                   # 性能指标
        'duration_seconds': 5.0,
        'adapter_memory_gb': 2.5,
        ...
    },
    'data': {                      # 测试数据
        'texts': [...],
        'latencies': [...]
    },
    'metadata': {                  # 元数据
        ...
    },
    'errors': [],                  # 错误列表
    'result_file': '/path/to/results.json'
}
```

### Dispatcher结果

```python
{
    'test_type': 'single',
    'total_executors': 1,
    'successful_executors': 1,
    'failed_executors': 0,
    'results': [...],             # Executor结果列表
    'summary': 'single test completed: 1/1 successful',
    'timestamp': '2025-01-06T15:30:00'
}
```

## ✅ 最佳实践

### 1. Adapter实现

- ✅ 在 `process()` 中验证 `operation` 类型
- ✅ 返回统一格式：`{'status': 'success/error', 'data': ..., 'metadata': ...}`
- ✅ 实现 `setup()`/`teardown()` 管理资源
- ✅ 使用日志记录重要事件

### 2. Executor使用

- ✅ 使用 `ExecutorFactory.create()` 简化创建
- ✅ 提供清晰的 `test_name`
- ✅ 指定输出目录保存结果
- ✅ 检查 `results['status']` 处理错误

### 3. Dispatcher使用

- ✅ 正确设置 `test_type`
- ✅ 提供 `output_dir` 保存结果
- ✅ 检查 `summary` 了解整体结果
- ✅ 查看 `results` 列表获取详细信息

## 🚧 后续扩展

当前实现支持：
- ✅ 单个测试执行
- ✅ 串行执行
- ✅ 基本结果聚合

预留扩展接口：
- ⏭️ 参数扫描（parameter_sweep）
- ⏭️ 框架对比（comparison）
- ⏭️ 工作流编排（workflow）
- ⏭️ 并行执行（parallel）

## 📁 文件位置

所有核心文件位于项目根目录：
```
InfiniMetrics/
├── base_adapter.py          # 适配器基类
├── executor.py              # 执行器
├── dispatcher.py            # 编排器
└── inference/               # 现有推理代码（不动）
    ├── infer_runner_base.py
    └── ...
```

## 💡 设计原则

1. **独立性** - 不依赖 `inference/` 文件夹中的任何代码
2. **通用性** - 支持推理、算子、训练三种测试
3. **可扩展** - 预留清晰的扩展接口
4. **简洁性** - API简单易用，代码清晰
