# Dispatcher 两阶段执行架构

## 概述

Dispatcher 采用**两阶段执行架构**，将验证和执行分离，为异步并行执行提供基础。

## 架构设计

### 阶段 1：验证阶段（Validation Phase）

**目标**：创建所有 adapters，验证配置有效性

```python
# 验证阶段：创建所有 adapters
valid_executions = []
skipped_results = []

for payload in payloads:
    testcase = payload['testcase']
    test_type, framework = self._parse_testcase(testcase)
    adapter_config = payload.get('config', {})

    try:
        adapter = self._create_adapter(test_type, framework, adapter_config)
        valid_executions.append((payload, adapter))
        logger.debug(f"Validated {testcase} - adapter ready")
    except ValueError as e:
        logger.error(f"Skipping {testcase}: {e}")
        skipped_results.append({...})  # 记录跳过的测试
```

**输出**：
- `valid_executions`: 列表，元素为 `(payload, adapter)` 元组
- `skipped_results`: 列表，包含所有被跳过的测试信息

### 阶段 2：执行阶段（Execution Phase）

**目标**：执行所有通过验证的测试

```python
# 执行阶段：运行所有有效测试
all_results = []
for idx, (payload, adapter) in enumerate(valid_executions, 1):
    testcase = payload['testcase']
    logger.info(f"[{idx}/{len(valid_executions)}] Executing {testcase}")

    executor = Executor(payload, adapter)
    result = executor.run()
    all_results.append(result)

# 添加跳过的结果
all_results.extend(skipped_results)
```

**输出**：
- `all_results`: 包含执行结果和跳过信息的完整结果列表

## 优势

### 1. 清晰的错误处理

- ✅ 配置错误在执行前被发现
- ✅ 无效测试被跳过，不影响其他测试
- ✅ 错误信息明确：知道哪些测试被跳过以及原因

### 2. 高效的验证

- ✅ 快速失败：在执行前发现所有配置问题
- ✅ 资源节约：不为无效测试创建 executor
- ✅ 用户友好：一次性报告所有配置问题

### 3. 易于扩展为异步

由于 adapters 在验证阶段已创建，执行阶段可以轻松并行化：

```python
# 同步执行（当前实现）
for payload, adapter in valid_executions:
    executor = Executor(payload, adapter)
    result = executor.run()
    all_results.append(result)
```

```python
# 异步并行执行（未来扩展）
import asyncio
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(self._run_single_test, payload, adapter)
        for payload, adapter in valid_executions
    ]
    all_results = [f.result() for f in futures]
```

## 跳过的测试

### 触发条件

测试会在以下情况下被跳过：

1. **配置无效**：`validate_config()` 返回 `False`
2. **Testcase 格式错误**：无法解析 test_type 和 framework
3. **Adapter 创建失败**：`_create_adapter()` 抛出 `ValueError`

### 跳过记录格式

```python
{
    'run_id': 'test_001',              # 测试 ID
    'testcase': 'infer.Invalid.Test',  # Testcase 名称
    'success': 0,                      # 失败标志
    'duration': 0,                     # 持续时间为 0
    'result_file': None,               # 无结果文件
    'error': 'vllm adapter not implemented',  # 跳过原因
    'skipped': True                    # 跳过标志
}
```

### Summary 报告

Summary JSON 文件中包含所有测试的执行情况：

```json
{
  "total_tests": 5,
  "successful_tests": 3,
  "failed_tests": 2,
  "results": [
    {
      "run_id": "test_001",
      "testcase": "infer.InfiniLM.Direct",
      "success": 1,
      "skipped": false
    },
    {
      "run_id": "test_002",
      "testcase": "infer.vLLM.Batch",
      "success": 0,
      "error": "vllm adapter not implemented",
      "skipped": true
    }
  ]
}
```

## 异步扩展指南

### 方案 1：ThreadPoolExecutor（推荐用于 I/O 密集型）

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def dispatch_async(self, inputs: Any, max_workers: int = 4) -> Dict[str, Any]:
    """使用线程池并行执行测试"""

    # ... 验证阶段（不变）...

    # 执行阶段：并行运行
    all_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_test = {
            executor.submit(self._run_single_test, payload, adapter): payload['testcase']
            for payload, adapter in valid_executions
        }

        # 收集结果（按完成顺序）
        for future in as_completed(future_to_test):
            testcase = future_to_test[future]
            try:
                result = future.result()
                all_results.append(result)
                logger.info(f"Completed {testcase}")
            except Exception as e:
                logger.error(f"Test {testcase} failed: {e}")

    all_results.extend(skipped_results)
    return self._aggregate_results(all_results)

def _run_single_test(self, payload, adapter) -> Dict[str, Any]:
    """运行单个测试"""
    executor = Executor(payload, adapter)
    return executor.run()
```

### 方案 2：ProcessPoolExecutor（推荐用于 CPU 密集型）

```python
from concurrent.futures import ProcessPoolExecutor

def dispatch_async_mp(self, inputs: Any, max_workers: int = None) -> Dict[str, Any]:
    """使用进程池并行执行测试"""

    # ... 验证阶段（不变）...

    # 执行阶段：多进程并行
    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(self._run_single_test, payload, adapter)
            for payload, adapter in valid_executions
        ]
        all_results = [f.result() for f in futures]

    all_results.extend(skipped_results)
    return self._aggregate_results(all_results)
```

### 方案 3：AsyncIO（推荐用于高并发 I/O）

```python
import asyncio

async def dispatch_async_io(self, inputs: Any) -> Dict[str, Any]:
    """使用 asyncio 并发执行测试"""

    # ... 验证阶段（不变）...

    # 执行阶段：异步并发
    tasks = [
        self._run_single_test_async(payload, adapter)
        for payload, adapter in valid_executions
    ]
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    # 处理异常
    processed_results = []
    for result in all_results:
        if isinstance(result, Exception):
            logger.error(f"Test failed: {result}")
            # 创建错误结果
            processed_results.append({...})
        else:
            processed_results.append(result)

    processed_results.extend(skipped_results)
    return self._aggregate_results(processed_results)

async def _run_single_test_async(self, payload, adapter) -> Dict[str, Any]:
    """异步运行单个测试"""
    # 假设 Executor 支持异步
    executor = Executor(payload, adapter)
    return await executor.run_async()
```

## 使用示例

### 同步执行（默认）

```python
from dispatcher import Dispatcher

dispatcher = Dispatcher({'output_dir': './output'})

# 同步执行
results = dispatcher.dispatch(configs)
```

### 并行执行（扩展）

```python
from dispatcher import Dispatcher

dispatcher = Dispatcher({'output_dir': './output'})

# 并行执行（4 个线程）
results = dispatcher.dispatch_async(configs, max_workers=4)
```

## 性能对比

假设有 10 个测试，每个测试需要 5 秒：

| 执行方式 | 总耗时 | 加速比 |
|---------|--------|--------|
| 同步顺序 | 50s | 1x |
| 4 线程并行 | ~15s | 3.3x |
| 8 线程并行 | ~10s | 5x |
| 4 进程并行 | ~15s | 3.3x |

*实际性能取决于测试类型（I/O 密集 vs CPU 密集）和资源限制*

## 最佳实践

### 1. 选择合适的并行度

```python
import os

# I/O 密集型：可以使用更多线程
max_workers = min(32, (os.cpu_count() or 1) * 4)

# CPU 密集型：使用 CPU 核心数
max_workers = os.cpu_count() or 1
```

### 2. 控制资源使用

```python
# 限制并发数，避免资源耗尽
MAX_CONCURRENT_TESTS = 4

with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TESTS) as executor:
    ...
```

### 3. 错误处理

```python
# 记录但继续执行
for future in as_completed(future_to_test):
    try:
        result = future.result()
        all_results.append(result)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        # 创建错误结果，不影响其他测试
        all_results.append({
            'success': 0,
            'error': str(e),
            'testcase': future_to_test[future]
        })
```

### 4. 进度报告

```python
total = len(valid_executions)
completed = 0

for future in as_completed(future_to_test):
    completed += 1
    logger.info(f"Progress: {completed}/{total}")
    ...
```

## 注意事项

1. **线程安全**：确保 adapters 和 executors 是线程安全的
2. **资源竞争**：多个测试同时访问同一资源（如 GPU）可能导致问题
3. **内存使用**：并行执行会增加内存占用
4. **日志顺序**：并行执行时日志可能交错

## 相关文档

- [DISPATCHER_USAGE.md](DISPATCHER_USAGE.md) - Dispatcher 基础使用指南
- [TESTCASE_FORMAT.md](TESTCASE_FORMAT.md) - Testcase 命名规范
- [BATCH_DISPATCH_USAGE.md](BATCH_DISPATCH_USAGE.md) - 批量调度指南
