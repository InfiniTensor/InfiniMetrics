# InfiniMetrics 架构设计文档

## 一、核心架构

### 1.1 架构分层

```
用户层 (main.py)
  ↓
Dispatcher (调度) - 两阶段执行：验证 → 执行
  ↓
Executor (执行) - 生命周期：setup → process → teardown
  ↓
Adapter (适配) - 统一接口，适配不同框架
  ↓
具体框架 (InfiniCore / InfiniLM)
```

### 1.2 核心组件

#### BaseAdapter

```python
class BaseAdapter(abc.ABC):
    @abc.abstractmethod
    def process(self, test_input: Union[TestInput, Dict]) -> Dict:
        """执行测试 (必须实现)"""
        pass

    def setup(self, config: Dict) -> None:
        """初始化资源 (可选)"""
        pass

    def teardown(self) -> None:
        """清理资源 (可选)"""
        pass
```

**设计特点：**
- 无状态 adapter：只实现 `process()`
- 有状态 adapter：实现 `setup + process + teardown()`

#### TestInput

统一测试输入数据结构，提供类型安全的输入封装（详见 [input.py](infinimetrics/input.py)）。

## 二、数据流

### 2.1 完整流程

```
JSON 输入
  ↓
Dispatcher.dispatch()
  ├─> 阶段 1: Validation - 创建所有 adapter
  │   └─> _create_adapter(test_type, framework)
  │
  └─> 阶段 2: Execution - 执行测试
      ├─> Executor(payload, adapter)
      │   ├─> setup()      # TestInput.from_dict() + adapter.setup()
      │   ├─> execute()     # adapter.process(test_input)
      │   └─> teardown()   # adapter.teardown() + _save_result()
      │
      └─> _aggregate_results()
```

### 2.2 数据格式

输入支持 JSON 格式，自动转换为类型安全的 TestInput 对象。

## 三、关键设计

### 3.1 两阶段执行

**问题**: 如何快速发现配置错误？

**解决**: Dispatcher 分两阶段执行

| 阶段 | 目的 | 失败处理 |
|------|------|----------|
| Validation | 验证 adapter 可创建 | 跳过该测试 |
| Execution | 执行测试 | 单个失败不影响其他 |

**优势**:
- ✅ Fail Fast: 执行前发现所有问题
- ✅ 批量反馈: 一次性报告所有无效配置
- ✅ 并行友好: 验证通过后可并行执行

### 3.2 Adapter 无参初始化

**问题**: 何时传递 config？

**解决**: Adapter 无参初始化，config 在 `setup()` 中传递

```python
# Dispatcher 创建 (无参)
adapter = InfiniCoreAdapter()

# Executor 传递 config
adapter.setup(config)
```

**优势**:
- ✅ 关注点分离：Dispatcher 创建，Executor 配置
- ✅ 延迟初始化：执行前才获取配置
- ✅ 灵活复用：同一 adapter 可用于不同配置

### 3.3 职责边界

| 组件 | 职责 | 范围 |
|------|------|------|
| **Dispatcher** | 路由和调度 | 批量测试 |
| **Executor** | 生命周期管理 | 单个测试 |

**Dispatcher**: 验证输入 → 创建 adapter → 聚合结果
**Executor**: 转换输入 → 初始化 adapter → 执行 → 清理 → 保存

## 四、扩展性

### 4.1 添加 Adapter

```python
# 1. 创建 adapter
class VLLMAdapter(BaseAdapter):
    def __init__(self):
        self.model = None

    def setup(self, config: Dict) -> None:
        from vllm import LLM
        self.model = LLM(model=config['model_path'])

    def process(self, test_input: TestInput) -> Dict:
        output = self.model.generate(...)
        return {'result_code': 0, 'data': {...}, 'metrics': []}

    def teardown(self) -> None:
        del self.model

# 2. 在 _ADAPTER_REGISTRY 中注册
_ADAPTER_REGISTRY = {
    ("operator", "infinicore"): lambda: InfiniCoreAdapter(),
    ("inference", "vllm"): lambda: VLLMAdapter(),  # 新增
}
```

### 4.2 添加测试类型

```python
# 1. 定义 testcase 格式
testcase = "training.infinilm.sft"

# 2. 解析
def _parse_testcase(self, testcase: str):
    parts = testcase.split(".")
    return parts[0].lower(), parts[1].lower()

# 3. 创建并注册 adapter
_ADAPTER_REGISTRY = {
    ("training", "infinilm"): lambda: TrainingAdapter(),
}
```

## 五、错误处理

- `result_code = 0`：成功（Linux 惯例）
- `result_code != 0`：错误码
- **验证阶段失败**: 跳过该测试
- **执行阶段失败**: 单个测试失败不影响其他测试
- **异常捕获**: 所有异常都被捕获，保证稳定性

## 六、文件组织

```
infinimetrics/
├── dispatcher.py       # 调度层：两阶段执行
├── executor.py         # 执行层：生命周期管理
├── adapter.py          # 适配层基类
├── input.py            # TestInput 数据类
├── operators/          # 算子适配器
│   └── infinicore_adapter.py
├── inference/          # 推理适配器
└── common/             # 公共工具
```

## 七、最佳实践

1. **实现 teardown**: 有状态 adapter 必须清理资源
2. **错误处理**: Adapter 异常应返回 `result_code!=0`
3. **类型注解**: 使用类型注解提高可读性

## 八、设计模式

| 模式 | 应用 | 优势 |
|------|------|------|
| 注册表模式 | _ADAPTER_REGISTRY | 避免 if-else，易扩展 |
| 工厂模式 | Dispatcher._create_adapter() | 解耦创建逻辑 |
| 模板方法 | Executor.execute() | 统一生命周期 |
| 策略模式 | 不同 Adapter | 灵活切换 |
