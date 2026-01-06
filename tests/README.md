# InfiniMetrics 框架单元测试

## 测试概述

本目录包含 InfiniMetrics 统一测试框架的单元测试，覆盖以下核心组件：

- `adapter.py` - 统一适配器接口
- `executor.py` - 通用测试执行器
- `dispatcher.py` - 测试编排器

## 测试文件

- `test_adapter.py` - 测试 BaseAdapter 接口
- `test_executor.py` - 测试 Executor 执行逻辑
- `test_dispatcher.py` - 测试 Dispatcher 编排逻辑

## 运行测试

### 运行所有测试

```bash
# 使用测试运行脚本
python run_tests.py

# 或使用 unittest（详细输出）
python -m unittest discover -s tests -p "test_*.py" -v

# 简洁输出
python -m unittest discover -s tests -p "test_*.py"
```

### 运行单个测试文件

```bash
# 测试 adapter
python -m unittest tests.test_adapter -v

# 测试 executor
python -m unittest tests.test_executor -v

# 测试 dispatcher
python -m unittest tests.test_dispatcher -v
```

### 运行单个测试用例

```bash
python -m unittest tests.test_adapter.TestBaseAdapter.test_process_method -v
```

## 测试覆盖范围

### test_adapter.py (13 个测试)
- ✅ BaseAdapter 抽象类验证
- ✅ process() 方法功能
- ✅ setup() 和 teardown() 生命周期
- ✅ validate() 验证方法
- ✅ get_info() 信息获取
- ✅ 完整生命周期集成测试
- ✅ metrics 返回测试

### test_executor.py (13 个测试)
- ✅ Executor 初始化
- ✅ 测试类型自动检测 (inference/operator)
- ✅ 成功执行流程
- ✅ 失败执行处理
- ✅ Metrics 收集
- ✅ 结果文件保存
- ✅ 异常处理
- ✅ ExecutorFactory 工厂方法
- ✅ 真实 payload 集成测试

### test_dispatcher.py (11 个测试)
- ✅ Dispatcher 初始化
- ✅ 推理测试分发
- ✅ 算子测试分发
- ✅ 测试类型检测
- ✅ 结果聚合
- ✅ 摘要文件保存
- ✅ Adapter 降级处理
- ✅ 真实 payload 集成测试
- ✅ 端到端工作流测试
- ✅ 错误处理

## 测试统计

- **总测试数**: 37 个
- **通过率**: 100%
- **执行时间**: ~0.012s

## Mock 对象

测试使用以下 Mock 对象：

- `MockAdapter` - 基础 Mock 适配器
- `MockInferenceAdapter` - 推理测试 Mock
- `MockOperatorAdapter` - 算子测试 Mock
- `FailingAdapter` - 失败场景 Mock

## 临时文件

测试使用 Python `tempfile` 模块创建临时目录，测试结束后自动清理。所有测试文件都在 `/tmp/` 目录下创建，不会污染项目目录。
