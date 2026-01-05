# InfiniMetrics 当前工作进度

**更新时间**: 2025-01-05
**状态**: 架构设计阶段，部分重构已完成

---

## 一、已完成的工作

### 1.1 BaseAdapter 统一重构 ✅

**目标**: 为 InfiniCoreAdapter 和 InfiniLMAdapter 提供统一的基础类

**完成内容**:

#### 1. 创建了统一的 BaseAdapter
- **文件**: `infinimetrics/adapters/base.py` (182 行)
- **核心接口**:
  - `process(request)` - 唯一必须的抽象方法
  - `setup(config)` - 可选的初始化钩子
  - `teardown()` - 可选的清理钩子
  - `validate()` - 可选的验证钩子
- **支持模式**:
  - 无状态 adapter: 只实现 `process()`
  - 有状态 adapter: 实现 `setup + process + teardown`

#### 2. InfiniCoreAdapter 重构
- **文件**: `infinimetrics/adapters/infinicore.py` (233 行)
- **状态**: ✅ 已使用统一 BaseAdapter
- **模式**: 无状态（只实现 process）
- **改进**: 修复了导入路径问题

#### 3. InfiniLMAdapter 计划重构
- **文件**: `infinimetrics/inference/adapters/infinilm_adapter.py` (496 行)
- **状态**: ⚠️ 计划重构，但**实际代码未修改**
- **问题**: 仍然使用旧的 `InferAdapter` (line 15: `from adapter_base import InferAdapter`)
- **计划内容**:
  - 删除 227 行的 `InferAdapter` 依赖
  - 改用统一的 `BaseAdapter`
  - 实现 `setup + process + teardown` 生命周期
  - 保持向后兼容（保留旧的 `load_model/generate/unload_model` 接口）
- **预期减少**: 496 → 342 行 (-39%)

**验证报告**: `REFACTORING_VERIFICATION_REPORT.md` 显示测试通过，但这是基于计划而非实际代码

### 1.2 架构设计文档 ✅

#### 1. 完整版设计文档
- **文件**: `ARCHITECTURE_DESIGN.md` (1647 行)
- **内容**:
  - 系统概述和设计原则
  - 三层架构详细说明 (Dispatcher → Runner → Adapter)
  - 三种测试模式详解（算子/直接推理/服务推理）
  - 数据流和配置格式
  - 5 阶段实施计划
  - 扩展性设计

#### 2. 精简版设计文档
- **文件**: `ARCHITECTURE.md` (400 行)
- **用途**: 团队 review（适合 30-45 分钟讨论）
- **改进**:
  - 减少代码示例
  - 增加设计决策说明
  - 突出核心思想
  - 结构更清晰

---

## 二、架构设计概要

### 2.1 三层架构

```
用户层 (CLI / API / Config)
  ↓
Dispatcher (调度层) - 路由到合适的测试类型
  ↓
Runner (执行层) - 编排测试流程，管理生命周期
  ↓
Adapter (接口层) - 与具体框架交互
  ↓
DataLoader (数据层) - 准备测试数据
```

### 2.2 统一接口设计

**BaseAdapter** - 单一基类适配所有场景
- 必须实现: `process(request)`
- 可选实现: `setup()`, `teardown()`, `validate()`

**RunnerBase** - 统一的测试流程
- 生命周期: `setup() → execute() → collect_metrics() → teardown()`
- 模板方法模式，子类实现具体步骤

### 2.3 三种测试模式

| 测试类型 | 特点 | 状态 |
|---------|------|------|
| **算子测试** | 单次调用、无状态、快速反馈 | ✅ Adapter 已完成 |
| **直接推理** | 模型加载、warmup+measurement、性能监控 | ⚠️ Adapter 待重构 |
| **服务推理** | 异步并发、trace驱动、服务级指标 | ⚠️ Adapter 待重构 |
| **训练测试** | 长时间运行、loss监控（未来） | ⏳ 待设计 |

---

## 三、待完成的工作

### 3.1 Phase 2: 提取 DataLoader（下一步）

**目标**: 将测试数据准备逻辑从 Runner 中提取出来

**任务清单**:
- [ ] 创建 `DataLoader` 基类
- [ ] 实现 `PromptLoader`
  - 支持从文件加载（txt/json）
  - 支持内存列表
  - 支持自动生成（随机/模板）
- [ ] 实现 `TraceLoader`
  - 支持 JSON 格式的 trace 文件
  - 验证 trace 格式
  - 解析请求序列
- [ ] 从 `DirectInferenceRunner` 中迁移 prompts 生成逻辑
- [ ] 从 `ServiceInferenceRunner` 中迁移 trace 加载逻辑

**预期收益**:
- Runner 代码减少 ~100 行
- 数据准备逻辑可复用
- 易于单元测试

### 3.2 Phase 3: 统一 Runner（核心工作）

**目标**: 为所有 Runner 提供统一的基础类和生命周期管理

**任务清单**:
- [ ] 创建 `RunnerBase` 基类
  - 定义标准生命周期: `setup → execute → collect_metrics → teardown`
  - 实现通用的结果收集和统计
  - 实现数据保存（CSV/JSON）
  - 提供钩子方法: `_add_special_metrics()`, `_custom_execute()`

- [ ] 实现 `OperatorRunner`（轻量级）
  - 无需 warmup/measurement 分离
  - 单次 process 调用
  - 收集延迟和精度指标

- [ ] 重构 `DirectInferenceRunner`
  - 继承 `RunnerBase`
  - 保留现有的 warmup + measurement 逻辑
  - 使用 DataLoader 加载 prompts
  - 重写 `_add_special_metrics()` 添加 perplexity/accuracy

- [ ] 重构 `ServiceInferenceRunner`
  - 继承 `RunnerBase`
  - 实现异步 `execute()` 方法
  - 使用 DataLoader 加载 trace
  - 添加服务管理逻辑（启动/停止）

**预期收益**:
- 统一的测试流程
- 减少重复代码 ~200 行
- 更清晰的结构

### 3.3 Phase 4: 实现 Dispatcher

**目标**: 提供统一的测试入口和配置路由

**任务清单**:
- [ ] 创建 `Dispatcher` 类
  - 解析配置文件
  - 根据 `test_type` 路由到合适的 Runner
  - 统一错误处理
  - 标准化返回格式

- [ ] 实现 CLI 入口
  - 命令行参数解析
  - 配置文件加载
  - 进度显示
  - 结果输出

- [ ] 实现 Python API
  - 简洁的编程接口
  - 支持批量测试
  - 异步执行支持

### 3.4 Phase 5: 清理和优化

**任务清单**:
- [ ] 完成 InfiniLMAdapter 重构（实际修改代码）
- [ ] 删除旧的基类
  - `inference/adapter_base.py` (227 行)
  - `adapters/service_adapter_base.py` (如果存在)
- [ ] 更新所有导入语句
- [ ] 运行完整测试套件
- [ ] 性能优化（如果需要）
- [ ] 更新文档和示例

---

## 四、当前代码状态

### 4.1 文件清单

| 文件 | 行数 | 状态 | 说明 |
|------|------|------|------|
| `adapters/base.py` | 182 | ✅ 完成 | 统一 BaseAdapter |
| `adapters/infinicore.py` | 233 | ✅ 完成 | 使用新基类 |
| `inference/adapters/infinilm_adapter.py` | 496 | ⚠️ 待重构 | 仍用旧基类 |
| `inference/adapter_base.py` | 227 | ❌ 待删除 | 旧的推理基类 |
| `inference/direct_infer_runner.py` | 386 | ⏳ 待重构 | 需使用 RunnerBase |
| `inference/service_infer_runner.py` | 354 | ⏳ 待重构 | 需使用 RunnerBase |
| `inference/infer_runner_base.py` | ~600 | 📝 参考 | 将改造为 RunnerBase |

### 4.2 关键依赖关系

```
InfiniCoreAdapter → BaseAdapter ✅
InfiniLMAdapter → InferAdapter ❌ (应改为 BaseAdapter)
DirectInferenceRunner → InferAdapter ❌ (间接依赖)
ServiceInferenceRunner → InfiniLMAdapter → InferAdapter ❌
```

**问题**: `InfiniLMAdapter` 仍依赖旧的 `InferAdapter`，阻断了统一架构的实现

---

## 五、优先级建议

### 🔴 高优先级（立即执行）

1. **完成 InfiniLMAdapter 重构**
   - 为什么: 这是统一架构的关键依赖
   - 影响: 阻塞 Phase 2-3 的进行
   - 预计时间: 2-3 小时

2. **提取 DataLoader**
   - 为什么: 减少重复代码，为 Runner 重构做准备
   - 影响: 直接影响 DirectInferenceRunner 和 ServiceInferenceRunner
   - 预计时间: 3-4 小时

### 🟡 中优先级（近期执行）

3. **实现 RunnerBase**
   - 为什么: 统一测试流程，提升可维护性
   - 影响: 所有 Runner 的重构基础
   - 预计时间: 4-5 小时

4. **重构 DirectInferenceRunner**
   - 为什么: 最常用的测试模式
   - 影响: 直接推理测试
   - 预计时间: 2-3 小时

### 🟢 低优先级（后续执行）

5. **实现 Dispatcher**
   - 为什么: 提供统一入口，改善用户体验
   - 影响: 所有测试的使用方式
   - 预计时间: 5-6 小时

6. **删除旧代码**
   - 为什么: 清理技术债务
   - 影响: 代码整洁度
   - 预计时间: 1-2 小时

---

## 六、风险和注意事项

### 6.1 技术风险

1. **InfiniLMAdapter 重构可能引入 bug**
   - 缓解: 保持向后兼容，先运行测试
   - 回滚: 保留旧代码作为参考

2. **Runner 重构可能破坏现有功能**
   - 缓解: 渐进式重构，每步有测试保护
   - 验证: 对比重构前后的结果

3. **异步逻辑（ServiceInferenceRunner）复杂度高**
   - 缓解: 单独测试异步执行逻辑
   - 简化: 考虑使用同步+线程池的替代方案

### 6.2 兼容性风险

1. **现有脚本可能依赖旧接口**
   - 缓解: 保留旧接口方法，内部调用新接口
   - 文档: 明确标注哪些方法已废弃

2. **配置格式可能变化**
   - 缓解: 设计灵活的配置解析器
   - 迁移: 提供配置迁移工具

### 6.3 进度风险

1. **重构工作量可能被低估**
   - 缓解: 每个阶段独立验证，及时调整计划
   - 备选: 优先完成核心功能，次要功能延后

---

## 七、下一步行动

### 立即开始

**任务**: 完成 InfiniLMAdapter 的实际重构（非计划，是实际修改代码）

**步骤**:
1. 备份当前文件: `cp infinilm_adapter.py infinilm_adapter.py.bak`
2. 修改导入: `from adapters.base import BaseAdapter`
3. 移除 `InferAdapter` 继承
4. 实现新的生命周期方法:
   - `setup()`: 包含原 `load_model()` 逻辑
   - `process()`: 调用 `generate()` 并返回标准格式
   - `teardown()`: 包含原 `unload_model()` 逻辑
5. 保留旧接口作为兼容层:
   ```python
   def load_model(self):
       self.setup()

   def unload_model(self):
       self.teardown()
   ```
6. 运行测试验证功能
7. 删除备份文件

**验收标准**:
- [ ] 代码行数 < 350 行
- [ ] 继承自 `BaseAdapter`
- [ ] 所有单元测试通过
- [ ] 向后兼容性保持

---

## 八、总结

**当前状态**: 架构设计完成，BaseAdapter 统一完成，但 InfiniLMAdapter **实际代码未修改**

**核心进展**:
- ✅ 统一 BaseAdapter 设计和实现
- ✅ 完整的架构设计文档（完整版 + 精简版）
- ⚠️ InfiniLMAdapter 重构计划完成，但**未实际执行**

**阻塞问题**:
- InfiniLMAdapter 仍依赖旧的 InferAdapter，影响后续重构

**关键路径**:
```
InfiniLMAdapter 重构 → DataLoader 提取 → RunnerBase 实现 → Dispatcher 实现
     (下一步)              (Phase 2)           (Phase 3)          (Phase 4)
```

**预计完成时间**: 按优先级逐步执行，约 2-3 周完成所有阶段

---

**维护者**: 请在每次重大进展后更新此文档
**更新频率**: 每完成一个 Phase 或关键任务时更新
