# 可用于 InfiniMetrics 的现成框架

我们讨论的设计模式（注册表 + 模板方法 + 插件化）在 Python 中有多个成熟框架可以使用。

## 🎯 推荐框架对比

| 框架 | 适用场景 | 优势 | 劣势 | 推荐度 |
|------|---------|------|------|--------|
| **Pluggy** | 插件系统 | pytest 使用的插件框架，成熟稳定 | 学习曲线 | ⭐⭐⭐⭐⭐ |
| **Stevedore** | 动态加载插件 | OpenStack 使用，专为插件设计 | 相对小众 | ⭐⭐⭐⭐⭐ |
| **Hydra** | 配置管理 + 组件实例化 | Facebook 出品，强大配置管理 | 配置较复杂 | ⭐⭐⭐⭐ |
| **Ray** | 分布式任务调度 | 成熟的分布式框架 | 较重 | ⭐⭐⭐⭐ |
| **Kedro** | 数据管道 | 模块化数据管道 | 主要针对数据科学 | ⭐⭐⭐ |
| **Prefect** | 工作流编排 | 现代化工作流框架 | 较重 | ⭐⭐⭐ |

---

## 🚀 方案 1：Pluggy（强烈推荐）

**Pluggy** 是 pytest 使用的插件框架，专为插件化设计。

### 优势

- ✅ **成熟稳定**：pytest 使用的框架，经过大量实战检验
- ✅ **钩子机制**：支持 before/after 钩子，完美匹配我们的需求
- ✅ **装饰器注册**：简洁的 `@hookimpl` 装饰器
- ✅ **轻量级**：核心代码简单易懂
- ✅ **命名空间**：支持多插件管理

### 使用示例

```python
import pluggy

# 1. 定义钩子规范
class TestRunnerSpec:
    """定义 Runner 钩子规范"""

    @pluggy.hookspec(firstresult=True)
    def before_setup(self, config):
        """setup 之前的钩子"""
        pass

    @pluggy.hookspec(firstresult=True)
    def do_setup(self, config):
        """setup 核心逻辑"""
        pass

    @pluggy.hookspec(firstresult=True)
    def after_setup(self, config):
        """setup 之后的钩子"""
        pass

# 2. 创建插件管理器
pm = pluggy.PluginManager("infinimetrics")
pm.add_hookspecs(TestRunnerSpec)

# 3. 实现钩子（基类）
class RunnerBase:
    @pluggy.hookimpl
    def before_setup(self, config):
        """公共逻辑：创建 adapter 和 data_loader"""
        self.adapter = AdapterFactory.create(config["framework"])
        self.data_loader = DataLoaderFactory.from_config(config["data"])
        self.metrics = {}

    @pluggy.hookimpl
    def after_setup(self, config):
        """公共逻辑：启动监控"""
        self.monitor = Monitor()
        self.monitor.start()

# 4. 具体实现（子类）
class DirectInferenceRunner(RunnerBase):
    @pluggy.hookimpl
    def do_setup(self, config):
        """子类特定逻辑"""
        self.adapter.setup(config["model"])
        self.prompts = self.data_loader.load()

# 5. 注册插件
pm.register(RunnerBase())
pm.register(DirectInferenceRunner())

# 6. 执行钩子（自动按顺序调用）
config = {...}
pm.hook.before_setup(config=config)  # 调用所有 before_setup
pm.hook.do_setup(config=config)      # 调用 do_setup
pm.hook.after_setup(config=config)   # 调用 after_setup
```

### 为什么适合我们？

| 需求 | Pluggy 解决方案 |
|------|----------------|
| 注册表模式 | `pm.register(plugin)` |
| 钩子方法 | `@pluggy.hookimpl` 装饰器 |
| before/after 钩子 | 天然支持，自动按顺序调用 |
| 插件化架构 | 完美的插件管理系统 |
| 公共逻辑分离 | 在基类实现钩子，子类可选实现 |

### 安装

```bash
pip install pluggy
```

---

## 🚀 方案 2：Stevedore（推荐）

**Stevedore** 是 OpenStack 开发的动态插件加载库，专为插件化设计。

### 优势

- ✅ **动态加载**：支持运行时动态加载插件
- ✅ **命名空间**：支持插件命名空间
- ✅ **验证机制**：插件加载前验证接口
- ✅ **失败处理**：优雅的失败处理

### 使用示例

```python
import stevedore

# 1. 定义 Adapter 接口
class BaseAdapter(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def process(self, request):
        pass

# 2. 实现 Adapter（自动发现）
class InfiniLMAdapter(BaseAdapter):
    def process(self, request):
        # 实现
        pass

# 3. 使用 Stevedore 加载
# 方式 1：通过命名空间加载
ext_mgr = stevedore.ExtensionManager(
    namespace='infinimetrics.adapters',
    invoke_on_load=True,
)

# 方式 2：通过类路径加载
ext_mgr = stevedore.ExtensionManager(
    namespace='infinimetrics.adapters',
    invoke_args=(config,),
)

# 4. 使用插件
def execute_adapter(ext):
    adapter = ext.obj
    return adapter.process(request)

# 调用所有 adapter
results = list(ext_mgr.map(execute_adapter, requests))

# 或调用特定 adapter
infinilm_adapter = ext_mgr['infinilm'].obj
result = infinilm_adapter.process(request)

# 5. 注册插件（setup.py 或 entry_points）
# setup.py
setup(
    name="infinimetrics-infinilm",
    entry_points={
        'infinimetrics.adapters': [
            'infinilm = infinimetrics.inference.adapters:InfiniLMAdapter',
        ]
    }
)
```

### 安装

```bash
pip install stevedore
```

---

## 🚀 方案 3：Hydra（推荐用于配置管理）

**Hydra** 是 Facebook 开发的配置管理框架，支持强大的组件实例化。

### 优势

- ✅ **配置管理**：YAML 配置文件
- ✅ **组件实例化**：通过配置实例化对象
- ✅ **多环境支持**：开发、测试、生产环境切换
- ✅ ** Sweeps**：超参数调优
- ✅ **集成性好**：可与 PyTorch Lightning 等集成

### 使用示例

```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # 1. 从配置创建 Runner（自动实例化）
    runner = hydra.utils.instantiate(cfg.runner)

    # 2. 从配置创建 Adapter
    adapter = hydra.utils.instantiate(cfg.adapter)

    # 3. 运行测试
    runner.run()

# config.yaml
runner:
  _target_: infinimetrics.inference.runners.DirectInferenceRunner
  framework: infinilm

adapter:
  _target_: infinimetrics.inference.adapters.InfiniLMAdapter
  model_path: /path/to/model
```

### 安装

```bash
pip install hydra-core
```

---

## 🚀 方案 4：自定义轻量级实现

如果不想引入外部依赖，可以自己实现一个轻量级的注册表模式（我们之前讨论的）：

```python
class Registry:
    """轻量级注册表"""

    def __init__(self):
        self._registry = {}

    def register(self, name: str):
        """装饰器：注册组件"""
        def decorator(cls):
            self._registry[name] = cls
            return cls
        return decorator

    def create(self, name: str, **kwargs):
        """创建实例"""
        if name not in self._registry:
            raise ValueError(f"Unknown component: {name}")
        return self._registry[name](**kwargs)

# 使用
adapter_registry = Registry()

@adapter_registry.register("infinilm")
class InfiniLMAdapter(BaseAdapter):
    pass

adapter = adapter_registry.create("infinilm", config={...})
```

**优势**：
- ✅ 无外部依赖
- ✅ 代码简单（~20 行）
- ✅ 完全控制

---

## 🎯 推荐方案组合

根据 InfiniMetrics 的需求，我推荐以下组合：

### **方案 A：轻量级（推荐）**

```python
# 1. 注册表模式（自定义）
class Registry: ...  # 20 行代码

# 2. 模板方法模式（自定义）
class RunnerBase:
    def setup(self):
        self.before_setup()
        self.do_setup()
        self.after_setup()

# 总计：~50 行代码
```

**适用场景**：
- 不想引入外部依赖
- 需要完全控制
- 代码量小，易维护

---

### **方案 B：使用 Pluggy（推荐）**

```python
# 1. 使用 Pluggy 管理钩子
import pluggy
pm = pluggy.PluginManager("infinimetrics")

# 2. 定义钩子规范
pm.add_hookspecs(TestRunnerSpec)

# 3. 实现钩子
@pluggy.hookimpl
def before_setup(config):
    ...

# 4. 执行钩子
pm.hook.before_setup(config=config)
```

**适用场景**：
- 需要强大的插件系统
- 希望社区支持和稳定性
- 未来可能有大量第三方扩展

---

### **方案 C：使用 Hydra + 自定义注册表**

```python
# 1. Hydra 管理配置
import hydra

@hydra.main(config_path="configs")
def main(cfg):
    # 2. 自定义注册表管理组件
    runner = RUNNER_REGISTRY[cfg.runner_type](cfg)
    adapter = ADAPTER_REGISTRY[cfg.adapter_type](cfg)
    runner.run()
```

**适用场景**：
- 需要复杂的配置管理
- 多环境支持（开发/测试/生产）
- 需要超参数调优

---

## 📊 最终推荐

根据你的项目情况，我推荐：

### **短期（当前阶段）**
使用 **自定义轻量级实现**（方案 A）

**理由**：
- ✅ 无外部依赖，易于理解
- ✅ 代码量小（~50 行）
- ✅ 完全满足当前需求
- ✅ 易于调试和维护

### **长期（项目成熟后）**
迁移到 **Pluggy**（方案 B）

**理由**：
- ✅ 成熟稳定（pytest 使用）
- ✅ 强大的插件系统
- ✅ 社区支持
- ✅ 便于第三方扩展

---

## 🚦 决策树

```
需要外部插件支持？
├─ 是 → 使用 Pluggy
└─ 否 →
    需要复杂配置管理？
    ├─ 是 → 使用 Hydra + 自定义注册表
    └─ 否 → 使用自定义轻量级实现
```

---

## 📚 参考资源

- **Pluggy**: https://pluggy.readthedocs.io/
- **Stevedore**: https://docs.openstack.org/stevedore/
- **Hydra**: https://hydra.cc/
- **Ray**: https://www.ray.io/

---

## 💡 我的建议

基于你的项目现状，我建议：

1. **第一阶段**：使用自定义轻量级实现（50 行代码）
2. **第二阶段**：如果需要插件系统，迁移到 Pluggy
3. **第三阶段**：如果需要复杂配置管理，引入 Hydra

这样可以：
- ✅ 快速启动
- ✅ 逐步演进
- ✅ 避免过度设计
- ✅ 保持灵活性

你觉得呢？需要我帮你实现某个具体方案吗？
