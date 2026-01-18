# HardwareAdapter 修改总结

## 📋 修改内容

### 1. 新增 `setup()` 方法 ✨

在 `hardware_adapter.py` 中重写了 `setup()` 方法，实现自动编译 CUDA 项目：

```python
def setup(self, config: Dict[str, Any]) -> None:
    """
    初始化资源（编译 CUDA 项目）

    - Mock 模式：跳过编译
    - 正常模式：自动检查并编译 cuda_perf_suite
    """
```

#### 功能特性：

1. **智能检查**
   - 检查 `cuda_perf_suite` 是否已存在
   - 如果存在，跳过编译
   - 如果不存在，自动触发编译

2. **Mock 模式优化**
   - Mock 模式下完全跳过编译检查
   - 适合无 GPU 开发环境

3. **配置控制**
   - `skip_build=True` 可强制跳过编译
   - 提供灵活的控制选项

4. **错误处理**
   - 编译失败时提供清晰的错误信息
   - 包含手动编译的指导
   - 5分钟超时保护

### 2. 编译方法 `_build_cuda_project()`

私有方法，负责实际的编译工作：

```python
def _build_cuda_project(self) -> None:
    """编译 CUDA memory benchmark 项目"""
```

#### 编译流程：

1. 检查 `cuda-memory-benchmark/` 目录
2. 验证 `build.sh` 脚本存在
3. 执行 `./build.sh`
4. 验证可执行文件生成
5. 记录编译日志

#### 错误处理：

- 目录不存在 → `FileNotFoundError`
- 编译失败 → `RuntimeError` (包含详细错误输出)
- 超时 → `RuntimeError` (5分钟超时)

---

## 🔄 工作流程

### 正常模式（有 GPU）

```
创建 Adapter
    ↓
Executor.execute()
    ↓
Executor.setup()
    ↓
adapter.setup(config)
    ↓
检查 cuda_perf_suite 是否存在?
    ├─ 是 → 跳过编译
    └─ 否 → 自动编译
            ↓
        运行 build.sh
            ↓
        验证可执行文件
            ↓
        adapter.process()
            ↓
        执行 CUDA 测试
```

### Mock 模式（无 GPU）

```
创建 Adapter (mock_mode=True)
    ↓
Executor.execute()
    ↓
adapter.setup(config)
    ↓
检测到 mock_mode
    ↓
跳过编译 ✓
    ↓
adapter.process()
    ↓
读取 outputs/*.log
```

---

## 📝 配置选项

### config 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `skip_build` | bool | False | 是否跳过编译 |
| `test_type` | str | "comprehensive" | 测试类型 |
| `output_dir` | str | "./output" | 输出目录 |
| `device_id` | int | 0 | GPU 设备 ID |
| `iterations` | int | 10 | 测试迭代次数 |

### 示例配置

```python
# 正常模式（自动编译）
config = {
    "skip_build": False,  # 允许自动编译
    "test_type": "comprehensive",
    "device_id": 0
}

# 跳过编译（已手动编译）
config = {
    "skip_build": True,  # 跳过编译
    "test_type": "bandwidth"
}

# Mock 模式（无需编译）
config = {
    "test_type": "comprehensive"
    # mock_mode=True 时编译会被自动跳过
}
```

---

## 🧪 测试

### 测试脚本

运行测试验证功能：

```bash
# 测试 setup 方法
python test_hardware_setup.py

# 测试 mock 模式
python test_hardware_adapter_outputs.py
```

### 测试覆盖

✅ Mock 模式 setup（跳过编译）
✅ skip_build 标志
✅ 自动编译（无 CUDA 时预期失败）
✅ Executor 集成

---

## 📂 文件结构

```
infinimetrics/hardware/
├── cuda-memory-benchmark/     # CUDA 测试程序
│   ├── build.sh              # 编译脚本
│   ├── CMakeLists.txt        # CMake 配置
│   └── build/                # 编译输出目录
│       └── cuda_perf_suite   # 可执行文件（编译后）
│
├── outputs/                   # Mock 模式输入
│   ├── bandwidth_test.log
│   ├── stream_test.log
│   ├── cache_test.log
│   └── comprehensive_test.log
│
└── hardware_adapter.py        # 适配器实现
    ├── __init__()
    ├── setup()              # ✨ 新增：自动编译
    ├── process()
    ├── _build_cuda_project() # ✨ 新增：编译方法
    └── teardown()
```

---

## 🔧 手动编译

如果自动编译失败，可以手动编译：

```bash
cd infinimetrics/hardware/cuda-memory-benchmark
./build.sh

# 验证编译结果
ls -la build/cuda_perf_suite
```

### 编译要求

- CUDA Toolkit (nvcc 编译器)
- CMake 3.18+
- C++17 兼容的编译器
- NVIDIA GPU

---

## 💡 使用建议

### 开发环境（无 GPU）

```python
# 使用 Mock 模式
adapter = HardwareTestAdapter(
    mock_mode=True,
    input_outputs_dir="./outputs"
)
```

### 生产环境（有 GPU）

```python
# 自动编译模式
adapter = HardwareTestAdapter(
    mock_mode=False  # 默认
)
# setup() 会自动检查并编译
```

### CI/CD 环境

```python
# 预编译 + 跳过检查
adapter = HardwareTestAdapter(
    mock_mode=False
)

config = {
    "skip_build": True  # 使用预编译的二进制文件
}
```

---

## 🎯 关键改进

1. ✅ **自动化**：无需手动编译，自动检测并构建
2. ✅ **智能跳过**：已存在时跳过编译，节省时间
3. ✅ **Mock 优化**：Mock 模式完全不触发编译
4. ✅ **错误处理**：清晰的错误信息和解决建议
5. ✅ **灵活配置**：支持多种使用场景

---

## 📊 性能影响

| 场景 | 编译时间 | 说明 |
|------|----------|------|
| 已存在可执行文件 | 0ms | 跳过编译 |
| 首次编译 | ~10-30秒 | 取决于系统性能 |
| Mock 模式 | 0ms | 始终跳过 |
| skip_build=True | 0ms | 强制跳过 |

---

## 🐛 故障排查

### 问题：编译失败

**错误信息：**
```
RuntimeError: Failed to build CUDA project: nvcc not found
```

**解决方案：**
1. 安装 CUDA Toolkit
2. 或使用 Mock 模式开发
3. 或使用预编译的二进制文件

### 问题：找不到 build.sh

**错误信息：**
```
FileNotFoundError: Build script not found: build.sh
```

**解决方案：**
1. 检查项目结构是否完整
2. 确保 `cuda-memory-benchmark/` 目录存在

### 问题：超时

**错误信息：**
```
RuntimeError: CUDA project build timed out after 5 minutes
```

**解决方案：**
1. 检查系统资源
2. 手动编译查看详细日志
3. 增加超时时间（修改代码）

---

## 📚 相关文档

- [USAGE.md](USAGE.md) - 详细使用说明
- [test_hardware_setup.py](../test_hardware_setup.py) - Setup 测试脚本
- [test_hardware_adapter_outputs.py](../test_hardware_adapter_outputs.py) - Mock 模式测试

---

**最后更新**: 2026-01-19
**版本**: 1.0
**作者**: Claude Code Assistant
