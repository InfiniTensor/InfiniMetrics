# Test Case Generator

自动生成 InfiniMetrics 测试用例配置文件的工具，支持灵活的形状分布参数。

## 📋 目录

- [功能特性](#功能特性)
- [安装依赖](#安装依赖)
- [快速开始](#快速开始)
- [Shape 分布类型](#shape-分布类型)
- [使用示例](#使用示例)
- [配置文件格式](#配置文件格式)
- [命令行参数](#命令行参数)

---

## ✨ 功能特性

- **7种分布类型**: uniform, log_uniform, geometric, powers_of_2, choice, randint, fixed
- **灵活组合**: 自动生成所有 shape 组合
- **可复现性**: 支持随机种子
- **双重接口**: CLI 和 YAML 配置文件
- **多算子支持**: matmul, add, conv2d

---

## 📦 安装依赖

```bash
pip install numpy pyyaml
```

---

## 🚀 快速开始

### 方式1: 使用 YAML 配置文件（推荐）

```bash
# 1. 查看示例配置
ls test_templates/

# 2. 生成测试用例
python scripts/generate_test_cases.py \
    --config test_templates/matmul_uniform.yaml \
    --output ./generated_tests/

# 3. 运行测试
python main.py ./generated_tests/
```

### 方式2: 使用命令行参数

```bash
# 生成 MatMul 测试
python scripts/generate_test_cases.py \
    --operator matmul \
    --shape-distribution "uniform(512, 4096, 4) uniform(512, 4096, 4)" \
    --dtype float16 \
    --device nvidia \
    --output ./generated_tests/
```

---

## 📊 Shape 分布类型

### 1. uniform - 均匀分布

在指定范围内均匀采样。

```python
uniform(min, max, samples)
```

**示例:**
```bash
"uniform(512, 4096, 4)"  → [512, 1536, 2560, 3584]
```

**适用场景:** 线性探索性能空间

---

### 2. log_uniform - 对数均匀分布

在对数尺度上均匀采样（适合规模探索）。

```python
log_uniform(min, max, samples)
```

**示例:**
```bash
"log_uniform(512, 8192, 3)"  → [512, 2048, 8192]
```

**适用场景:** 测试不同数量级的规模

---

### 3. geometric - 几何级数

按固定比例增长。

```python
geometric(start, ratio, count)
```

**示例:**
```bash
"geometric(512, 2, 4)"  → [512, 1024, 2048, 4096]
```

**适用场景:** 模拟网络层深度增长

---

### 4. powers_of_2 - 2的幂次

生成 2 的幂次（硬件友好）。

```python
powers_of_2(min_exp, max_exp)
```

**示例:**
```bash
"powers_of_2(9, 12)"  → [512, 1024, 2048, 4096]
```

**适用场景:** CPU/GPU 缓存优化测试

---

### 5. choice - 从列表选择

随机选择指定数量的值。

```python
choice([values], count, replace=True)
```

**示例:**
```bash
"choice([512, 1024, 2048, 4096], 2)"      → 随机选2个
"choice([512, 1024, 2048], 3, replace=False)" → 全部选择
```

**适用场景:** 自定义特定尺寸

---

### 6. randint - 随机整数

生成指定范围内的随机整数。

```python
randint(min, max, count)
```

**示例:**
```bash
"randint(512, 4096, 3)"  → [723, 1890, 3541]  (随机)
```

**适用场景:** 随机探索

---

### 7. fixed - 固定值

直接指定具体值。

```python
[1024, 768]
```

**示例:**
```bash
"[1024, 768]"  → [1024, 768]
```

**适用场景:** 混合固定和动态尺寸

---

## 💡 使用示例

### 示例1: MatMul 均匀分布

```bash
python scripts/generate_test_cases.py \
    --operator matmul \
    --shape-distribution "uniform(512, 4096, 4) uniform(512, 4096, 4)" \
    --dtype float16 \
    --device nvidia \
    --output ./generated_tests/
```

**结果:** 生成 4×4=16 种组合的 MatMul 测试

---

### 示例2: MatMul 几何级数（2的幂次）

```bash
python scripts/generate_test_cases.py \
    --operator matmul \
    --shape-distribution "powers_of_2(9, 12) powers_of_2(9, 11)" \
    --dtype float16 \
    --device nvidia \
    --seed 42 \
    --output ./generated_tests/
```

**结果:** 生成 4×3=12 种组合 (512×512, 512×1024, ..., 4096×2048)

---

### 示例3: Conv2D 混合分布

```bash
python scripts/generate_test_cases.py \
    --operator conv2d \
    --shape-distribution "[1] geometric(64, 2, 4) choice([56,28,14,7], 2) choice([56,28,14,7], 2) | geometric(64, 2, 4) geometric(64, 2, 3) [3] [3]" \
    --stride 1 \
    --padding 1 \
    --dtype float16 \
    --device nvidia \
    --max-combinations 20 \
    --output ./generated_tests/
```

**说明:**
- `[1]`: 固定 batch size
- `geometric(64, 2, 4)`: 通道数 64, 128, 256, 512
- `choice([56,28,14,7], 2)`: 从 ResNet 尺寸中随机选2个
- `|`: 分隔两个张量（input 和 weight）

---

### 示例4: 使用 YAML 配置

```bash
# 查看配置文件
cat test_templates/matmul_uniform.yaml

# 生成测试
python scripts/generate_test_cases.py \
    --config test_templates/matmul_uniform.yaml \
    --output ./generated_tests/
```

---

### 示例5: Element-wise Add

```bash
python scripts/generate_test_cases.py \
    --operator add \
    --shape-distribution "uniform(1024, 8192, 5) uniform(1024, 8192, 5)" \
    --dtype float32 \
    --device cpu \
    --output ./generated_tests/
```

---

## 📄 YAML 配置文件格式

```yaml
# 算子名称
operator: matmul

# Shape 分布配置（每个张量的维度分布）
shape_distributions:
  - - "uniform(512, 2048, 3)"   # 张量 A, 维度 0
    - "uniform(512, 2048, 3)"   # 张量 A, 维度 1
  - - "uniform(512, 2048, 3)"   # 张量 B, 维度 0
    - "uniform(512, 2048, 3)"   # 张量 B, 维度 1

# 数据类型（可多个）
dtypes:
  - float16
  - float32

# 设备（可多个）
devices:
  - nvidia
  - cpu

# 算子特定参数
operator_params:
  stride: 1      # Conv2D 使用
  padding: 0     # Conv2D 使用

# 可选: 限制最大组合数
max_combinations: 50

# 可选: 随机种子（保证可复现）
random_seed: 42
```

### 完整示例: `test_templates/matmul_uniform.yaml`

```yaml
operator: matmul

shape_distributions:
  - - "uniform(512, 4096, 4)"
    - "uniform(512, 4096, 4)"
  - - "uniform(512, 4096, 4)"
    - "uniform(512, 4096, 4)"

dtypes: [float16, float32]
devices: [nvidia]
max_combinations: 50
random_seed: 42
```

---

## 🔧 命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--config, -c` | YAML 配置文件路径 | `test_templates/matmul.yaml` |
| `--operator, -op` | 算子名称 | `matmul`, `add`, `conv2d` |
| `--shape-distribution, -sd` | Shape 分布表达式 | `"uniform(512,2048,4)"` |
| `--dtype` | 数据类型 | `float16`, `float32` |
| `--device` | 目标设备 | `nvidia`, `cpu` |
| `--seed` | 随机种子 | `42` |
| `--output, -o` | 输出目录或文件 | `./generated_tests/` |
| `--max-combinations` | 最大组合数限制 | `50` |
| `--stride` | Conv2D stride | `1` |
| `--padding` | Conv2D padding | `1` |

---

## 📁 目录结构

```
InfiniMetrics/
├── scripts/
│   ├── generate_test_cases.py          # 主脚本
│   ├── README.md                        # 本文档
│   └── test_templates/                  # 配置模板
│       ├── matmul_uniform.yaml          # 均匀分布
│       ├── matmul_geometric.yaml        # 几何级数
│       ├── matmul_log_uniform.yaml      # 对数均匀
│       ├── conv2d_resnet.yaml           # ResNet 风格
│       └── add_elementwise.yaml         # 逐元素加法
└── generated_tests/                     # 生成的测试文件（输出目录）
    ├── format_input_*.json
    └── ...
```

---

## 🎯 工作流程

```bash
# 1. 生成测试用例
python scripts/generate_test_cases.py \
    --config test_templates/matmul_uniform.yaml \
    --output ./generated_tests/

# 2. 查看生成的文件
ls ./generated_tests/
# format_input_operator_InfiniCore_Matmul_*.json

# 3. 运行测试
python main.py ./generated_tests/

# 4. 查看结果
ls ./summary_output/
# dispatcher_summary_*.json
```

---

## 📝 输出文件格式

生成的 JSON 文件格式与 `format_input_*.json` 一致：

```json
{
  "run_id": "test.matmul.float16.nvidia.1024x1024_1024x1024",
  "testcase": "operator.InfiniCore.Matmul.float16_nvidia_1024x1024_1024x1024",
  "config": {
    "operator": "matmul",
    "device": "nvidia",
    "inputs": [
      {"name": "in_0", "shape": [1024, 1024], "dtype": "float16"},
      {"name": "in_1", "shape": [1024, 1024], "dtype": "float16"}
    ],
    "outputs": [{"name": "output", "shape": [1024, 1024], "dtype": "float16"}],
    "warmup_iterations": 10,
    "measured_iterations": 100,
    "tolerance": {"atol": 1e-3, "rtol": 1e-3}
  },
  "metrics": [
    {"name": "operator.latency"},
    {"name": "operator.tensor_accuracy"},
    {"name": "operator.flops"},
    {"name": "operator.bandwidth"}
  ]
}
```

---

## 🤔 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

本项目遵循 InfiniMetrics 的许可证。
