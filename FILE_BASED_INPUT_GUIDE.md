# 随机输入生成与文件加载功能 - 使用指南

## 功能概述

本功能为 InfiniMetrics 添加了随机输入生成能力,支持将随机数据保存到文件,并在测试配置中引用文件路径。InfiniCore 侧支持从文件加载数据进行测试。

## 核心特性

1. **独立工具脚本** - 通过 CLI 工具生成随机输入数据
2. **时间戳命名** - 使用时间戳生成唯一文件名,便于数据对比测试
3. **相对路径基准** - 支持配置 `data_base_dir` 来管理相对路径
4. **多种分布类型** - 支持均匀分布、正态分布、随机整数
5. **多种文件格式** - 支持 .npy, .pt, .pth 格式
6. **向后兼容** - 完全兼容现有的 shape 模式配置

## 快速开始

### 1. 生成随机输入

使用 CLI 工具从现有配置生成随机输入:

```bash
cd InfiniMetrics
python infinimetrics/utils/input_generator_cli.py \
    --config test_file_based_input.json \
    --output-dir ./generated_data \
    --seed 42 \
    --output-json test_with_files.json
```

这会:
- 根据 `test_file_based_input.json` 中的配置生成随机数据
- 将数据保存到 `./generated_data` 目录
- 使用随机种子 42 确保可重现性
- 生成新的配置文件 `test_with_files.json`,其中包含生成的文件路径

### 2. 配置文件格式

#### 使用 shape 模式(现有方式)

```json
{
    "config": {
        "operator": "add",
        "inputs": [
            {"name": "a", "shape": [1024, 1024], "dtype": "float16"},
            {"name": "b", "shape": [1024, 1024], "dtype": "float16"}
        ]
    }
}
```

#### 使用 file 模式(新功能)

```json
{
    "config": {
        "operator": "add",
        "data_base_dir": "./generated_data",
        "inputs": [
            {
                "name": "a",
                "file_path": "input_a_1024x1024_float16_20250119_143025.npy",
                "dtype": "float16",
                "shape": [1024, 1024]
            },
            {
                "name": "b",
                "file_path": "input_b_1024x1024_float16_20250119_143026.npy",
                "dtype": "float16",
                "shape": [1024, 1024]
            }
        ]
    }
}
```

#### 混合模式

```json
{
    "config": {
        "operator": "matmul",
        "data_base_dir": "./pretrained",
        "inputs": [
            {
                "name": "weight",
                "file_path": "weight_layer1.npy",
                "dtype": "float32"
            },
            {
                "name": "activation",
                "shape": [1024, 1024],
                "dtype": "float16"
            }
        ]
    }
}
```

### 3. 运行测试

使用生成的文件路径运行测试:

```bash
python main.py test_with_files.json
```

## CLI 工具详细说明

### 基本用法

```bash
python infinimetrics/utils/input_generator_cli.py --config <config.json>
```

### 参数说明

- `--config, -c` (必需): 输入配置文件路径
- `--output-dir, -o`: 输出目录(默认: `./generated_data`)
- `--seed, -s`: 随机种子,用于可重现性(默认: None)
- `--output-json`: 保存更新后的配置到指定文件
- `--verbose, -v`: 启用详细输出

### 示例

#### 示例 1: 基本生成

```bash
python infinimetrics/utils/input_generator_cli.py \
    --config format_input_add.json
```

#### 示例 2: 指定输出目录和种子

```bash
python infinimetrics/utils/input_generator_cli.py \
    --config format_input_add.json \
    --output-dir ./my_test_data \
    --seed 123
```

#### 示例 3: 生成并保存新配置

```bash
python infinimetrics/utils/input_generator_cli.py \
    --config format_input_add.json \
    --output-dir ./generated_data \
    --seed 42 \
    --output-json format_input_with_paths.json
```

## Python API 使用

### RandomInputGenerator 类

```python
from infinimetrics.utils.input_generator import RandomInputGenerator

# 创建生成器
generator = RandomInputGenerator(
    output_dir="./test_data",
    default_format=".npy",
    seed=42
)

# 生成单个输入
config = generator.generate_input_config(
    base_name="input_a",
    shape=[2048, 2048],
    dtype="float16",
    distribution="uniform",
    low=-1.0,
    high=1.0
)

print(config)
# 输出:
# {
#     "name": "input_a",
#     "file_path": "./test_data/input_a_2048x2048_float16_20250119_143025.npy",
#     "dtype": "float16",
#     "shape": [2048, 2048],
#     "_metadata": {...}
# }
```

### 批量生成函数

```python
from infinimetrics.utils.input_generator import generate_random_inputs_from_config

inputs_config = [
    {
        "name": "input_a",
        "shape": [1024, 1024],
        "dtype": "float16",
        "_random": {
            "distribution": "uniform",
            "params": {"low": -1.0, "high": 1.0}
        }
    },
    {
        "name": "input_b",
        "shape": [1024, 1024],
        "dtype": "float16",
        "_random": {
            "distribution": "normal",
            "params": {"mean": 0.0, "std": 0.5}
        }
    }
]

generated = generate_random_inputs_from_config(
    inputs_config=inputs_config,
    output_dir="./generated_data",
    seed=42
)

# generated 包含带 file_path 的配置
```

## 支持的分布类型

### 1. Uniform Distribution (均匀分布)

```python
config = generator.generate_input_config(
    base_name="input",
    shape=[100, 100],
    dtype="float32",
    distribution="uniform",
    low=-1.0,   # 默认: 0.0
    high=1.0,   # 默认: 1.0
    scale=1.0,  # 可选: 缩放因子
    bias=0.0    # 可选: 偏移量
)
```

### 2. Normal Distribution (正态分布)

```python
config = generator.generate_input_config(
    base_name="input",
    shape=[100, 100],
    dtype="float32",
    distribution="normal",
    mean=0.0,   # 默认: 0.0
    std=1.0     # 默认: 1.0
)
```

### 3. Random Integer (随机整数)

```python
config = generator.generate_input_config(
    base_name="input",
    shape=[100, 100],
    dtype="int32",
    distribution="randint",
    low=-100,   # 下限(包含)
    high=100    # 上限(不包含)
)
```

## 文件命名规则

生成的文件使用时间戳命名,格式为:

```
{name}_{shape}_{dtype}_{timestamp}{extension}
```

示例:
- `input_a_1024x1024_float16_20250119_143025.npy`
- `weight_2048x1024_float32_20250119_143026.pt`

## 相对路径处理

当使用相对路径时,可以配置 `data_base_dir` 作为基准目录:

```json
{
    "config": {
        "data_base_dir": "./generated_data",
        "inputs": [
            {
                "name": "a",
                "file_path": "input_a.npy",  // 相对路径
                "dtype": "float16"
            }
        ]
    }
}
```

实际解析为: `./generated_data/input_a.npy`

如果不配置 `data_base_dir`,默认使用当前目录 `.` 作为基准目录。

## 支持的文件格式

- **`.npy`** - NumPy 数组格式(推荐,默认)
- **`.pt`** - PyTorch 张量格式
- **`.pth`** - PyTorch 模型/checkpoint 格式

## 支持的数据类型

- `float16` - 16位浮点数
- `float32` - 32位浮点数
- `float64` - 64位浮点数
- `bfloat16` - 脑浮点数(使用 float32 生成)
- `int8` - 8位整数
- `int16` - 16位整数
- `int32` - 32位整数
- `int64` - 64位整数
- `uint8` - 8位无符号整数
- `bool` - 布尔值

## 错误处理

### 文件不存在

如果配置的文件路径不存在,InfiniCore 会抛出 `FileNotFoundError`:

```
FileNotFoundError: File not found: /path/to/file.npy
```

### 形状不匹配

如果文件中的数据形状与配置的 `shape` 不匹配,会抛出 `ValueError`:

```
ValueError: Shape mismatch: expected [1024, 1024], got [2048, 1024]
```

### 数据类型不匹配

如果文件中的数据类型与配置不同,会自动转换为目标类型。

## 使用场景

### 1. 性能基准测试

使用固定数据集进行性能测试,确保可重现性:

```bash
# 一次性生成测试数据
python infinimetrics/utils/input_generator_cli.py \
    --config benchmark_config.json \
    --seed 42 \
    --output-json benchmark_fixed.json

# 多次运行测试,使用相同数据
python main.py benchmark_fixed.json
```

### 2. 数据对比测试

使用时间戳命名生成多组数据,进行对比:

```bash
# 生成第一组数据
python infinimetrics/utils/input_generator_cli.py \
    --config test.json --seed 42 \
    --output-json test_v1.json

# 生成第二组数据
python infinimetrics/utils/input_generator_cli.py \
    --config test.json --seed 43 \
    --output-json test_v2.json

# 对比两次测试结果
```

### 3. 使用预训练权重

测试模型时使用真实的预训练权重:

```json
{
    "config": {
        "operator": "linear",
        "data_base_dir": "./pretrained_models",
        "inputs": [
            {
                "name": "weight",
                "file_path": "layer1_weight.npy",
                "dtype": "float32"
            },
            {
                "name": "bias",
                "file_path": "layer1_bias.npy",
                "dtype": "float32"
            },
            {
                "name": "input",
                "shape": [1024, 768],
                "dtype": "float16"
            }
        ]
    }
}
```

## 实现细节

### 修改的文件

1. **InfiniMetrics/infinimetrics/common/constants.py**
   - 添加 `TensorSpec.FILE_PATH` 常量

2. **InfiniCore/test/infinicore/framework/utils/load_utils.py**
   - 修改 `_dict_to_spec()` 函数,支持 file_path 和 init_mode

3. **InfiniMetrics/infinimetrics/operators/infinicore_adapter.py**
   - 修改 `_convert_to_request()`,支持文件路径传递和相对路径解析

### 新建的文件

4. **InfiniMetrics/infinimetrics/utils/input_generator.py**
   - `RandomInputGenerator` 类实现

5. **InfiniMetrics/infinimetrics/utils/input_generator_cli.py**
   - CLI 工具实现

## 验证安装

运行验证脚本检查实现:

```bash
cd InfiniMetrics
python verify_implementation.py
```

应该看到所有检查都通过:

```
✓ All checks passed! Implementation looks complete.
```

## 故障排除

### 问题: ModuleNotFoundError: No module named 'numpy'

**解决方案**: 安装依赖

```bash
pip install numpy torch
```

### 问题: 找不到生成的文件

**解决方案**: 检查相对路径和 `data_base_dir` 配置

```json
{
    "config": {
        "data_base_dir": "./generated_data",  // 确保此目录存在
        ...
    }
}
```

### 问题: 测试失败,提示文件不存在

**解决方案**:
1. 确认文件已生成: `ls ./generated_data/`
2. 确认文件路径在 JSON 中正确
3. 使用绝对路径测试以排除路径问题

## 最佳实践

1. **使用随机种子**: 始终指定 `--seed` 参数确保可重现性
2. **组织数据目录**: 为不同的测试用例创建独立的输出目录
3. **验证生成的数据**: 生成后检查文件大小和内容
4. **版本控制**: 不要将生成的数据文件提交到版本控制,只提交配置
5. **文档化配置**: 在配置文件的注释中记录数据来源和用途

## 示例: 完整工作流

```bash
# 1. 创建配置文件
cat > my_test.json << EOF
{
    "run_id": "test.my.custom",
    "testcase": "operator.InfiniCore.Add.Custom",
    "config": {
        "operator": "add",
        "device": "nvidia",
        "inputs": [
            {"name": "a", "shape": [512, 512], "dtype": "float16"},
            {"name": "b", "shape": [512, 512], "dtype": "float16"}
        ]
    },
    "metrics": [{"name": "operator.latency"}]
}
EOF

# 2. 生成随机输入
python infinimetrics/utils/input_generator_cli.py \
    --config my_test.json \
    --output-dir ./my_test_data \
    --seed 42 \
    --output-json my_test_with_files.json

# 3. 运行测试
python main.py my_test_with_files.json

# 4. 查看结果
cat summary_output/dispatcher_summary_*.json
```
