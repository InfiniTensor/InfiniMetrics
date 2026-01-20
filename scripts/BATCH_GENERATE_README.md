# Batch Generate Script - 使用说明

## 简介

`batch_generate.py` 是一个简单的命令行工具，用于批量生成InfiniMetrics测试用例。无需复杂的Web界面或YAML配置，直接通过命令行参数生成。

## 特点

1. ✅ **简单直接** - 单条命令生成N个测试用例
2. ✅ **自动打包** - 自动创建ZIP文件，包含所有配置和数据
3. ✅ **避免冲突** - 使用微秒级时间戳确保文件名唯一
4. ✅ **形状变化** - 支持none/random/progressive三种模式
5. ✅ **混合分布** - 每个输入可以使用不同的分布类型

## 快速开始

### 1. 基础用法 - 生成固定形状的测试用例

```bash
python scripts/batch_generate.py \
    --operator matmul \
    --count 10 \
    --shape-base 1024 1024 \
    --dtype float16 \
    --output ./batch_output
```

**输出**:
- `batch_output/matmul.batch.20260120_123456_123456.zip` (1.13 MB)
- 包含: 10个JSON配置 + 20个数据文件

### 2. 随机形状变化

```bash
python scripts/batch_generate.py \
    --operator matmul \
    --count 20 \
    --shape-base 1024 1024 \
    --shape-range 512 \
    --shape-variation random \
    --dtype float16 \
    --output ./batch_output
```

**效果**: 每个测试用例的形状在 [512, 1536] 范围内随机变化

### 3. 渐进式形状增长

```bash
python scripts/batch_generate.py \
    --operator matmul \
    --count 30 \
    --shape-base 512 512 \
    --shape-range 256 \
    --shape-variation progressive \
    --dtype float16 \
    --output ./batch_output
```

**效果**: 形状线性增长
- Test 0: [512, 512]
- Test 1: [768, 768]
- Test 2: [1024, 1024]
- ...

### 4. 混合分布

```bash
python scripts/batch_generate.py \
    --operator matmul \
    --count 15 \
    --shape-base 1024 1024 \
    --mixed-distributions \
    --distributions uniform normal \
    --dtype float16 \
    --output ./batch_output
```

**效果**:
- Input A 使用 uniform 分布
- Input B 使用 normal 分布

## 输出结构

解压ZIP文件后：

```
matmul.batch.20260120_123456_123456/
├── format_input_matmul_batch.0000.json
├── format_input_matmul_batch.0001.json
├── format_input_matmul_batch.0002.json
├── format_input_matmul_batch.0003.json
├── format_input_matmul_batch.0004.json
└── data/
    ├── a_1024x1024_fp16_20260120_123456_123456.npy
    ├── b_1024x1024_fp16_20260120_123456_123789.npy
    ├── a_1024x1024_fp16_20260120_123456_124012.npy
    └── ... (每个测试用例2个数据文件)
```

## 使用生成的测试用例

```bash
# 解压ZIP文件
unzip matmul.batch.20260120_123456_123456.zip -d ./test_cases

# 进入目录
cd test_cases/matmul.batch.20260120_123456_123456/

# 运行单个测试
python /path/to/InfiniMetrics/main.py format_input_matmul_batch.0000.json

# 批量运行所有测试
for json_file in format_input_*.json; do
    echo "Running $json_file..."
    python /path/to/InfiniMetrics/main.py "$json_file"
done
```

## CLI 参数说明

### 必需参数

- `--operator` / `-op`: 算子类型 (matmul, add, conv2d)
- `--count` / `-n`: 生成测试用例的数量
- `--shape-base`: 基础形状 (例如: 1024 1024 或 1024 1024 1024)

### 可选参数

#### 形状变化

- `--shape-variation`: 变化模式 (none, random, progressive)
- `--shape-range`: 变化范围 (默认: 0)

#### 数据类型

- `--dtype`: 数据类型 (float16, float32, float64, int32, int64)
- `--format`: 文件格式 (.npy, .pt, .pth)

#### 分布选项

- `--mixed-distributions`: 启用混合分布模式
- `--distributions`: 指定分布类型列表
- `--seed`: 随机种子（用于可重复性）

#### 输出选项

- `--output` / `-o`: 输出目录 (默认: ./batch_output)
- `--no-zip`: 跳过ZIP打包
- `--keep-files`: 保留未打包的原始文件

#### 性能参数

- `--device`: 目标设备 (默认: nvidia)
- `--warmup`: 预热迭代次数 (默认: 10)
- `--measured`: 测量迭代次数 (默认: 100)

## 形状变化模式详解

### 1. none - 无变化

```bash
--shape-base 1024 1024 --shape-variation none
```

所有测试用例使用相同形状 [1024, 1024]

### 2. random - 随机变化

```bash
--shape-base 1024 1024 --shape-variation random --shape-range 512
```

每个维度在 [512, 1536] 范围内随机变化：
- Test 0: [1024, 1024]
- Test 1: [1280, 768]
- Test 2: [800, 1408]
- ...

### 3. progressive - 渐进增长

```bash
--shape-base 512 512 --shape-variation progressive --shape-range 256
```

形状线性增长：
- Test 0: [512, 512]
- Test 1: [768, 768]
- Test 2: [1024, 1024]
- Test 3: [1280, 1280]
- ...

## 操作符特定的形状说明

### matmul

```bash
# 2个参数: [M, K] - 两个矩阵都使用相同的shape
--shape-base 1024 1024

# 3个参数: [M, K, N] - 完全指定
--shape-base 1024 1024 2048
```

生成的输入：
- Input A: [M, K]
- Input B: [K, N]
- Output: [M, N]

### add

```bash
--shape-base 1024 1024 2048  # 至少2维
```

两个输入使用相同形状

### conv2d

```bash
--shape-base N C_in H_in W_in  # 至少4维
```

生成的输入：
- Input: [N, C_in, H_in, W_in]
- Weight: [C_out, C_in, 3, 3] (自动计算)

## 分布类型

支持的15种分布类型：

1. **uniform** - 均匀分布 (默认)
2. **normal** - 正态分布
3. **standard_normal** - 标准正态分布
4. **randint** - 随机整数
5. **lognormal** - 对数正态分布
6. **exponential** - 指数分布
7. **laplace** - 拉普拉斯分布
8. **cauchy** - 柯西分布
9. **poisson** - 泊松分布
10. **zipf** - 齐普夫分布
11. **ones** - 全1矩阵
12. **zeros** - 全0矩阵
13. **identity** - 单位矩阵
14. **orthogonal** - 正交矩阵
15. **sparse** - 稀疏矩阵

## 常见用例

### 生成性能测试数据集

```bash
python scripts/batch_generate.py \
    --operator matmul \
    --count 100 \
    --shape-base 2048 2048 2048 \
    --dtype float16 \
    --output ./performance_tests
```

### 生成形状扫描数据集

```bash
python scripts/batch_generate.py \
    --operator matmul \
    --count 50 \
    --shape-base 512 512 \
    --shape-range 512 \
    --shape-variation progressive \
    --dtype float16 \
    --output ./shape_sweep
```

### 生成混合分布测试

```bash
python scripts/batch_generate.py \
    --operator matmul \
    --count 20 \
    --shape-base 1024 1024 \
    --mixed-distributions \
    --distributions orthogonal normal \
    --dtype float32 \
    --output ./mixed_dist_tests
```

### 可重复的测试数据

```bash
python scripts/batch_generate.py \
    --operator matmul \
    --count 10 \
    --shape-base 768 768 \
    --dtype float16 \
    --seed 42 \
    --output ./reproducible_tests
```

## 故障排除

### 问题1: 生成的数据文件被覆盖

**原因**: 使用了旧版本的 `input_generator.py`

**解决**: 确保使用更新后的版本，文件名包含微秒时间戳：
```
a_1024x1024_fp16_20260120_123456_123456.npy
```

### 问题2: 形状验证失败

**原因**: MatMul要求 [M, K] @ [K, N]，所以 Input A 的最后一维要等于 Input B 的倒数第二维

**解决**:
```bash
# 正确
--shape-base 1024 1024  # [M=1024, K=1024] @ [K=1024, N=1024]

# 错误
--shape-base 1024 768    # [M=1024, K=1024] @ [K=768, N=???] ❌
```

### 问题3: ZIP文件太大

**解决**:
1. 使用更小的形状
2. 使用 float16 而不是 float32
3. 生成较少的测试用例

## 对比Web应用

| 特性 | Web应用 | batch_generate.py |
|------|---------|-------------------|
| 复杂度 | 高 | 低 |
| 依赖 | Docker, FastAPI, React | Python标准库 |
| 界面 | 浏览器GUI | 命令行 |
| 适用场景 | 频繁交互调试 | 一次性批量生成 |
| 部署 | 复杂 | 简单（单文件脚本） |
| 学习曲线 | 陡峭 | 平缓 |

## 总结

`batch_generate.py` 提供了一个简单、高效的方式来批量生成InfiniMetrics测试用例：

- ✅ 无需Web界面，一行命令完成
- ✅ 自动打包ZIP，方便分发
- ✅ 支持形状变化和混合分布
- ✅ 微秒时间戳避免文件冲突
- ✅ 可直接集成到CI/CD流程

**推荐使用场景**：
- 一次性生成大量测试数据
- 自动化测试流程
- 性能基准测试
- 形状扫描实验
