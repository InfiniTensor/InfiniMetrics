# Benchmark模式 - 性能测试形状生成

## 功能说明

`benchmark` 模式专门为性能基准测试设计，生成一组精心挑选的测试形状，用于全面测试不同场景下的算子性能。

## 使用方法

### 1. 查看所有可用的benchmark形状

```bash
python scripts/batch_generate.py --list-benchmark-shapes
```

输出示例：
```
============================================================
Available Benchmark Shapes (MatMul & Element-wise)
============================================================
Total: 17 preset shapes

   0.   64 ×   64
   1.  128 ×  128
   2.  256 ×  256
   3.  512 ×  512
   4.  768 ×  768
   5. 1024 ×  768
   6.  768 × 1024
   7. 1024 × 1024
   8. 1536 × 1536
   9. 2048 × 2048
  10. 3072 ×  3072
  11. 4096 ×  4096
  12. 1000 × 1000
  13. 1008 × 1008
  14. 1020 × 1020
  15.  128 × 4096
  16. 4096 ×  128
============================================================
```

### 2. 生成benchmark测试用例

```bash
# 生成前5个benchmark形状（小到中等尺寸）
python scripts/batch_generate.py \
    --operator matmul \
    --count 5 \
    --shape-base 0 0 \
    --shape-variation benchmark \
    --dtype float16 \
    --output ./benchmark_small \
    --seed 42

# 生成所有17个benchmark形状（全面性能测试）
python scripts/batch_generate.py \
    --operator matmul \
    --count 17 \
    --shape-base 0 0 \
    --shape-variation benchmark \
    --dtype float16 \
    --output ./benchmark_full \
    --seed 42

# Element-wise算子使用相同的形状
python scripts/batch_generate.py \
    --operator add \
    --count 10 \
    --shape-base 0 0 \
    --shape-variation benchmark \
    --dtype float16 \
    --output ./benchmark_add \
    --seed 42
```

### 3. 结合种子实现可重复性

```bash
# 使用相同种子生成，确保数据一致
python scripts/batch_generate.py \
    --operator matmul \
    --count 5 \
    --shape-base 0 0 \
    --shape-variation benchmark \
    --dtype float16 \
    --seed 12345 \
    --output ./benchmark_run1

# 多次运行使用相同种子，生成的测试数据完全相同
python scripts/batch_generate.py \
    --operator matmul \
    --count 5 \
    --shape-base 0 0 \
    --shape-variation benchmark \
    --dtype float16 \
    --seed 12345 \
    --output ./benchmark_run2
```

## Benchmark形状设计原理

### 测试场景覆盖

预设的17个形状覆盖以下性能特征：

#### 1. 小矩阵（Cache & Latency测试）
- **64×64**: 极小尺寸，测试缓存和延迟效应
- **128×128**: 小尺寸，接近L1缓存
- **256×256**: 小到中等尺寸，L1/L2缓存边界

#### 2. 中等矩阵（Balanced测试）
- **512×512**: 中等尺寸，计算和内存访问平衡
- **768×768**: 中等尺寸，接近1024
- **1024×768**: 矩形矩阵，测试非对称访问模式
- **768×1024**: 矩形矩阵，反向长宽比

#### 3. 大矩阵（Memory Bandwidth测试）
- **1024×1024**: 标准大规模基准
- **1536×1536**: 大规模，接近2048
- **2048×2048**: 常用大规模测试
- **3072×3072**: 超大规模，内存带宽限制
- **4096×4096**: 极大规模，压力测试

#### 4. 非2次幂（Alignment测试）
- **1000×1000**: 十进制友好尺寸
- **1008×1008**: 对齐但非2次幂（1008 = 16 × 63）
- **1020×1020**: 接近2次幂但不是

#### 5. 极端长宽比（Access Pattern测试）
- **128×4096**: 瘦高矩阵
- **4096×128**: 矮胖矩阵

### 性能指标分析

通过这17个形状，可以分析：

1. **计算效率**: 不同尺寸下的FLOPS利用率
2. **内存带宽**: 大矩阵下的带宽瓶颈
3. **缓存效率**: 小/中矩阵的缓存命中率
4. **对齐效应**: 2次幂 vs 非2次幂的性能差异
5. **访问模式**: 方形 vs 矩形矩阵的性能差异

## 使用场景

### 1. 算子性能基准测试

```bash
# 全面测试MatMul性能
python scripts/batch_generate.py \
    --operator matmul \
    --count 17 \
    --shape-base 0 0 \
    --shape-variation benchmark \
    --dtype float16 \
    --output ./matmul_benchmark \
    --seed 42

# 生成后批量运行测试
cd matmul_benchmark/matmul.batch.*
for json in format_input_*.json; do
    echo "Testing $json..."
    python /path/to/InfiniMetrics/main.py "$json"
done
```

### 2. 不同实现的性能对比

```bash
# 使用相同种子生成两批数据，对比两个实现
# 实现1
python scripts/batch_generate.py \
    --operator matmul \
    --count 10 \
    --shape-base 0 0 \
    --shape-variation benchmark \
    --seed 999 \
    --output ./impl1_test

# 实现2（使用相同种子）
python scripts/batch_generate.py \
    --operator matmul \
    --count 10 \
    --shape-base 0 0 \
    --shape-variation benchmark \
    --seed 999 \
    --output ./impl2_test

# 对比运行结果...
```

### 3. 回归测试

```bash
# 使用固定的seed和count，建立性能基准
# 每次代码变更后重新运行，确保性能不退化
python scripts/batch_generate.py \
    --operator matmul \
    --count 17 \
    --shape-base 0 0 \
    --shape-variation benchmark \
    --seed 42 \
    --output ./regression_test
```

### 4. 硬件/驱动升级验证

```bash
# 升级前后使用相同测试，验证性能提升
python scripts/batch_generate.py \
    --operator matmul \
    --count 10 \
    --shape-base 0 0 \
    --shape-variation benchmark \
    --seed 12345 \
    --output ./hardware_test
```

## 与其他模式对比

### vs random模式

| 特性 | benchmark | random |
|------|-----------|--------|
| **形状覆盖** | 精心设计，覆盖所有场景 | 随机生成，可能遗漏某些场景 |
| **可重复性** | 完全可重复（相同seed） | 可重复（相同seed） |
| **测试目的** | 性能基准测试 | 随机鲁棒性测试 |
| **形状数量** | 固定17个预设 | 无限制 |

**使用random模式的场景**：
- 需要大量随机形状进行压力测试
- 不关心具体测试什么，只看不崩溃
- 测试算子对不同尺寸的适应性

**使用benchmark模式的场景**：
- 需要全面测试不同尺寸的性能
- 需要对比不同实现/版本的性能
- 需要建立性能基准线

### vs progressive模式

| 特性 | benchmark | progressive |
|------|-----------|-------------|
| **形状模式** | 预设的17个形状 | 线性增长：base + range×index |
| **灵活性** | 固定形状 | 可自定义base和range |
| **测试覆盖** | 覆盖多种性能特征 | 单一趋势（从小到大） |

**使用progressive模式的场景**：
- 想测试某个特定范围的性能增长
- 需要自定义尺寸范围
- 只关心尺寸对性能的影响趋势

## 最佳实践

### 1. 建立性能基准

```bash
# 建立基准性能数据
python scripts/batch_generate.py \
    --operator matmul \
    --count 17 \
    --shape-base 0 0 \
    --shape-variation benchmark \
    --seed 42 \
    --dtype float16 \
    --output ./baseline

# 运行并记录结果
cd baseline/matmul.batch.*
for json in format_input_*.json; do
    python /path/to/InfiniMetrics/main.py "$json" > ../results_$(basename $json .json).txt
done

# 保存为baseline_results.txt
```

### 2. 定期回归测试

```bash
# 每次代码提交后运行
python scripts/batch_generate.py \
    --operator matmul \
    --count 17 \
    --shape-base 0 0 \
    --shape-variation benchmark \
    --seed 42 \
    --dtype float16 \
    --output ./current_test

# 对比current_test和baseline的性能差异
```

### 3. 多设备对比

```bash
# NVIDIA GPU
python scripts/batch_generate.py \
    --operator matmul \
    --count 17 \
    --shape-base 0 0 \
    --shape-variation benchmark \
    --seed 42 \
    --device nvidia \
    --dtype float16 \
    --output ./nvidia_test

# Cambricon MLU
python scripts/batch_generate.py \
    --operator matmul \
    --count 17 \
    --shape-base 0 0 \
    --shape-variation benchmark \
    --seed 42 \
    --device cambricon \
    --dtype float16 \
    --output ./cambricon_test

# 对比两个设备的性能
```

### 4. 精度测试

```bash
# 测试float16 vs float32的精度差异
python scripts/batch_generate.py \
    --operator matmul \
    --count 10 \
    --shape-base 0 0 \
    --shape-variation benchmark \
    --seed 42 \
    --dtype float16 \
    --output ./fp16_test

python scripts/batch_generate.py \
    --operator matmul \
    --count 10 \
    --shape-base 0 0 \
    --shape-variation benchmark \
    --seed 42 \
    --dtype float32 \
    --output ./fp32_test

# 对比精度和性能
```

## 参数说明

### --shape-variation benchmark
启用benchmark模式，使用预设的性能测试形状

### --count
指定生成的测试用例数量：
- 必须≤17（benchmark形状总数）
- 建议使用5-10个形状进行快速测试
- 使用全部17个进行完整性能测试

### --shape-base 0 0
在benchmark模式下，shape-base参数被忽略
但仍然需要提供以满足参数解析要求
建议使用 `0 0` 表示"使用benchmark预设"

### --seed
指定随机种子：
- **重要**：使用相同种子可以生成完全相同的测试数据
- 建议固定种子（如42、12345）以便结果可重复
- 不指定seed则每次生成不同的数据

## 输出结构

```
benchmark_test/
└── matmul.batch.20260120_161102_034190/
    ├── format_input_matmul_batch.0000.json  # 64×64
    ├── format_input_matmul_batch.0001.json  # 128×128
    ├── format_input_matmul_batch.0002.json  # 256×256
    ├── format_input_matmul_batch.0003.json  # 512×512
    ├── format_input_matmul_batch.0004.json  # 768×768
    └── data/
        ├── a_64x64_float16_*.npy
        ├── b_64x64_float16_*.npy
        ├── a_128x128_float16_*.npy
        ├── b_128x128_float16_*.npy
        └── ... (更多数据文件)
```

## 性能分析建议

### 1. 按尺寸分组分析
- 小尺寸（0-2）: Cache & Latency
- 中尺寸（3-6）: Balanced
- 大尺寸（7-11）: Memory Bandwidth
- 其他（12-16）: 特殊场景

### 2. 关注关键指标
- 小尺寸：延迟（latency）
- 中尺寸：计算效率（FLOPS）
- 大尺寸：内存带宽（GB/s）

### 3. 绘制性能曲线
```python
import matplotlib.pyplot as plt

shapes = [(64,64), (128,128), (256,256), ...]
latencies = [...]

plt.plot(range(len(shapes)), latencies, 'o-')
plt.xlabel('Benchmark Test Case')
plt.ylabel('Latency (ms)')
plt.title('MatMul Performance Benchmark')
plt.show()
```

## 总结

`benchmark` 模式提供了一个**标准化、可重复、全面覆盖**的性能测试方案：

✅ **精心设计**：17个形状覆盖所有关键性能场景
✅ **可重复性**：相同seed生成完全相同的数据
✅ **标准化**：统一的测试基准，便于对比
✅ **全面性**：从cache到带宽，从方形到矩形，从2次幂到非2次幂

非常适合用于：
- 算子性能基准测试
- 不同实现的性能对比
- 回归测试
- 硬件/驱动升级验证
