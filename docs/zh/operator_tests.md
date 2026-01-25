# 算子测试

算子测试测量单个计算操作和内核的性能，提供对特定计算模式的详细见解。

## InfiniCore 算子测试

| 测试名称 | 框架 | 描述 |
|-----------|----------|-------------|
| `operator.infinicore.*` | InfiniCore | 单个算子性能、FLOPS 计算 |

## 支持的算子

InfiniCore 测试支持各种操作，包括：
- 卷积 (Conv2D)
- 矩阵乘法 (MatMul)
- 逐元素操作
- 归约操作

## FLOPS 计算

算子测试包括自动 FLOPS（每秒浮点运算次数）计算：

- **Conv2D**: `2 * H_out * W_out * C_in * K_h * K_w * C_out`
- **MatMul**: `2 * M * N * K`

## 配置示例

```json
{
    "run_id": "op_test_001",
    "testcase": "operator.infinicore.Conv2D",
    "config": {
        "input_shape": [1, 64, 224, 224],
        "kernel_shape": [64, 64, 3, 3],
        "stride": 1,
        "padding": 1,
        "output_dir": "./output"
    },
    "metrics": [
        {"name": "operator.flops"},
        {"name": "operator.latency"},
        {"name": "operator.throughput"}
    ]
}
```

## 运行算子测试

```bash
python main.py operator_test_config.json
```

## 理解结果

### FLOPS

每秒浮点运算次数 - 测量计算吞吐量。

- 数值越高表示性能越好
- 与理论峰值比较（FLOPs/周期 * 核心数 * 时钟频率）

### 延迟

完成单个操作所需的时间。

- 数值越低表示性能越好
- 对于实时应用很重要

### 吞吐量

单位时间内完成的操作数。

- 数值越高表示性能越好
- 对于批处理很重要

## 性能优化技巧

1. **批大小**: 较大的批次通常提高吞吐量
2. **张量布局**: 使用最优内存布局（NHWC vs NCHW）
3. **精度**: 考虑混合精度（FP16/BF16）以获得更好的性能
4. **算子融合**: 融合算子减少内存传输

## 添加新算子

要添加新的算子测试，请参阅[开发指南](./development.md)。

## 示例

更多算子测试配置，请参阅[使用示例](./examples_overview.md)。
