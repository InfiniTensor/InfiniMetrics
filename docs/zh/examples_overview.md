# 使用示例

本节提供在不同场景下使用 InfiniMetrics 的实际示例。

## 示例类别

### 硬件基准测试
- [硬件基准测试示例](./hardware_benchmarks.md) - 内存、STREAM 和缓存测试

### 推理评估
- [推理示例](./inference_evaluation.md) - InfiniLM 和 vLLM 推理测试

### 高级用法
- [高级用法](./advanced_usage.md) - 多个测试、自定义配置、批处理

## 快速示例

### 运行单个测试

```bash
python main.py format_input_comprehensive_hardware.json
```

### 运行多个测试

```bash
# 运行目录中的所有 JSON 配置
python main.py ./test_configs/
```

### 详细输出

```bash
python main.py input.json --verbose
```

### 自定义输出目录

```bash
python main.py input.json --output ./my_results
```

## 示例配置文件

仓库包含几个示例配置：

- `format_input_comprehensive_hardware.json` - 综合硬件基准测试
- 其他示例可以在各个示例文档中找到

## 创建自己的配置

1. 从示例配置开始
2. 根据需要修改参数
3. 使用 `--verbose` 运行以验证
4. 在 `./output/` 目录中检查输出

## 后续步骤

- 探索特定的示例类别
- 参阅[配置指南](./configuration.md)了解参数详情
- 参考[测试类型](./hardware_tests.md)了解可用测试
