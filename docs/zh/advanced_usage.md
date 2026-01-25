# 高级用法示例

本文档涵盖 InfiniMetrics 的高级使用模式和配置。

## 示例 1：批量运行多个测试

按顺序执行多个测试配置。

### 目录设置

```
test_configs/
├── hardware_memory.json
├── hardware_stream.json
├── inference_infinilm.json
└── inference_vllm.json
```

### 运行所有测试

```bash
python main.py ./test_configs/
```

### 输出

每个测试创建自己的输出目录：
```
./output/
├── hardware.cudaUnified.MemoryBandwidth/
├── hardware.cudaUnified.STREAM/
├── infer.infinilm.direct/
└── infer.vllm.default/
```

## 示例 2：自定义输出目录

指定自定义输出位置以便更好地组织。

### 配置

```json
{
    "run_id": "test_001",
    "testcase": "hardware.cudaUnified.Comprehensive",
    "config": {
        "output_dir": "./my_results/hardware_tests"
    },
    "metrics": [...]
}
```

### 命令行替代方案

```bash
python main.py input.json --output ./my_results
```

## 示例 3：详细输出

启用详细日志记录以进行调试和分析。

### 命令

```bash
python main.py input.json --verbose
```

### 优点
- 详细的执行信息
- 指标收集进度
- 错误堆栈跟踪
- 性能计时细分

## 示例 4：组合硬件和推理测试

创建综合评估流程。

### 步骤 1：硬件测试

```bash
python main.py hardware_config.json
```

### 步骤 2：算子测试

```bash
python main.py operator_config.json
```

### 步骤 3：推理测试

```bash
python main.py inference_config.json
```

### 步骤 4：聚合结果

```bash
# 所有结果在 ./summary_output/
cat ./summary_output/dispatcher_summary_*.json
```

## 示例 5：测试不同的加速器类型

为不同硬件供应商配置测试。

### NVIDIA

```json
{
    "config": {
        "device": "nvidia"
    }
}
```

### AMD

```json
{
    "config": {
        "device": "amd"
    }
}
```

### 华为

```json
{
    "config": {
        "device": "huawei"
    }
}
```

## 示例 6：自定义指标收集

为特定需求定义自定义指标。

### 配置

```json
{
    "run_id": "custom_test_001",
    "testcase": "hardware.cudaUnified.Custom",
    "config": {
        "custom_param": "value",
        "output_dir": "./output"
    },
    "metrics": [
        {"name": "custom.metric1"},
        {"name": "custom.metric2"},
        {"name": "standard.metric"}
    ]
}
```

## 示例 7：脚本化测试执行

创建 shell 脚本以自动化测试执行。

### run_tests.sh

```bash
#!/bin/bash

# 设置输出目录
OUTPUT_DIR="./results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# 运行测试
python main.py hardware_config.json --output $OUTPUT_DIR
python main.py inference_config.json --output $OUTPUT_DIR

# 生成摘要
echo "测试完成。结果在 $OUTPUT_DIR"
```

### 运行

```bash
chmod +x run_tests.sh
./run_tests.sh
```

## 示例 8：编程执行

在 Python 中以编程方式使用 InfiniMetrics。

### script.py

```python
from infinimetrics.dispatcher import Dispatcher

# 创建调度器
dispatcher = Dispatcher()

# 加载配置
config = {
    "run_id": "prog_test_001",
    "testcase": "hardware.cudaUnified.Comprehensive",
    "config": {...},
    "metrics": [...]
}

# 运行测试
result = dispatcher.run(config)

# 访问结果
print(f"测试结果: {result}")
```

## 示例 9：结果分析

以编程方式解析和分析测试结果。

### analyze_results.py

```python
import json
import glob

# 查找所有摘要文件
summaries = glob.glob("./summary_output/dispatcher_summary_*.json")

for summary_file in summaries:
    with open(summary_file, 'r') as f:
        data = json.load(f)

    print(f"总测试数: {data['total_tests']}")
    print(f"成功: {data['successful_tests']}")
    print(f"失败: {data['failed_tests']}")

    # 处理每个结果
    for result in data['results']:
        if result['result_code'] == 0:
            with open(result['result_file'], 'r') as f:
                metrics = json.load(f)
                print(f"指标: {metrics}")
```

## 示例 10：持续基准测试

设置定期基准测试以进行性能回归测试。

### cron_job.sh

```bash
#!/bin/bash

# 每日运行基准测试
python main.py comprehensive_config.json \
    --output ./benchmark_results/$(date +%Y%m%d)

# 与基线比较
python compare_results.py \
    --current ./benchmark_results/$(date +%Y%m%d) \
    --baseline ./baseline_results
```

### Crontab 条目

```
0 2 * * * /path/to/cron_job.sh
```

## 最佳实践

### 组织
- 为不同的测试类别使用单独的目录
- 在输出目录中包含时间戳
- 保留输入配置的副本与结果一起

### 可重现性
- 对配置文件进行版本控制
- 记录硬件和软件版本
- 对随机测试使用固定的随机种子

### 性能
- 尽可能并行运行测试
- 使用批量执行处理多个测试
- 在执行期间监控系统资源

### 调试
- 对新配置始终使用 `--verbose`
- 检查输出目录中的日志文件
- 在运行测试前验证硬件状态

## 后续步骤

- 参阅[配置指南](./configuration.md)了解参数详情
- 参考[开发指南](./development.md)以扩展功能
