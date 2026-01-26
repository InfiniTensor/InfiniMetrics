# 输出格式

本文档描述测试结果的输出格式和目录结构。

## 输出目录结构

```
./summary_output/
└── dispatcher_summary_YYYYMMDD_HHMMSS.json    # 总体摘要
./output/
└── <test_case_name>/
    ├── seperated_test_result.json              # 分测试结果
    └── metrics.csv                             # 时间序列指标
```

## 摘要格式

调度器摘要文件包含总体测试执行结果：

```json
{
  "total_tests": 3,
  "successful_tests": 2,
  "failed_tests": 1,
  "results": [
    {
      "run_id": "test_run_001",
      "testcase": "hardware.cudaUnified.Comprehensive",
      "result_code": 0,
      "result_file": "./output/hardware.cudaUnified.Comprehensive/metrics.json"
    }
  ],
  "timestamp": "2026-01-22T10:56:09.338373"
}
```

### 摘要字段

| 字段 | 类型 | 描述 |
|-------|------|-------------|
| `total_tests` | int | 执行的测试总数 |
| `successful_tests` | int | 成功的测试数量 |
| `failed_tests` | int | 失败的测试数量 |
| `results` | array | 测试结果对象数组 |
| `timestamp` | string | ISO 格式的执行时间戳 |

## 测试特定输出

### 硬件基准测试输出

#### 指标 JSON

```json
{
   {
      "name": "hardware.mem_sweep_d2d",
      "type": "timeseries",
      "raw_data_url": "./mem_sweep_d2d_hardware.cudaUnified.Comprehensive.fadfsdf_20260126_110701.csv",
      "unit": "GB/s"
    },
    {
      "name": "hardware.stream_copy",
      "value": 435.88,
      "type": "scalar",
      "unit": "GB/s"
    },
}
```

#### 控制台输出

```
===================================================
内存拷贝带宽扫描测试
方向：主机到设备
内存类型：固定
===================================================

大小 (MB)      时间 (ms)  带宽 (GB/s)      CV (%)
------------------------------------------------------
     0.06            0.12             25.60        1.20
     0.12            0.24             25.80        0.90
     0.25            0.48             26.10        0.85
   256.00           98.45             26.20        1.10
   512.00          195.12             26.50        0.95
```

### 推理基准测试输出

#### 指标 JSON

```json
{
  "throughput_tokens_per_sec": 1250.5,
  "latency_ms": 8.5,
  "memory_usage_mb": 2048,
  "model_name": "infinilm-7b",
  "batch_size": 32
}
```

## 结果代码参考

| 代码 | 含义 |
|------|---------|
| 0 | 成功 |
| 非零 | 失败（特定错误） |
