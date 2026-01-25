# 硬件基准测试示例

本文档提供在 InfiniMetrics 中运行硬件基准测试的示例。

## 示例 1：综合硬件基准测试

测试所有硬件能力，包括内存带宽、STREAM 基准测试和缓存性能。

### 配置

```json
{
    "run_id": "hw_comprehensive_001",
    "testcase": "hardware.cudaUnified.Comprehensive",
    "config": {
        "device": "nvidia",
        "array_size": 67108864,
        "buffer_size_mb": 256,
        "output_dir": "./output"
    },
    "metrics": [
        {"name": "hardware.mem_bw_h2d"},
        {"name": "hardware.mem_bw_d2h"},
        {"name": "hardware.mem_bw_d2d"},
        {"name": "hardware.stream_copy"},
        {"name": "hardware.stream_scale"},
        {"name": "hardware.stream_add"},
        {"name": "hardware.stream_triad"},
        {"name": "hardware.gpu_cache_l1"},
        {"name": "hardware.gpu_cache_l2"}
    ]
}
```

### 运行

```bash
python main.py format_input_comprehensive_hardware.json
```

### 测试内容
- 多种缓冲区大小的内存带宽 (H2D、D2H、D2D)
- STREAM 基准测试（复制、缩放、加法、三元组操作）
- GPU 缓存性能 (L1、L2)

## 示例 2：仅内存带宽

仅关注内存传输性能。

### 配置

```json
{
    "run_id": "hw_memory_001",
    "testcase": "hardware.cudaUnified.MemoryBandwidth",
    "config": {
        "device": "nvidia",
        "buffer_size_mb": 512,
        "output_dir": "./output"
    },
    "metrics": [
        {"name": "hardware.mem_bw_h2d"},
        {"name": "hardware.mem_bw_d2h"},
        {"name": "hardware.mem_bw_d2d"}
    ]
}
```

### 运行

```bash
python main.py memory_only_config.json
```

## 示例 3：STREAM 基准测试

测试计算模式的可持续内存带宽。

### 配置

```json
{
    "run_id": "hw_stream_001",
    "testcase": "hardware.cudaUnified.STREAM",
    "config": {
        "device": "nvidia",
        "array_size": 67108864,
        "output_dir": "./output"
    },
    "metrics": [
        {"name": "hardware.stream_copy"},
        {"name": "hardware.stream_scale"},
        {"name": "hardware.stream_add"},
        {"name": "hardware.stream_triad"}
    ]
}
```

### 运行

```bash
python main.py stream_config.json
```

## 示例 4：缓存性能

评估 L1 和 L2 缓存带宽。

### 配置

```json
{
    "run_id": "hw_cache_001",
    "testcase": "hardware.cudaUnified.Cache",
    "config": {
        "device": "nvidia",
        "output_dir": "./output"
    },
    "metrics": [
        {"name": "hardware.gpu_cache_l1"},
        {"name": "hardware.gpu_cache_l2"}
    ]
}
```

### 运行

```bash
python main.py cache_config.json
```

## 示例 5：多个配置

批量运行多个硬件测试。

### 目录结构

```
test_configs/
├── memory.json
├── stream.json
└── cache.json
```

### 运行所有

```bash
python main.py ./test_configs/
```

## 理解结果

### 内存带宽

GB/s 为单位的结果：
- H2D/D2H: 20-30 GB/s（受 PCIe 限制）
- D2D: 300-700 GB/s（HBM 带宽）

### STREAM 结果

测量计算的可持续带宽：
- 复制: 2 字节/元素
- 缩放: 2 字节/元素
- 加法: 3 字节/元素
- 三元组: 3 字节/元素

典型值：现代 GPU 为 300-700 GB/s

### 缓存结果

- L1: 1-2 TB/s
- L2: 500-800 GB/s

## 输出文件

结果保存在：
```
./output/
└── hardware.cudaUnified.Comprehensive/
    ├── metrics.json
    ├── trace.json
    └── log.txt
```

## 故障排除

### 构建错误

如果硬件测试编译失败：

```bash
cd infinimetrics/hardware/cuda-memory-benchmark
bash build.sh
```

### 内存不足

减少缓冲区大小：

```json
{
    "config": {
        "buffer_size_mb": 128
    }
}
```

## 后续步骤

- 参阅[硬件测试文档](../test_types/hardware_tests.md)了解详情
- 探索[高级用法](./advanced_usage.md)了解更多示例
