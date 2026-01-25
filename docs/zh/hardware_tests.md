# 硬件测试

硬件测试评估加速器硬件的物理能力，包括内存带宽、缓存性能和计算吞吐量。

## CUDA 内存基准测试

### 主机到设备 (H2D) 测试

| 测试名称 | 描述 | 指标 |
|-----------|-------------|---------|
| `hardware.mem_sweep_h2d` | 主机到设备扫描 (64KB-1GB) | 带宽 (GB/s), 时间 (ms) |
| `hardware.mem_bw_h2d` | H2D 带宽（固定大小） | 带宽 (GB/s) |

### 设备到主机 (D2H) 测试

| 测试名称 | 描述 | 指标 |
|-----------|-------------|---------|
| `hardware.mem_sweep_d2h` | 设备到主机扫描 | 带宽 (GB/s), 时间 (ms) |
| `hardware.mem_bw_d2h` | D2H 带宽（固定大小） | 带宽 (GB/s) |

### 设备到设备 (D2D) 测试

| 测试名称 | 描述 | 指标 |
|-----------|-------------|---------|
| `hardware.mem_sweep_d2d` | 设备到设备扫描 | 带宽 (GB/s), 时间 (ms) |
| `hardware.mem_bw_d2d` | D2D 带宽（固定大小） | 带宽 (GB/s) |

## STREAM 基准测试

STREAM 基准测试测量简单向量内核的可持续内存带宽和简单计算速率。

| 测试名称 | 描述 | 字节数/元素 |
|-----------|-------------|---------------|
| `hardware.stream_copy` | 复制操作 | 2 |
| `hardware.stream_scale` | 缩放操作 | 2 |
| `hardware.stream_add` | 加法操作 | 3 |
| `hardware.stream_triad` | 三元组操作 | 3 |

### STREAM 操作

- **复制**: `a[i] = b[i]`
- **缩放**: `a[i] = q * b[i]`
- **加法**: `a[i] = b[i] + c[i]`
- **三元组**: `a[i] = b[i] + q * c[i]`

## GPU 缓存测试

| 测试名称 | 描述 |
|-----------|-------------|
| `hardware.gpu_cache_l1` | L1 缓存带宽 |
| `hardware.gpu_cache_l2` | L2 缓存带宽 |

## 运行硬件测试

### 配置示例

```json
{
    "run_id": "hw_test_001",
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
        {"name": "hardware.stream_triad"}
    ]
}
```

### 命令

```bash
python main.py format_input_comprehensive_hardware.json
```

## 理解结果

### 内存带宽

结果以 GB/s（千兆字节/秒）为单位报告。数值越高表示性能越好。

典型值：
- H2D: 20-30 GB/s（受 PCIe 3.0/4.0 限制）
- D2H: 20-30 GB/s（受 PCIe 3.0/4.0 限制）
- D2D: 300-700 GB/s（HBM2/HBM2e 带宽）

### STREAM 结果

STREAM 指标测量计算模式的可持续内存带宽。

典型值：现代 GPU 为 300-700 GB/s

### 缓存结果

缓存测试显示 L1 和 L2 缓存的带宽。

典型值：
- L1: 1-2 TB/s
- L2: 500-800 GB/s

## 构建硬件测试

如果需要构建 CUDA 内存基准测试套件：

```bash
cd infinimetrics/hardware/cuda-memory-benchmark
bash build.sh
```

更多详情，请参阅[安装指南](./installation.md)。

## 故障排除

有关常见硬件测试问题，请参阅[故障排除](./troubleshooting.md)。
