# 通信测试

通信测试使用 NCCL（NVIDIA 集合通信库）集合操作对 GPU 间和节点间通信性能进行基准测试。

## NCCL 测试

| 测试名称 | 框架 | 描述 |
|-----------|----------|-------------|
| `comm.nccltest.AllReduce` | NCCL | AllReduce 集合操作 |
| `comm.nccltest.AllGather` | NCCL | AllGather 集合操作 |
| `comm.nccltest.Broadcast` | NCCL | 广播操作 |
| `comm.nccltest.Reduce` | NCCL | 归约操作 |

## 集合操作

### AllReduce
组合来自所有 GPU 的值并将结果分发回所有 GPU。

**用例**: 分布式训练中的梯度平均

### AllGather
从所有 GPU 收集数据并在所有 GPU 上使其可用。

**用例**: 模型并行性中的数据分发

### Broadcast
将数据从一个 GPU（根）复制到所有其他 GPU。

**用例**: 广播参数或输入数据

### Reduce
组合来自所有 GPU 的值并将结果存储在一个 GPU（根）上。

**用例**: 将结果收集到单个进程

## 配置示例

```json
{
    "run_id": "comm_test_001",
    "testcase": "comm.nccltest.AllReduce",
    "config": {
        "num_gpus": 4,
        "min_bytes": 1024,
        "max_bytes": 1073741824,
        "step_factor": 2,
        "output_dir": "./output"
    },
    "metrics": [
        {"name": "comm.bandwidth"},
        {"name": "comm.latency"}
    ]
}
```

## 运行通信测试

### 单节点

```bash
# 单节点上的多 GPU
python main.py nccl_test_config.json
```

### 多节点

```bash
# 需要正确的 NCCL/SHARP 配置
# 在每个节点上运行
mpirun -np 8 -hostfile hosts python main.py nccl_test_config.json
```

## 理解结果

### 带宽
- **单位**: GB/s（千兆字节/秒）
- **描述**: 数据传输速率
- **越高越好**
- **典型值**: 100-300 GB/s (NVLink)、25-50 GB/s (PCIe)

### 延迟
- **单位**: 微秒 (µs)
- **描述**: 完成操作的时间
- **越低越好**
- **典型值**: 5-20 µs (NVLink)、10-50 µs (PCIe)

## 性能因素

### 网络拓扑
- **NVLink**: 最高带宽，最低延迟
- **PCIe**: 中等带宽，较高延迟
- **InfiniBand/以太网**: 多节点通信

### 消息大小
- 小消息：延迟受限
- 大消息：带宽受限

### GPU 数量
- 更多 GPU：更高的聚合带宽
- 可能增加每个操作的延迟

## NCCL 环境变量

用于调优的常见环境变量：

```bash
# NCCL 调试
export NCCL_DEBUG=INFO

# 网络接口
export NCCL_SOCKET_IFNAME=ib0

# 禁用 SHARP（对于某些 InfiniBand 设置）
export NCCL_SHARP_DISABLE=1

# 设置线程数
export NCCL_NTHREADS=4
```

## 故障排除

### 通信挂起

1. 检查防火墙设置
2. 验证网络连接
3. 确保正确的 NCCL 安装
4. 检查进程同步问题

### 性能不佳

1. 验证 NVLink/PCIe 拓扑: `nvidia-smi topo -m`
2. 检查网络带宽: `ibstat` (InfiniBand)
3. 尝试不同的 NCCL 算法
4. 减少进程数

### 多节点问题

1. 验证节点间的 SSH 信任
2. 检查防火墙规则
3. 确保一致的 NCCL 版本
4. 验证网络配置

## 示例

更多示例，请参阅[配置指南](./configuration.md)。

## NCCL 测试子模块

项目使用 NCCL 测试作为 git 子模块。确保已初始化：

```bash
git submodule update --init --recursive
```

更多 NCCL 测试信息，请参阅 [NCCL-tests 仓库](https://github.com/NVIDIA/nccl-tests)。
