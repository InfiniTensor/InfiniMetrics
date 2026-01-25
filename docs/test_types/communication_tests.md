# Communication Tests

Communication tests benchmark inter-GPU and inter-node communication performance using NCCL (NVIDIA Collective Communications Library) collective operations.

## NCCL Tests

| Test Name | Framework | Description |
|-----------|----------|-------------|
| `comm.nccltest.AllReduce` | NCCL | AllReduce collective operation |
| `comm.nccltest.AllGather` | NCCL | AllGather collective operation |
| `comm.nccltest.Broadcast` | NCCL | Broadcast operation |
| `comm.nccltest.Reduce` | NCCL | Reduce operation |

## Collective Operations

### AllReduce
Combines values from all GPUs and distributes the result back to all GPUs.

**Use case**: Gradient averaging in distributed training

### AllGather
Gathers data from all GPUs and makes it available on all GPUs.

**Use case**: Data distribution in model parallelism

### Broadcast
Copies data from one GPU (root) to all other GPUs.

**Use case**: Broadcasting parameters or input data

### Reduce
Combines values from all GPUs and stores the result on one GPU (root).

**Use case**: Collecting results to a single process

## Configuration Example

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

## Running Communication Tests

### Single Node

```bash
# Multi-GPU on single node
python main.py nccl_test_config.json
```

### Multi-Node

```bash
# Requires proper NCCL/SHARP configuration
# Run on each node
mpirun -np 8 -hostfile hosts python main.py nccl_test_config.json
```

## Understanding Results

### Bandwidth
- **Unit**: GB/s (gigabytes per second)
- **Description**: Data transfer rate
- **Higher is better**
- **Typical values**: 100-300 GB/s (NVLink), 25-50 GB/s (PCIe)

### Latency
- **Unit**: microseconds (µs)
- **Description**: Time to complete operation
- **Lower is better**
- **Typical values**: 5-20 µs (NVLink), 10-50 µs (PCIe)

## Performance Factors

### Network Topology
- **NVLink**: Highest bandwidth, lowest latency
- **PCIe**: Moderate bandwidth, higher latency
- **InfiniBand/Ethernet**: Multi-node communication

### Message Size
- Small messages: Latency-bound
- Large messages: Bandwidth-bound

### Number of GPUs
- More GPUs: Higher aggregate bandwidth
- May increase per-operation latency

## NCCL Environment Variables

Common environment variables for tuning:

```bash
# NCCL debugging
export NCCL_DEBUG=INFO

# Network interface
export NCCL_SOCKET_IFNAME=ib0

# Disable SHARP (for some InfiniBand setups)
export NCCL_SHARP_DISABLE=1

# Set number of threads
export NCCL_NTHREADS=4
```

## Troubleshooting

### Communication Hangs

1. Check firewall settings
2. Verify network connectivity
3. Ensure proper NCCL installation
4. Check for process synchronization issues

### Poor Performance

1. Verify NVLink/PCIe topology: `nvidia-smi topo -m`
2. Check network bandwidth: `ibstat` (InfiniBand)
3. Try different NCCL algorithms
4. Reduce number of processes

### Multi-Node Issues

1. Verify SSH trust between nodes
2. Check firewall rules
3. Ensure consistent NCCL versions
4. Verify network configuration

## Examples

See [Configuration Guide](../configuration.md) for more examples.

## NCCL Tests Submodule

The project uses NCCL tests as a git submodule. Ensure it's initialized:

```bash
git submodule update --init --recursive
```

For more NCCL test information, see the [NCCL-tests repository](https://github.com/NVIDIA/nccl-tests).
