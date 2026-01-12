#!/usr/bin/env python3
"""
Hardware Specifications for Performance Efficiency Calculation

This module contains theoretical peak performance specs for various accelerators.
Used to calculate computational efficiency (actual TFLOPS / peak TFLOPS) and
memory efficiency (actual bandwidth / peak bandwidth).

Data sources:
- NVIDIA: https://www.nvidia.com/en-us/data-center/products/
- AMD: https://www.amd.com/en/products/accelerators
- Intel: https://www.intel.com/content/www/us/en/products/details/discrete-gpus.html
"""

from typing import Dict, Optional


# GPU Specifications (TFLOPS and GB/s)
# TFLOPS values are for FP16/BF16 (unless specified otherwise)
# Bandwidth values are in GB/s
GPU_SPECS = {
    # NVIDIA GPUs
    "h100": {
        "name": "NVIDIA H100 SXM5",
        "fp16_tflops": 989.5,  # Hopper with FP16
        "fp32_tflops": 67.0,   # TF32
        "int8_tflops": 1979.0,  # INT8
        "bandwidth_gbs": 3350,  # HBM3
        "architecture": "Hopper",
        "memory": "80 GB HBM3",
    },
    "h100-pcie": {
        "name": "NVIDIA H100 PCIe",
        "fp16_tflops": 378.6,  # FP16
        "fp32_tflops": 49.4,   # TF32
        "int8_tflops": 757.2,  # INT8
        "bandwidth_gbs": 2000,  # HBM3
        "architecture": "Hopper",
        "memory": "80 GB HBM3",
    },
    "a100-sxm": {
        "name": "NVIDIA A100 SXM",
        "fp16_tflops": 312.0,  # FP16
        "fp32_tflops": 19.5,   # TF32
        "int8_tflops": 624.0,  # INT8
        "bandwidth_gbs": 2039,  # HBM2e
        "architecture": "Ampere",
        "memory": "80 GB HBM2e",
    },
    "a100-pcie": {
        "name": "NVIDIA A100 PCIe",
        "fp16_tflops": 312.0,  # FP16
        "fp32_tflops": 19.5,   # TF32
        "int8_tflops": 624.0,  # INT8
        "bandwidth_gbs": 1935,  # HBM2e
        "architecture": "Ampere",
        "memory": "80 GB HBM2e",
    },
    "a100-40gb": {
        "name": "NVIDIA A100 40GB",
        "fp16_tflops": 312.0,
        "fp32_tflops": 19.5,
        "int8_tflops": 624.0,
        "bandwidth_gbs": 1555,  # HBM2e
        "architecture": "Ampere",
        "memory": "40 GB HBM2e",
    },
    "a800": {
        "name": "NVIDIA A800 SXM (China)",
        "fp16_tflops": 312.0,
        "fp32_tflops": 19.5,
        "int8_tflops": 624.0,
        "bandwidth_gbs": 1800,  # HBM2e (reduced from A100)
        "architecture": "Ampere",
        "memory": "80 GB HBM2e",
    },
    "h800": {
        "name": "NVIDIA H800 SXM (China)",
        "fp16_tflops": 989.5,
        "fp32_tflops": 67.0,
        "int8_tflops": 1979.0,
        "bandwidth_gbs": 2800,  # HBM3 (reduced from H100)
        "architecture": "Hopper",
        "memory": "80 GB HBM3",
    },
    "l40s": {
        "name": "NVIDIA L40S",
        "fp16_tflops": 183.0,
        "fp32_tflops": 91.5,
        "int8_tflops": 366.0,
        "bandwidth_gbs": 864,  # GDDR6
        "architecture": "Ada",
        "memory": "48 GB GDDR6",
    },
    "l4": {
        "name": "NVIDIA L4",
        "fp16_tflops": 59.5,
        "fp32_tflops": 30.3,
        "int8_tflops": 119.0,
        "bandwidth_gbs": 300,  # GDDR6
        "architecture": "Ada",
        "memory": "24 GB GDDR6",
    },
    "rtx4090": {
        "name": "NVIDIA RTX 4090",
        "fp16_tflops": 82.6,   # FP16 (estimated)
        "fp32_tflops": 82.6,   # FP32
        "int8_tflops": 330.5,  # Tensor cores (estimated)
        "bandwidth_gbs": 1008,  # GDDR6X
        "architecture": "Ada",
        "memory": "24 GB GDDR6X",
    },
    "rtx6000ada": {
        "name": "NVIDIA RTX 6000 Ada",
        "fp16_tflops": 91.1,
        "fp32_tflops": 91.1,
        "int8_tflops": 364.4,
        "bandwidth_gbs": 960,  # GDDR6
        "architecture": "Ada",
        "memory": "48 GB GDDR6",
    },
    "v100": {
        "name": "NVIDIA V100 SXM",
        "fp16_tflops": 125.0,
        "fp32_tflops": 15.7,   # FP32
        "int8_tflops": 125.0,  # INT8 (not natively supported, estimated)
        "bandwidth_gbs": 900,  # HBM2
        "architecture": "Volta",
        "memory": "32 GB HBM2",
    },
    "t4": {
        "name": "NVIDIA T4",
        "fp16_tflops": 38.9,
        "fp32_tflops": 8.1,
        "int8_tflops": 130.5,
        "bandwidth_gbs": 320,  # GDDR6
        "architecture": "Turing",
        "memory": "16 GB GDDR6",
    },

    # AMD GPUs
    "mi300x": {
        "name": "AMD MI300X",
        "fp16_tflops": 528.0,  # FP16 (estimated)
        "fp32_tflops": 264.0,  # FP32
        "int8_tflops": 1056.0,  # INT8 (estimated)
        "bandwidth_gbs": 5300,  # HBM3
        "architecture": "CDNA 3",
        "memory": "192 GB HBM3",
    },
    "mi250x": {
        "name": "AMD MI250X",
        "fp16_tflops": 181.9,
        "fp32_tflops": 47.6,
        "int8_tflops": 363.8,
        "bandwidth_gbs": 3276,  # HBM2e
        "architecture": "CDNA 2",
        "memory": "128 GB HBM2e",
    },
    "mi210": {
        "name": "AMD MI210",
        "fp16_tflops": 181.9,
        "fp32_tflops": 22.7,
        "int8_tflops": 363.8,
        "bandwidth_gbs": 1638,  # HBM2e
        "architecture": "CDNA 2",
        "memory": "64 GB HBM2e",
    },

    # Intel GPUs
    "gpuserver": {
        "name": "Intel Data Center GPU Max",
        "fp16_tflops": 148.0,  # FP16 (estimated)
        "fp32_tflops": 74.0,   # FP32
        "int8_tflops": 592.0,  # INT8 (estimated)
        "bandwidth_gbs": 1638,  # HBM2e
        "architecture": "Xe HPC",
        "memory": "128 GB HBM2e",
    },

    # Chinese Accelerators
    "ascend910b": {
        "name": "Huawei Ascend 910B",
        "fp16_tflops": 376.0,  # FP16 (estimated, similar to A100)
        "fp32_tflops": 75.2,   # FP32
        "int8_tflops": 752.0,  # INT8 (estimated)
        "bandwidth_gbs": 1200,  # HBM2e (estimated)
        "architecture": "Da Vinci",
        "memory": "64 GB HBM2e",
    },
    "ascend910": {
        "name": "Huawei Ascend 910",
        "fp16_tflops": 256.0,  # FP16 (estimated)
        "fp32_tflops": 64.0,   # FP32
        "int8_tflops": 512.0,  # INT8 (estimated)
        "bandwidth_gbs": 900,  # HBM2
        "architecture": "Da Vinci",
        "memory": "32 GB HBM2",
    },
}


# CPU specifications (for comparison)
CPU_SPECS = {
    "intel-xeon-platinum": {
        "name": "Intel Xeon Platinum 8480+",
        "fp32_tflops": 4.6,    # AVX-512 FP32
        "bandwidth_gbs": 120,  # DDR5 (8 channels)
        "cores": 56,
    },
    "amd-epyc": {
        "name": "AMD EPYC 9654",
        "fp32_tflops": 3.8,    # AVX-512 FP32
        "bandwidth_gbs": 153,  # DDR5 (12 channels)
        "cores": 96,
    },
}


class HardwareSpecs:
    """Hardware specifications for efficiency calculations"""

    @staticmethod
    def get_gpu_specs(gpu_name: str) -> Optional[Dict]:
        """
        Get GPU specifications by name.

        Args:
            gpu_name: GPU identifier (e.g., 'h100', 'a100-sxm', 'rtx4090')

        Returns:
            Dict with specs or None if not found
        """
        return GPU_SPECS.get(gpu_name.lower())

    @staticmethod
    def get_cpu_specs(cpu_name: str) -> Optional[Dict]:
        """
        Get CPU specifications by name.

        Args:
            cpu_name: CPU identifier (e.g., 'intel-xeon-platinum')

        Returns:
            Dict with specs or None if not found
        """
        return CPU_SPECS.get(cpu_name.lower())

    @staticmethod
    def list_all_gpus() -> list:
        """List all available GPU names"""
        return list(GPU_SPECS.keys())

    @staticmethod
    def list_all_cpus() -> list:
        """List all available CPU names"""
        return list(CPU_SPECS.keys())


def calculate_efficiency(
    actual_tflops: float,
    actual_bandwidth_gbs: float,
    hardware_name: str,
    dtype: str = "fp16",
) -> Dict[str, float]:
    """
    Calculate computational and memory efficiency.

    Args:
        actual_tflops: Measured TFLOPS
        actual_bandwidth_gbs: Measured bandwidth in GB/s
        hardware_name: Hardware identifier (e.g., 'h100', 'a100-sxm')
        dtype: Data type for calculation ('fp16', 'fp32', 'int8')

    Returns:
        Dict with:
            - compute_efficiency_pct: (actual_tflops / peak_tflops) * 100
            - memory_efficiency_pct: (actual_bandwidth / peak_bandwidth) * 100
            - peak_tflops: Theoretical peak TFLOPS
            - peak_bandwidth_gbs: Theoretical peak bandwidth GB/s
    """
    specs = HardwareSpecs.get_gpu_specs(hardware_name)
    if not specs:
        return {
            "compute_efficiency_pct": 0.0,
            "memory_efficiency_pct": 0.0,
            "peak_tflops": 0.0,
            "peak_bandwidth_gbs": 0.0,
            "error": f"Hardware '{hardware_name}' not found",
        }

    # Get peak TFLOPS based on dtype
    peak_tflops = specs.get(f"{dtype}_tflops", specs.get("fp16_tflops", 0.0))
    peak_bandwidth = specs.get("bandwidth_gbs", 0.0)

    compute_efficiency = (actual_tflops / peak_tflops * 100) if peak_tflops > 0 else 0.0
    memory_efficiency = (actual_bandwidth_gbs / peak_bandwidth * 100) if peak_bandwidth > 0 else 0.0

    return {
        "compute_efficiency_pct": round(compute_efficiency, 2),
        "memory_efficiency_pct": round(memory_efficiency, 2),
        "peak_tflops": peak_tflops,
        "peak_bandwidth_gbs": peak_bandwidth,
    }
