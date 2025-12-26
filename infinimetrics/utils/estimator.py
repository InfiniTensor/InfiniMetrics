import math
from typing import Dict, Any, Callable, List

# =========================================================================
# Workload / FLOPS Estimator (Strategy Pattern)
# =========================================================================
class WorkloadEstimator:
    """
    Strategy class for FLOPs calculation.
    """
    _calculators: Dict[str, Callable] = {}

    @classmethod
    def register(cls, op_name: str):
        def decorator(func):
            cls._calculators[op_name.lower()] = func
            return func
        return decorator

    @classmethod
    def get_flops(cls, op_type: str, inputs: list, outputs: list, attrs: dict) -> float:
        calculator = cls._calculators.get(op_type.lower())
        if not calculator:
            return 0.0
        try:
            return calculator(inputs, outputs, attrs)
        except Exception as e:
            # Log detailed warning but don't crash
            print(f"[Warn] FLOPS calc error for {op_type}: {e}")
            return 0.0

# --- Formulas ---

@WorkloadEstimator.register("matmul")
def _calc_matmul(inputs, outputs, attrs):
    # Shape assumption: [..., M, K] x [..., K, N]
    shape_a = inputs[0]["shape"]
    shape_b = inputs[1]["shape"]
    M = shape_a[-2]
    K = shape_a[-1]
    N = shape_b[-1]
    # Handle batch dimensions if needed, here simplified
    batch = math.prod(shape_a[:-2]) if len(shape_a) > 2 else 1
    return 2.0 * batch * M * N * K

@WorkloadEstimator.register("conv")
def _calc_conv(inputs, outputs, attrs):
    in_shape = inputs[0]["shape"]
    out_shape = outputs[0]["shape"]
    
    # N, C_out, H_out, W_out
    N, C_out, H_out, W_out = out_shape[0], out_shape[1], out_shape[2], out_shape[3]
    C_in = in_shape[1]
    
    k_shape = attrs.get("kernel_shape", [1, 1])
    group = attrs.get("group", 1)
    
    return 2.0 * N * H_out * W_out * C_out * (C_in / group) * k_shape[0] * k_shape[1]
