# FLOPS Calculator - Adding New Operators

## Overview

`FLOPSCalculator` calculates FLOPS (floating point operations) and bandwidth requirements for different operators using a **registry pattern** for easy extension.

## Supported Operators

### 1. Matrix Operations

#### Matmul (Matrix Multiplication)
- **Formula**: FLOPS = 2 * M * N * K
- **Input**: A [M, K], B [K, N]
- **Output**: C [M, N]
- **Aliases**: matmul, bmm, batchmm

#### AddMM (Matrix Multiply-Add)
- **Formula**: FLOPS = 2 * M * N * K + M * N
- **Input**: bias [M, N], input [M, K], weight [K, N]
- **Output**: C [M, N]
- **Aliases**: addmm, linear

### 2. Convolution

#### Conv2D
- **Formula**: FLOPS = 2 * N * C_out * H_out * W_out * K_h * K_w * C_in
- **Parameters**:
  - N: batch size
  - C_out: output channels
  - H_out, W_out: output spatial dimensions
  - K_h, K_w: kernel size
  - C_in: input channels
- **Note**: Uses output shape to account for stride/padding effects
- **Aliases**: conv2d, conv2d_backward

### 3. Element-wise Operations

#### Add, Mul, Div, Sub
- **Formula**: FLOPS = output_size
- **Description**: One operation per output element

### 4. Activation Functions

#### ReLU, Sigmoid, Tanh
- **Formula**: FLOPS = output_size * multiplier
  - ReLU: multiplier = 1.0
  - Sigmoid, Tanh: multiplier = 2.0

#### GELU, SiLU, Swish
- **Formula**: FLOPS = output_size * 3.0
- **Description**: Complex activation functions requiring more operations

### 5. Normalization

#### Layer Normalization
- **Formula**: FLOPS = input_size * 6.0
- **Description**: mean, variance, normalize, scale, shift

#### RMS Normalization
- **Formula**: FLOPS = input_size * 4.0
- **Description**: Simplified compared to layer norm

#### Batch Normalization
- **Formula**: FLOPS = input_size * 4.0
- **Description**: mean, variance, scale/shift

### 6. Special Operators

#### Softmax
- **Formula**: FLOPS = input_size * 5.0
- **Description**: exp, sum, div operations

#### SwiGLU
- **Formula**: FLOPS = input_size * 4.0
- **Description**: SiLU (3) + multiplication (1)

#### Causal Softmax
- **Formula**: FLOPS = input_size * 2.5
- **Description**: Masked softmax (~half elements)

#### Embedding
- **Formula**: FLOPS = 0
- **Description**: Memory lookup only, no computation

## Adding New Operators

### Method 1: Simple Multiplier (for activation/normalization)

```python
# Add to ACTIVATION_MULTIPLIERS dictionary
ACTIVATION_MULTIPLIERS = {
    "relu": 1.0,
    "sigmoid": 2.0,
    "your_new_activation": 3.0,  # Add here
}

# Or add to NORM_MULTIPLIERS
NORM_MULTIPLIERS = {
    "layernorm": 6.0,
    "your_norm": 5.0,  # Add here
}
```

### Method 2: Registry Pattern (for complex operators)

```python
@FLOPSCalculator.register(["your_operator", "alias1", "alias2"])
def _your_operator_flops(inputs: List[Dict], outputs: List[Dict]) -> float:
    """
    Your operator description.

    Formula: FLOPS = ...
    """
    if not inputs or not outputs:
        return 0.0

    # Your calculation logic here
    input_shape = inputs[0].get("shape", [])
    output_shape = outputs[0].get("shape", [])

    # Calculate and return FLOPS
    return calculated_flops
```

## Examples

### Example 1: Single Operator Registration

```python
@FLOPSCalculator.register(["flash_attention"])
def _flash_attn_flops(inputs: List[Dict], outputs: List[Dict]) -> float:
    """Flash Attention (FLOPS = 9 * N * S * d^2)"""
    if not inputs:
        return 0.0

    batch, seq_len, hidden = inputs[0].get("shape", [1, 1, 1])
    return 9.0 * batch * seq_len * hidden * hidden
```

### Example 2: Multiple Aliases

```python
@FLOPSCalculator.register(["matmul", "bmm", "batchmm", "dot"])
def _matmul_flops(inputs: List[Dict], outputs: List[Dict]) -> float:
    """Matrix Multiplication: C = A @ B (FLOPS = 2 * M * N * K)"""
    # ... implementation ...
```

### Example 3: Using Simple Multiplier

```python
# Add to the dictionary
ACTIVATION_MULTIPLIERS = {
    "relu": 1.0,
    "mish": 4.0,  # Add new activation
}
```

## Testing New Operators

```python
from infinimetrics.utils.flops_calculator import FLOPSCalculator

# Test your new operator
inputs = [{"shape": [batch, channels, height, width], "dtype": "float16"}]
outputs = [{"shape": [batch, out_channels, out_h, out_w], "dtype": "float16"}]

flops = FLOPSCalculator.get_flops("your_operator", inputs, outputs)
print(f"FLOPS: {flops:,}")
```

## Check Registered Operators

```python
# List all registered operators
print(list(FLOPSCalculator._flops_registry.keys()))
# Output: ['matmul', 'bmm', 'batchmm', 'addmm', 'linear', 'conv2d', 'conv2d_backward']
```

## Bandwidth Calculation

The `calculate_bandwidth` function automatically calculates:

1. **Read bytes**: Total bytes of all input tensors
2. **Write bytes**: Total bytes of all output tensors
3. **Total bytes**: Sum of read and write bytes

### dtype Mapping

- float16, bfloat16: 2 bytes
- float32: 4 bytes
- float64: 8 bytes
- int8: 1 byte
- int16: 2 bytes
- int32: 4 bytes
- int64: 8 bytes

## Calculation Examples

### Matmul Example

```python
# Input: A [2, 3], B [3, 4]
# Output: C [2, 4]
# FLOPS = 2 * 2 * 4 * 3 = 48

inputs = [
    {'shape': [2, 3], 'dtype': 'float16'},
    {'shape': [3, 4], 'dtype': 'float16'}
]
outputs = [
    {'shape': [2, 4], 'dtype': 'float16'}
]

flops = FLOPSCalculator.get_flops('matmul', inputs, outputs)
# flops = 48.0

# If latency is 1ms
# TFLOPS = (48 / 0.001) / 1e12 = 0.000000048 TFLOPS
```

## Design Patterns

1. **Registry Pattern**: Dictionary mapping operator names to calculation functions
2. **Decorator Pattern**: `@register` decorator for clean registration
3. **Strategy Pattern**: Different calculation strategies for different operators

## Migration Guide

### Before (if-elif chains)
```python
if op == "matmul":
    return self._matmul_flops(inputs, outputs)
elif op == "conv2d":
    return self._conv2d_flops(inputs, outputs)
elif op == "your_op":
    return self._your_op_flops(inputs, outputs)
# ... grows indefinitely
```

### After (registry pattern)
```python
# Registered once with decorator
@FLOPSCalculator.register(["your_op"])
def _your_op_flops(inputs, outputs):
    # ... implementation

# Lookup is automatic
if op in FLOPSCalculator._flops_registry:
    return FLOPSCalculator._flops_registry[op](inputs, outputs)
```

## Benefits of Registry Pattern

- **No more long if-elif chains**: Operators are registered in a dictionary
- **Easy extension**: Add new operators with decorators
- **Multi-alias support**: One function can handle multiple operator names
- **Clean separation**: Each operator has its own function
- **Better testability**: Each operator can be tested independently

## FAQ

**Q: Why do some operators have FLOPS = 0?**
A: Some operators (like Embedding) are primarily memory accesses with negligible computation.

**Q: How do I handle operators with variable FLOPS?**
A: For operators where FLOPS depend on input dimensions, infer from input shapes.

**Q: Is bandwidth calculation accurate?**
A: This is theoretical bandwidth. Actual bandwidth may vary due to caching, memory alignment, and hardware optimizations.

**Q: Can I register the same function multiple times?**
A: Yes, use the `@register` decorator with a list of operator names to create aliases.

## Notes

1. **dtype affects bandwidth but not FLOPS**: FLOPS is based on operation count, bandwidth considers dtype bytes
2. **Batch dimensions**: Supports arbitrary batch dimensions for matrix multiplication
3. **Error handling**: Returns 0.0 for invalid input formats
4. **Extensibility**: Easy to add new operators without modifying core logic
