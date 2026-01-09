#!/usr/bin/env python3
"""Test InfiniCore Adapter Integration"""

import json
from infinimetrics.dispatcher import Dispatcher

# Test input with operator configuration
test_input = {
    "run_id": "test.conv2d.001",
    "testcase": "operator.InfiniCore.Conv2D",
    "config": {
        "operator": "conv2d",
        "device": "nvidia",
        "inputs": [
            {
                "name": "input",
                "shape": [1, 64, 224, 224],
                "dtype": "float32"
            },
            {
                "name": "weight",
                "shape": [128, 64, 3, 3],
                "dtype": "float32"
            }
        ],
        "attributes": [
            {"name": "stride", "value": [1, 1]},
            {"name": "padding", "value": [1, 1]},
            {"name": "kernel_size", "value": [3, 3]}
        ],
        "outputs": [
            {
                "name": "output",
                "shape": [1, 128, 224, 224],
                "dtype": "float32"
            }
        ],
        "warmup_iterations": 5,
        "measured_iterations": 100
    },
    "metrics": [
        {"name": "operator.latency"},
        {"name": "operator.tensor_accuracy"},
        {"name": "operator.flops"},
        {"name": "operator.bandwidth"}
    ]
}

if __name__ == "__main__":
    print("=" * 60)
    print("Testing InfiniCore Adapter")
    print("=" * 60)

    # Run test
    dispatcher = Dispatcher()
    result = dispatcher.dispatch(test_input)

    # Print result
    print("\nResult:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
