#!/usr/bin/env python3
import time
import random

def benchmark_operator(op_type: str, fqn_name: str, input_shape: list):
    """
    This is a "mock" operator library interface.
    It pretends to run a test and returns a simulated latency.
    
    (In the future, we can replace this with a real call like:
     `actual_latency = infiniCore.benchmark(...)`)
    """
    
    # 1. Simulate work (e.g., 0.01 to 0.1 seconds)
    mock_duration_sec = random.uniform(0.01, 0.1)
    time.sleep(mock_duration_sec)
    
    # 2. Convert seconds to milliseconds and round to 4 decimal places
    mock_latency_ms = round(mock_duration_sec * 1000, 4)
    
    # 3. Return measurement results
    return {
        "actual_latency_ms": mock_latency_ms,
        "status": "SUCCESS"
    }
