# File: main.py

import sys
import os
import json
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from infinimetrics.framework.dispatcher import WorkloadDispatcher


def create_mock_payload():
    """
    Creates a mock payload simulating the content of 'add.json'.
    Used as a fallback if no file is provided.
    """
    return {
        "run_id": "test_run_manual_001",
        "time": "2025-12-22 10:00:00",
        "testcase": "infiniCore.operator.Mul",
        "config": {
            "model_source": "Manual_Test",
            "operator": "mul",
            "device": "cuda",
            "measured_iterations": 100,
            "inputs": [
                {
                    "name": "a",
                    "dtype": "float32",
                    "shape": [13, 4, 4],
                    "strides": [20, 4, 1],
                },
                {
                    "name": "b",
                    "dtype": "float32",
                    "shape": [13, 4, 4],
                    "strides": [20, 4, 1],
                },
            ],
        },
        "metrics": [
            {
                "name": "operator.latency",
                "type": "timeseries",
                "raw_data_url": "./latency.csv",
                "unit": "ms",
            }
        ],
    }


def load_payload_from_file(file_path):
    """
    [New] Read external JSON file and parse into a dictionary.
    """
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        sys.exit(1)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            print(f"[Main] Loading payload from: {file_path}")
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"[Error] Failed to parse JSON file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Unexpected error reading file: {e}")
        sys.exit(1)


def main():
    # 1. Parse command line arguments
    parser = argparse.ArgumentParser(description="InfiniCore Test System CLI")
    parser.add_argument(
        "input_file", 
        nargs="?", 
        help="Path to the input JSON file (e.g., add.json). If omitted, uses mock data."
    )
    args = parser.parse_args()

    print("========================================")
    print("   InfiniCore Test System - CLI Entry   ")
    print("========================================")

    # 2. Initialize Dispatcher
    try:
        dispatcher = WorkloadDispatcher()
    except Exception as e:
        print(f"[Fatal Error] Failed to initialize Dispatcher: {e}")
        return

    # 3. Determine Payload Source (File vs Mock)
    if args.input_file:
        payload = load_payload_from_file(args.input_file)
    else:
        print("\n[Info] No input file provided. Using internal Mock Payload for demo.")
        payload = create_mock_payload()

    print(f"[Main] RunID: {payload.get('run_id', 'Unknown')}")

    # 4. Dispatch the Task
    try:
        result = dispatcher.dispatch(payload)

        # 5. Output Result
        print("\n[Main] Execution Finished. Result:")
        print("-" * 40)
        print(json.dumps(result, indent=4, ensure_ascii=False))
        print("-" * 40)

        # 6. Verification
        if result.get("success") == 0:
            print("[Main] ✅ Test Passed Successfully.")
        else:
            print(f"[Main] ❌ Test Failed (Code: {result.get('success')}).")

    except Exception as e:
        print(f"\n[Main] ❌ Execution crashed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
