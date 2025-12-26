import sys
import os
import json
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from infinimetrics.framework.dispatcher import WorkloadDispatcher


def create_mock_payload():
    """Fallback mock payload."""
    return {
        "run_id": "mock_run_001",
        "testcase": "infiniCore.operator.Mul",
        "config": {"operator": "mul", "device": "cuda", "inputs": []}, # Simplified demo
        "metrics": []
    }

def load_payload_from_file(file_path):
    """Read and parse JSON file."""
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[Error] Failed to read {file_path}: {e}")
        return None

def process_single_task(dispatcher, payload, source_name, output_dir="results"):
    """
    Core logic for processing a single task.
    1. Dispatch execution.
    2. Save result to JSON file.
    3. Print summary.
    
    Returns: bool (True if success, False otherwise)
    """
    run_id = payload.get('run_id', 'Unknown')
    print(f"\n>>> Processing Task: {source_name} (RunID: {run_id})")
    
    try:
        # 1. Execute task
        result = dispatcher.dispatch(payload)
        
        # 2. Save Result to File
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Generate filename: use filename for file input, run_id for mock input
        if os.path.isfile(source_name):
            base_name = os.path.basename(source_name).replace(".json", "")
            save_name = f"{base_name}_result.json"
        else:
            save_name = f"{run_id}_result.json"
            
        save_path = os.path.join(output_dir, save_name)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            # indent=4 for pretty printing, ensure_ascii=False ensures non-ASCII characters display correctly
            json.dump(result, f, indent=4, ensure_ascii=False)
            
        print(f"    💾 Saved result to: {save_path}")
        
        # 3. Result summary check
        success_code = result.get("success")
        if success_code == 0:
            print(f"    ✅ Passed.")
            return True
        else:
            print(f"    ❌ Failed (Error Code: {success_code}).")
            if "error_msg" in result:
                print(f"       Reason: {result['error_msg']}")
            return False

    except Exception as e:
        print(f"    ❌ Crashed: {e}")
        # If crashed, print detailed traceback for debugging
        import traceback
        traceback.print_exc()
        return False
        
def main():
    # 1. Define command line arguments
    parser = argparse.ArgumentParser(description="InfiniMetrics Batch Runner")
    parser.add_argument(
        "input_files", 
        nargs="*",  # accepts 0 or more files
        help="List of JSON files to execute sequentially."
    )
    args = parser.parse_args()

    print("========================================")
    print("      InfiniMetrics Batch Runner        ")
    print("========================================")

    # 2. Initialize Dispatcher (Initialize once, reuse)
    try:
        dispatcher = WorkloadDispatcher()
    except Exception as e:
        print(f"[Fatal Error] Dispatcher Init Failed: {e}")
        return

    # 3. Prepare task list
    tasks = []
    if args.input_files:
        # If files are provided, process them
        for fpath in args.input_files:
            data = load_payload_from_file(fpath)
            if data:
                tasks.append((fpath, data))
    else:
        # No inputs provided, run Mock
        print("[Info] No inputs provided. Using Mock data.")
        tasks.append(("Mock Data", create_mock_payload()))

    # 4. Batch Execution
    stats = {"total": len(tasks), "passed": 0, "failed": 0}
    
    for source_name, payload in tasks:
        is_success = process_single_task(dispatcher, payload, source_name)
        if is_success:
            stats["passed"] += 1
        else:
            stats["failed"] += 1

    # 5. Final Report
    print("\n" + "="*40)
    print(f" 📊 Execution Summary")
    print(f"    Total:  {stats['total']}")
    print(f"    Passed: {stats['passed']}")
    print(f"    Failed: {stats['failed']}")
    print("="*40)
    
    # Set exit code to non-zero if failures occurred (useful for CI/CD)
    if stats["failed"] > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
