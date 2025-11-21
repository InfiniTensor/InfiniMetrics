#!/usr/bin/env python3

import json
import argparse
import sys
from datetime import datetime

try:
    from . import op_library_mock as op_library
    # import op_library_real as op_library # (Future real call)
except ImportError:
    # Fallback for running script directly without -m
    import op_library_mock as op_library


def run_operator_tests(test_cases: list) -> list:
    """
    Iterates through test cases, calls the operator library, and returns a list containing the results.
    """
    
    results_list = []
    total_cases = len(test_cases)
    
    print("\n" + "="*60)
    print(f"🚀 Starting operator performance test ({total_cases} total)...")
    print("="*60 + "\n")
    
    for i, case in enumerate(test_cases):
        
        fqn_name = case["fully_qualified_name"]
        op_type = case["op_type"]
        input_shape = case["input_shape"]

        print(f"--- Running ({i+1}/{total_cases}): {fqn_name} ---")
        
        try:
            # 1. Call the underlying operator library
            result = op_library.benchmark_operator(
                op_type=op_type,
                fqn_name=fqn_name,
                input_shape=input_shape
                # (In the future we may can pass more parameters, e.g., "precision=...")
            )
            
            # 2. Merge "original config" and "measurement results"
            case_result = case.copy()
            case_result.update(result)
            
            print(f"  > Status: {result['status']}, Latency: {result['actual_latency_ms']} ms\n")
            
        except Exception as e:
            #Ensure robustness: a single operator failure should not terminate the entire test
            print(f"  > Status: FAILED, Error: {e}\n")
            case_result = case.copy()
            case_result.update({
                "actual_latency_ms": None,
                "status": f"FAILED: {e}"
            })
        
        results_list.append(case_result)
        
    print("="*60)
    print("✅ All operator tests completed.")
    print("="*60)
    
    return results_list


def main():
    """
    Main function:
    1. (Load) Read "Test Case" JSON file.
    2. (Execute) Call the operator library to run tests.
    3. (Report) Merge "Config" and "Result", save as new "Test Result" JSON file.
    """
    
    parser = argparse.ArgumentParser(
        description="Reads operator test cases (JSON) and executes benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input-config",
        default="operator_test_cases.json",
        help="Path to the input 'Operator Test Cases' (JSON) file."
    )
    parser.add_argument(
        "-o", "--output-report",
        default=f"operator_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        help="Path to the output 'Operator Test Results' (JSON) file."
    )
    
    args = parser.parse_args()

    # --- 1. (Load) ---
    try:
        print(f"--- 1. Loading Test Configuration ---")
        with open(args.input_config, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        print(f"Successfully loaded {len(test_cases)} test cases (from {args.input_config})")
    
    except FileNotFoundError:
        print(f"Error: Test case file {args.input_config} not found.")
        print("Did you run `parse_summary.py` first to generate it?")
        sys.exit(1)
    except Exception as e:
        print(f"Error occurred during loading: {e}")
        sys.exit(1)

    # --- 2. (Execute) ---
    results_with_data = run_operator_tests(test_cases)
    
    # --- 3. (Report) ---
    try:
        print(f"--- 3. Saving Final Report ---")
        with open(args.output_report, 'w', encoding='utf-8') as f:
            json.dump(results_with_data, f, indent=4)
        print(f"Final assessment report (including latency) saved to: {args.output_report}")
    except Exception as e:
        print(f"Error occurred while saving report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
