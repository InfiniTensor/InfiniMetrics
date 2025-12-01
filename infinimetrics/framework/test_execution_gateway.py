import sys
import os
import json
import importlib
import inspect
import argparse
from typing import Any, Optional, List

# Ensure infinicore is available (Mock or Real)
import infinicore 

# Path adaptation
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from framework.base import BaseOperatorTest, TestCase, TensorSpec, TestConfig
from framework.runner import GenericTestRunner
from adapters import ExternalSystemAdapter

class TestExecutionGateway:
    """
    Test Execution Gateway
    Orchestrates the test execution flow using an injected adapter.
    """

    def __init__(self, adapter: ExternalSystemAdapter):
        """
        Args:
            adapter: An instance implementing ExternalSystemAdapter interface.
                     This is REQUIRED.
        """
        if adapter is None:
            raise ValueError("TestExecutionGateway requires an adapter instance.")
        self.adapter = adapter

    def run(self, json_file_path: str, device="cuda", config=None) -> str:
        """
        Main entry point.
        
        Flow:
        1. Validate Input
        2. Parse Request (Adapter) -> Logic Object
        3. Dispatch Execution (Operator / Training / etc.) -> Raw Results
        4. Format Response (Adapter) -> JSON String
        
        Returns:
            str: The final formatted JSON response string.
        """
        print(f"🚀 Gateway: Start processing...")

        # 1. Validation
        if not json_file_path:
            raise ValueError("❌ 'json_file_path' is required.")
        
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"❌ JSON file not found: {json_file_path}")

        # 2. Parse Request using Injected Adapter
        print(f"📄 Source: Loading from JSON file: {json_file_path}")
        
        # Expects: Op/Task Name (str), TestCase (Obj), Runtime Args (Namespace)
        target_name, test_case, runtime_args = self.adapter.parseRequest(json_file_path)
        
        # Prioritize external config if provided
        final_config = config if config is not None else runtime_args

        # 3. Dispatch Execution
        # The Gateway delegates the "How to run" logic to a dispatcher
        print(f"⚙️  Dispatching execution for target: '{target_name}'")
        raw_results = self._dispatch_execution(target_name, test_case, final_config)

        # 4. Format Response
        # The Gateway asks the Adapter to convert raw results back to the external format
        print(f"📝 Formatting response...")
        final_json_response = self.adapter.formatResponse(raw_results)
        
        print(f"🏁 Gateway: Process finished.")
        return final_json_response

    def _dispatch_execution(self, target_name: str, test_case: Any, config: argparse.Namespace) -> List[Any]:
        """
        Dispatcher: Determines which execution engine to use based on the target.
        
        Currently supports:
        - Operator Benchmarks (default)
        
        Future expansion:
        - Model Training
        - Inference Service Testing
        """
        # Logic to determine execution type. 
        # For now, we assume standard Operator Benchmark if it's not explicitly something else.
        
        # Example extensibility:
        # if target_name == "Training":
        #     return self._execute_training_job(test_case, config)
        
        return self._execute_operator_benchmark(target_name, test_case, config)

    def _execute_operator_benchmark(self, op_name: str, test_case: TestCase, config: argparse.Namespace) -> List[Any]:
        """
        Internal execution logic specifically for Operator Benchmarks.
        Handles dynamic module loading, class proxying, and the GenericTestRunner.
        """
        cases_to_run = [test_case]
        print(f"  -> Executing Operator Benchmark: '{op_name}'...")
        
        # A. Dynamic import of the operator module
        # Note: Op names are often CamelCase (e.g., "Conv"), but files are snake_case (e.g., "ops/conv.py")
        module_name = op_name.lower()
        module_path = f"ops.{module_name}"
        
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            print(f"❌ Module not found: {module_path}")
            print(f"   (Please ensure 'ops/{module_name}.py' exists)")
            return []

        # B. Find the original OpTest class within the module
        OriginalOpTest = None
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseOperatorTest) and obj is not BaseOperatorTest:
                OriginalOpTest = obj
                break
        
        if not OriginalOpTest:
            print(f"❌ Valid OpTest subclass not found in {module_path}")
            return []

        # C. Dynamic subclass (Proxy Class) to inject test cases
        class ProxyOpTest(OriginalOpTest):
            def __init__(self):
                super().__init__() 

            def get_test_cases(self):
                return cases_to_run
        
        # D. Run tests
        generic_runner = GenericTestRunner(ProxyOpTest, config)
        success, internal_runner = generic_runner.run()

        # E. Retrieve and Debug results
        results = getattr(internal_runner, "test_results", [])
        
        self._debug_print_results(results)
        
        return results

    def _debug_print_results(self, results: List[Any]):
        """Helper to print raw results for debugging"""
        print("\n" + "="*50)
        print(f"📊 Debug: Internal Results (Count: {len(results)})")
        if isinstance(results, list):
            for i, res in enumerate(results):
                print(f"  [Result {i}]: Success={getattr(res, 'success', 'Unknown')}")
        print("="*50 + "\n")


# ==========================================
# Execution Entry Point
# ==========================================

if __name__ == "__main__":
    from adapters import InfiniCoreAdapter 

    # 1. Configure Argument Parser
    parser = argparse.ArgumentParser(description="Test Execution Gateway")
    parser.add_argument("file_path", type=str, help="Path to input JSON request file")
    args = parser.parse_args()
    
    # 2. Create Concrete Adapter
    specific_adapter = InfiniCoreAdapter()

    try:
        # 3. Inject into Gateway
        gateway = TestExecutionGateway(adapter=specific_adapter)

        # 4. Run (Returns Final JSON String)
        final_response_json = gateway.run(json_file_path=args.file_path)

        # 5. Output
        print("\n[Final Output]:")
        print(final_response_json)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Execution Failed: {e}")
        sys.exit(1)
