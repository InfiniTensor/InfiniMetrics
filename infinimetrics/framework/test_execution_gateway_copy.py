import sys
import os
import json
import importlib
import inspect
from typing import Any, Optional

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

    def run(self, json_file_path: str, device="cuda", config=None) -> Any:
        """
        Main entry point.
        
        Args:
            json_file_path: Path to the JSON request file (Required).
            device: Target device (e.g., 'cuda', 'cpu').
            config: Optional TestConfig override.
        """
        print(f"🚀 Gateway: Start processing...")

        # 1. Validation
        if not json_file_path:
            raise ValueError("❌ 'json_file_path' is required.")
        
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"❌ JSON file not found: {json_file_path}")

        # 2. Parse Request using Injected Adapter
        print(f"📄 Source: Loading from JSON file: {json_file_path}")
        print(f"    Using Adapter: {type(self.adapter).__name__}")
        
        # Polymorphic call: Gateway doesn't know which specific adapter is used
        parsed_op_name, parsed_case, parsed_config = self.adapter.parseRequest(json_file_path)
        
        cases_to_run = [parsed_case]
        
        # Prioritize external config if provided, otherwise use parsed config
        override_config = config if config is not None else parsed_config

        # 3. Execute Tests
        if not parsed_op_name or not cases_to_run:
            raise RuntimeError("❌ Adapter failed to produce valid test cases or operator name.")

        results = self._executeTests(parsed_op_name, cases_to_run, device, override_config)
        
        print(f"🏁 Gateway: Process finished.")
        return results

    def _executeTests(self, op_name, cases_to_run, device, config):
        """
        Private method: Dynamically load operator and execute test logic
        """
        print(f"  -> Executing {len(cases_to_run)} tests for operator: '{op_name}'...")
        
        # 1. Dynamic import of the operator module
        module_path = f"ops.{op_name}"
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            print(f"❌ Module not found: {module_path}")
            print(f"   (Please ensure 'ops/{op_name}.py' exists)")
            return None

        # 2. Find the original OpTest class
        OriginalOpTest = None
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseOperatorTest) and obj is not BaseOperatorTest:
                OriginalOpTest = obj
                break
        
        if not OriginalOpTest:
            print("❌ OpTest class not found in module")
            return None

        # 3. Dynamic subclass (Proxy Class) to inject test cases
        class ProxyOpTest(OriginalOpTest):
            def __init__(self):
                super().__init__() 

            def get_test_cases(self):
                return cases_to_run
        
        # 4. Run tests
        generic_runner = GenericTestRunner(ProxyOpTest, config)
        
        success, internal_runner = generic_runner.run()
                
        # Return results (Assuming runner stores them)
        return getattr(internal_runner, "test_results", "Test Finished (No result object returned)")


# ==========================================
# Execution Entry Point
# ==========================================

if __name__ == "__main__":
    import argparse
    from adapters import InfiniCoreAdapter 

    parser = argparse.ArgumentParser(description="Test Execution Gateway")
    parser.add_argument("file_path", type=str, help="Path to input JSON request file") # Made required
    args = parser.parse_args()
    
    # 1. Create Concrete Adapter
    specific_adapter = InfiniCoreAdapter()

    try:
        # 2. Inject into Gateway
        gateway = TestExecutionGateway(adapter=specific_adapter)

        # 3. Run
        final_response = gateway.run(json_file_path=args.file_path, device="cuda")

        print("\n[Final Execution Result]:")
        print(final_response)

    except Exception as e:
        print(f"\n❌ Execution Failed: {e}")
        sys.exit(1)
