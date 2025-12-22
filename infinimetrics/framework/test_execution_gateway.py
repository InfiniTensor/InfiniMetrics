import sys
import os
import json
import argparse
from typing import Any, Dict, List

# Ensure infinicore is available
import infinicore 

# Path adaptation
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- Key Modification: Import TestCaseManager ---
# Assuming your TestCaseManager is defined in framework.manager or framework.runner
# If your manager is elsewhere, please modify this import
from infinicore.test.framework import TestCaseManager
from adapters import ExternalSystemAdapter

class TestExecutionGateway:
    """
    Test Execution Gateway (File-Based Architecture)
    
    Flow:
    1. External JSON -> Adapter -> Internal JSON File (Temp)
    2. Internal JSON File -> TestCaseManager -> Execution Results (Dict)
    3. Execution Results -> Adapter -> Final JSON Response
    """

    def __init__(self, adapter: ExternalSystemAdapter):
        if adapter is None:
            raise ValueError("TestExecutionGateway requires an adapter instance.")
        self.adapter = adapter

    def run(self, json_file_path: str, device="cuda", config=None) -> str:
        """
        Main entry point.
        """
        print(f"🚀 Gateway: Start processing...")

        # 1. Validation
        if not json_file_path:
            raise ValueError("❌ 'json_file_path' is required.")
        
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"❌ JSON file not found: {json_file_path}")

        # 2. Parse Request using Adapter
        # [Modification 1] parseRequest now only returns the path to the generated internal config file
        print(f"📄 Adapter: Translating request from {json_file_path}...")
        temp_config_path = self.adapter.parseRequest(json_file_path)
        
        # Prepare config overrides
        final_config = config if config is not None else {}
        # If device argument is passed, add to override
        if device:
            final_config["device"] = device

        try:
            # 3. Dispatch Execution
            print(f"⚙️  Dispatching execution with internal config: {temp_config_path}")
            run_output_full = self._dispatch_execution(temp_config_path, final_config)

            # =========================================================
            # 👇👇👇 [DEBUG] Print content of run_output_full 👇👇👇
            # =========================================================
            print("\n" + "="*30 + " DEBUG: run_output_full " + "="*30)
            try:
                # default=str prevents errors due to non-serializable objects (like class instances)
                print(json.dumps(run_output_full, indent=4, default=str))
            except Exception as e:
                # If json dump fails, fallback to raw print
                print(f"⚠️ JSON dump failed ({e}), printing raw object:")
                print(run_output_full)
            print("="*80 + "\n")
            # =========================================================

            # 4. Format Response
            # [Modification 2] Pass the full execution result dictionary back to the adapter
            print(f"📝 Formatting response...")
            final_json_response = self.adapter.formatResponse(run_output_full)
            
            print(f"🏁 Gateway: Process finished.")
            return final_json_response

        finally:
            # [Optional] Clean up temporary file
            if os.path.exists(temp_config_path):
                # os.remove(temp_config_path) 
                print(f"🧹 [Debug] Temp file preserved at: {temp_config_path}")

    def _dispatch_execution(self, config_file_path: str, overrides: Dict) -> Dict[str, Any]:
        """
        [Modification 3] The dispatcher is now greatly simplified.
        It no longer needs to import modules or construct classes itself, but delegates directly to TestCaseManager.
        """
        manager = TestCaseManager()
        
        print(f"   -> Manager running file...")
        
        # Manager.run usually returns a List[SuiteResult]
        # Since we map one file to one request, we take the result of the first Suite
        results = manager.run(
            json_file_path=config_file_path, 
            config=overrides, 
            save_path=None # Gateway mode doesn't need Manager to save another file
        )
        
        if not results or not isinstance(results, list):
            return {"error": "Invalid results from manager", "execution_results": []}

        # Return the full dictionary of the first Suite (containing operator, args, testcases, execution_results, etc.)
        return results[0]


# ==========================================
# Execution Entry Point
# ==========================================

if __name__ == "__main__":
    from adapters import InfiniCoreAdapter 

    # 1. Configure Argument Parser
    parser = argparse.ArgumentParser(description="Test Execution Gateway")
    parser.add_argument("file_path", type=str, help="Path to input JSON request file")
    # Allow CLI overrides
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--bench", type=str, default=None)
    
    args = parser.parse_args()
    
    # Construct config dictionary
    cli_config = {}
    if args.bench: cli_config["bench"] = args.bench
    # device is passed as a separate argument

    # 2. Create Concrete Adapter
    specific_adapter = InfiniCoreAdapter()

    try:
        # 3. Inject into Gateway
        gateway = TestExecutionGateway(adapter=specific_adapter)

        # 4. Run
        final_response_json = gateway.run(
            json_file_path=args.file_path, 
            device=args.device,
            config=cli_config
        )

        # 5. Output
        print("\n[Final Output]:")
        print(final_response_json)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Execution Failed: {e}")
        sys.exit(1)
