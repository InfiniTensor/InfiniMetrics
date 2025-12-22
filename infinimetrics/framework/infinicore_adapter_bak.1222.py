import abc
import argparse
import json
import os
import sys
import time
from typing import Dict, Any, Tuple, Optional, List

from devices import DeviceBackend

# --- Third-party / Internal Dependencies ---
import infinicore 

# --- Framework Dependencies ---
# Assuming these exist, though parseRequest uses them less now
from framework import TensorSpec 

# =========================================================================
# Base Class: ExternalSystemAdapter
# =========================================================================
class ExternalSystemAdapter(abc.ABC):
    """
    Abstract Adapter Class
    """

    @abc.abstractmethod
    def parseRequest(self, json_file_path: str) -> str:
        """
        Parses external request and returns path to converted internal config file.
        """
        pass

    @abc.abstractmethod
    def formatResponse(self, test_results: Dict[str, Any]) -> str:
        """
        Formats execution results into external response JSON string.
        """
        pass

    # -------------------------------------------------------------------------
    # Shared Utility Methods
    # -------------------------------------------------------------------------

    def _load_and_validate_json(self, file_path: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON file not found at: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {e}")

    def _dtype_to_str(self, dtype_obj) -> str:
        """
        Helper to convert internal dtype object back to string for report.
        Example: infinicore.float32 -> 'float32'
        If it is already a string, return it as is.
        """
        if isinstance(dtype_obj, str):
            return dtype_obj.split('.')[-1] # Handle "infinicore.float32" string
        return str(dtype_obj).split('.')[-1]


# =========================================================================
# Concrete Class: InfiniCoreAdapter
# =========================================================================
class InfiniCoreAdapter(ExternalSystemAdapter):
    """
    Concrete Adapter Implementation.
    Transforms External JSON -> Internal JSON File -> Final Response JSON
    """
    
    def parseRequest(self, json_file_path: str) -> str:
        """
        [Modified]
        Reads external JSON, converts it to the Standard Test Suite JSON format,
        saves it to a temporary file, and returns the file path.
        """
        req_data = self._load_and_validate_json(json_file_path)
        config_data = req_data.get("config", {})

        # 1. Extract Basic Info
        op_name = config_data.get("operator", "Unknown")
        target_device_str = config_data.get("device", "cpu").lower()

        # 2. Map Device Name (e.g. "cuda" -> "NVIDIA")
        device_display_name = "CPU" 
        for backend in DeviceBackend:
            if backend.value in target_device_str:
                device_display_name = backend.name # Use Enum name (e.g. NVIDIA)
                break
        
        # 3. Build Inputs List (Keep as pure JSON for the Manager)
        inputs_list = []
        input_name_to_index = {}
        for idx, inp in enumerate(config_data.get("inputs", [])):
            inputs_list.append({
                "name": inp["name"],
                "shape": inp["shape"],
                "dtype": inp["dtype"], 
                "strides": inp.get("strides", None)
            })
            input_name_to_index[inp["name"]] = idx

        # 4. Build Kwargs (Attributes List -> Dict)
        kwargs = {}
        raw_attrs = config_data.get("attributes", [])
        if isinstance(raw_attrs, list):
            for attr in raw_attrs:
                if isinstance(attr, dict):
                    kwargs[attr.get("name")] = attr.get("value")
        elif isinstance(raw_attrs, dict):
            kwargs.update(raw_attrs)

        # 5. Handle Outputs / Inplace Logic
        comparison_target = None
        desc_suffix = "Standard"
        raw_outputs = config_data.get("outputs", [])
        output_spec = None

        if raw_outputs:
            out_def = raw_outputs[0]
            
            # Case A: Inplace
            if "inplace" in out_def:
                target_name = out_def['inplace']
                if target_name in input_name_to_index:
                    target_idx = input_name_to_index[target_name]
                    # Set 'out' index
                    kwargs['out'] = target_idx
                    # Set comparison target
                    comparison_target = target_idx
                    desc_suffix = f"INPLACE({target_name})"
            
            # Case B: Explicit Output
            else:
                # Construct output spec dictionary
                output_spec = {
                    "name": out_def.get("name"),
                    "shape": out_def.get("shape"),
                    "dtype": out_def.get("dtype"),
                    "strides": out_def.get("strides")
                }
                if out_def.get("name"):
                    kwargs['out_name'] = out_def["name"]

        # 6. Tolerance
        metrics_cfg = req_data.get("metrics", [])
        tolerance = {"atol": 1e-3, "rtol": 1e-3}
        for m in metrics_cfg:
            if m.get("type") == "tensor_diff" and "tolerance" in m:
                tolerance = m["tolerance"]
                break

        # 7. Assemble Suite Configuration
        suite_config = {
            "operator": op_name,
            "device": device_display_name,
            "torch_op": f"torch.{op_name.lower()}",
            "infinicore_op": f"infinicore.{op_name.lower()}",
            
            # Runtime Arguments
            "args": {
                "bench": config_data.get("bench", "both"),
                "num_prerun": int(config_data.get("warmup_iterations", 10)),
                "num_iterations": int(config_data.get("measured_iterations", 100)),
                "verbose": False,
                "debug": False
            },
            
            # Test Cases
            "testcases": [
                {
                    "description": f"{op_name} - {desc_suffix}",
                    "inputs": inputs_list,
                    "kwargs": kwargs,
                    "comparison_target": comparison_target,
                    "tolerance": tolerance,
                    # Add output_spec only if it exists
                    **({"output_spec": output_spec} if output_spec else {})
                }
            ]
        }

        # Wrap in List (TestCaseManager expects a List of Suites)
        final_internal_json = [suite_config]

        # 8. Save to Temp File
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_filename = f"internal_req_{op_name}_{timestamp}.json"
        temp_path = os.path.abspath(temp_filename)
        
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(final_internal_json, f, indent=4)
            # print(f"💾 [Adapter] Transformed request saved to: {temp_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to write temp config file: {e}")

        return temp_path

    def formatResponse(self, run_output: Dict[str, Any]) -> str:
        """
        Receives the dict from Gateway and formats it back to the original Request's style.
        Adapted for the structure where 'result' is embedded inside 'testcases'.
        """
        # 1. Safety Check
        # Check if testcases exist
        testcases = run_output.get("testcases", [])
        if not testcases:
            return json.dumps({"error": "No testcases found in output"}, indent=2)

        # Assuming 1-to-1 mapping (Single Operator Benchmark)
        # We take the first testcase
        test_case = testcases[0]
        
        # Extract the result dictionary embedded in the testcase
        result_data = test_case.get("result", {})
        
        # Extract args (Global config)
        args_obj = run_output.get("args", {})

        # Helper to safely get values from args (dict or object)
        def get_arg(obj, key, default):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # ---------------------------------------------------------
        # 2. Reconstruct Config
        # ---------------------------------------------------------
        
        # A. Restore Inputs
        inputs_json = []
        raw_inputs = test_case.get("inputs", []) 
        for inp in raw_inputs:
            inputs_json.append({
                "name": inp.get("name"),
                "dtype": self._dtype_to_str(inp.get("dtype")),
                "shape": list(inp.get("shape")),
                "strides": list(inp.get("strides")) if inp.get("strides") else None
            })
        
        # B. Restore Attributes (kwargs)
        attributes_json = []
        raw_kwargs = test_case.get("kwargs", {})
        for k, v in raw_kwargs.items():
            if k not in ["out", "out_name"]: 
                attributes_json.append({"name": k, "value": v})

        # C. Restore Outputs logic
        outputs_json = []
        
        # Case 1: Inplace
        if "out" in raw_kwargs:
            out_val = raw_kwargs["out"]
            # Check inputs by index
            if isinstance(out_val, int) and 0 <= out_val < len(raw_inputs):
                target_name = raw_inputs[out_val].get("name")
                outputs_json.append({"inplace": target_name})
        
        # Case 2: Explicit Output Spec
        else:
            specs_to_process = []
            if test_case.get("output_specs"):
                specs_to_process = test_case["output_specs"]
            elif test_case.get("output_spec"):
                specs_to_process = [test_case["output_spec"]]
            
            for spec in specs_to_process:
                out_def = {
                    "name": spec.get("name"),
                    "dtype": self._dtype_to_str(spec.get("dtype")),
                    "shape": list(spec.get("shape")) if spec.get("shape") else None,
                    "strides": list(spec.get("strides")) if spec.get("strides") else None
                }
                if "out_name" in raw_kwargs:
                    out_def["name"] = raw_kwargs["out_name"]
                
                if out_def["shape"] or out_def["name"]:
                     outputs_json.append(out_def)

        config_obj = {
            "model_source": "Manual_Test",
            "operator": run_output.get("operator", "Unknown"),
            "device": run_output.get("device", "cpu"), 
            "warmup_iterations": int(get_arg(args_obj, "num_prerun", 0)),
            "measured_iterations": int(get_arg(args_obj, "num_iterations", 0)),
            "bench": str(get_arg(args_obj, "bench", "False")),
            "attributes": attributes_json,
            "inputs": inputs_json,
            "outputs": outputs_json
        }

        # ---------------------------------------------------------
        # 3. Build Metrics (From embedded 'result')
        # ---------------------------------------------------------
        
        # Extract Status
        status_info = result_data.get("status", {})
        is_success = status_info.get("success", False)
        error_msg = status_info.get("error", "")

        # Extract Latency (Perf)
        perf_info = result_data.get("perf_ms", {})
        # Target infinicore device time
        latency_val = perf_info.get("infinicore", {}).get("device", 0.0)

        metrics_list = [
            {
                "name": "operator.latency",
                "value": round(float(latency_val), 4),
                "unit": "ms",
                "type": "timeseries"
            },
            {
                "name": "operator.tensor_accuracy",
                "status": "PASS" if is_success else "FAIL",
                "unit": "",
                "type": "tensor_diff",
                "info": str(error_msg) if not is_success else "Matches reference"
            }
        ]

        response = {
            "config": config_obj,
            "metrics": metrics_list
        }

        return json.dumps(response, indent=4)


# ==========================================
# Execution Entry Point (Mock Test)
# ==========================================
if __name__ == "__main__":
    
    # Simulate the framework logic for testing formatResponse
    from framework import TestCase as FrameworkTestCase, TensorSpec as FrameworkTensorSpec
    
    def run_full_flow_test(adapter, file_path):
        print("=" * 60)
        print(f"📂 Processing Input: {file_path}")
        print("-" * 60)

        try:
            # 1. Adapter: Parse Request -> Saves Temp File -> Returns Path
            temp_file_path = adapter.parseRequest(file_path)
            print(f"✅ [ParseRequest] Created Internal Config: {temp_file_path}")
            
            # --- SIMULATION OF GATEWAY & MANAGER ---
            # In a real run, Gateway would pass temp_file_path to TestCaseManager.
            # Here we mock what the Manager would return after running that file.
            
            # Read back the temp file to know what to mock
            with open(temp_file_path, 'r') as f:
                suite_data = json.load(f)[0] # List[0]
            
            # Mock Runtime Args
            class MockArgs:
                def __init__(self, d):
                    self.num_prerun = d['num_prerun']
                    self.num_iterations = d['num_iterations']
                    self.bench = d['bench']
            
            # Mock Result
            class MockResult:
                success = True
                infini_device_time = 1.2345
                error_message = ""

            # Reconstruct TestCase Object (Mocking what Manager does)
            # Simplified reconstruction for the sake of the test
            mock_inputs = [
                FrameworkTensorSpec.from_tensor(shape=tuple(i['shape']), dtype=infinicore.float32, name=i['name'])
                for i in suite_data['testcases'][0]['inputs']
            ]
            
            mock_case = FrameworkTestCase(
                inputs=mock_inputs,
                kwargs=suite_data['testcases'][0]['kwargs'],
                description="Mocked Case",
                output_count=1
            )

            # Construct the Payload expected by formatResponse
            run_output_payload = {
                "operator": suite_data["operator"],
                "device": suite_data["device"],
                "torch_op": suite_data["torch_op"],
                "infinicore_op": suite_data["infinicore_op"],
                "args": MockArgs(suite_data["args"]),
                "testcases": [mock_case],
                "execution_results": [MockResult()]
            }
            
            # 2. Adapter: Format Response
            print("📝 Formatting Response...")
            final_json = adapter.formatResponse(run_output_payload)
            print(final_json)
            
            # Cleanup
            os.remove(temp_file_path)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n❌ Error: {e}")

    # Create dummy input file
    input_file = "test_input.json"
    content = {
        "config": {
            "model_source": "Manual_Test",
            "operator": "Add",
            "inputs": [
                {"name": "a", "dtype": "float32", "shape": [13, 4], "strides": [4, 1]},
                {"name": "b", "dtype": "float32", "shape": [13, 4], "strides": [4, 1]}
            ],
            "outputs": [{"inplace": "a"}],
            "warmup_iterations": 10,
            "measured_iterations": 1000,
            "device": "cuda", # Should map to NVIDIA
            "bench": "both"
        }
    }
    
    try:
        with open(input_file, 'w') as f:
            json.dump(content, f, indent=2)
        
        adapter = InfiniCoreAdapter()
        run_full_flow_test(adapter, input_file)
        
    finally:
        if os.path.exists(input_file):
            os.remove(input_file)
