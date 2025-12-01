import abc
import argparse
import json
import os
import sys
from typing import Dict, Any, Tuple, Optional, List

# --- Third-party / Internal Dependencies ---
import infinicore 

# --- Framework Dependencies ---
# Ensure these are importable from your current directory structure
from framework import TestCase, TensorSpec, TestConfig
from framework.config import get_args


# =========================================================================
# Base Class: ExternalSystemAdapter
# =========================================================================
class ExternalSystemAdapter(abc.ABC):
    """
    Abstract Adapter Class (Rich Interface)
    
    Defines standard interfaces for parsing external requests.
    Provides shared utility methods for:
    1. JSON File Loading & Validation
    2. Runtime Argument Initialization
    3. Common Configuration Parsing (Device, Iterations)
    4. Data Type Mapping
    """

    @abc.abstractmethod
    def parseRequest(self, json_file_path: str) -> Any:
        """
        Abstract method to be implemented by subclasses.
        Should orchestrate the parsing flow.
        Returns: (op_name, test_case, runtime_args)
        """
        pass

    @abc.abstractmethod
    def formatResponse(self, test_results: List[Any]) -> Any:
        """Format internal result object into JSON string required by external system"""
        pass

    # -------------------------------------------------------------------------
    # Shared Utility Methods (Available to all subclasses)
    # -------------------------------------------------------------------------

    def _load_and_validate_json(self, file_path: str) -> Dict[str, Any]:
        """
        Common: Validates file existence and parses JSON.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON file not found at: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {e}")

    def _initialize_base_args(self) -> argparse.Namespace:
        """
        Common: Initializes the standard argparse object cleanly.
        Prevents interference from current sys.argv.
        """
        original_argv = sys.argv
        sys.argv = [sys.argv[0]] 
        args = get_args()
        sys.argv = original_argv
        
        # Reset all hardware flags to False by default to ensure clean state
        for flag in ['cpu', 'nvidia', 'cambricon', 'ascend', 'iluvatar', 
                     'metax', 'moore', 'kunlun', 'hygon', 'qy']:
            if hasattr(args, flag):
                setattr(args, flag, False)
        return args

    def _apply_common_config(self, config_data: Dict[str, Any], args: argparse.Namespace):
        """
        Common: Maps standard fields (Iterations, Device) to args.
        Note: Specific benchmark flags (like args.bench) should be handled by the concrete adapter.
        """
        # 1. Map Iterations
        if "warmup_iterations" in config_data:
            args.num_prerun = int(config_data["warmup_iterations"])
        
        if "measured_iterations" in config_data:
            args.num_iterations = int(config_data["measured_iterations"])

        # 2. Map Device (String -> Boolean Flag)
        target_device = config_data.get("device", config_data.get("target_device", "cpu")).lower()
        
        device_map = {
            "cpu": "cpu",
            "cuda": "nvidia", "nvidia": "nvidia",
            "mlu": "cambricon", "cambricon": "cambricon",
            "npu": "ascend", "ascend": "ascend",
            "iluvatar": "iluvatar",
            "metax": "metax",
            "musa": "moore", "moore": "moore",
            "xpu": "kunlun", "kunlun": "kunlun",
            "dcu": "hygon", "hygon": "hygon"
        }

        matched_flag = "cpu" # Default fallback
        for key, flag in device_map.items():
            if key in target_device:
                matched_flag = flag
                break
        
        if hasattr(args, matched_flag):
            setattr(args, matched_flag, True)
        else:
            print(f"⚠️ [Adapter] Warning: Device flag '{matched_flag}' not found in args. Defaulting to CPU.")
            args.cpu = True

    def _parse_dtype(self, dtype_str: str):
        """
        Common: Converts string dtype to infinicore.dtype.
        """
        dtype_map = {
            "float16": infinicore.float16,
            "float32": infinicore.float32,
            "bfloat16": infinicore.bfloat16,
            "int8": infinicore.int8,
            "int32": infinicore.int32,
            "int64": infinicore.int64,
            "bool": infinicore.bool,
        }
        return dtype_map.get(dtype_str, infinicore.float32)


# =========================================================================
# Concrete Class: InfiniCoreAdapter
# =========================================================================
class InfiniCoreAdapter(ExternalSystemAdapter):
    """
    Concrete Adapter Implementation.
    Leverages base class for IO/Config and implements specific logic for 
    Operator Benchmarks.
    """
    def __init__(self):
        # Store metrics template from request
        self.req_metrics = []

    def parseRequest(self, json_file_path: str) -> Tuple[str, TestCase, argparse.Namespace]:
        # 1. Use Base Method: Load JSON
        req_data = self._load_and_validate_json(json_file_path)
        
        # [Fix] Save metrics template here so we can use it in formatResponse
        self.req_metrics = req_data.get("metrics", [])
        
        config_data = req_data.get("config", {})

        # 2. Use Base Method: Init Args
        args = self._initialize_base_args()

        # 3. Use Base Method: Parse Common Config (Device, Iterations)
        self._apply_common_config(config_data, args)

        # 4. Apply InfiniCore Specific Logic (Bench Flag)
        # If measured iterations are set, auto-enable benchmark mode for this specific adapter
        if args.num_iterations > 0 and not args.bench:
            args.bench = "both"

        # 5. Operator Logic Only
        op_name = "Unknown"
        test_case = None

        if "operator" in config_data:
            # print("   [Adapter] Detected Type: Operator Benchmark") # Optional logging
            op_name, test_case = self._parse_operator_case(config_data)
        else:
            raise ValueError("Unknown JSON config type: Missing 'operator' field.")

        return op_name, test_case, args

    # -------------------------------------------------------------------------
    # Response Formatting Logic (Matches Input Metrics)
    # -------------------------------------------------------------------------
    def formatResponse(self, test_results: List[Any]) -> str:
        """
        Formats the results into a JSON object strictly following the 'metrics' 
        structure from the input request.
        """
        if not test_results:
            return json.dumps({"error": "No results generated"}, indent=2)

        # Use the first result (assuming single operator benchmark context)
        result = test_results[0]
        
        filled_metrics = []
        
        # If no metrics were provided in input, fallback to simple dump
        if not self.req_metrics:
            return json.dumps({"results": "No metrics template in request", "raw": str(result)}, indent=2)

        for metric_template in self.req_metrics:
            # Create a copy to fill in values
            metric = metric_template.copy()
            name = metric.get("name")
            
            # --- 1. Fill Latency ---
            if name == "operator.latency":
                # Get latency from InfiniCore device time (ms)
                latency = getattr(result, "infini_device_time", 0.0)
                metric["value"] = round(latency, 4)
            
            # --- 2. Fill Accuracy ---
            elif name == "operator.tensor_accuracy":
                # Get success status
                is_success = getattr(result, "success", False)
                metric["status"] = "PASS" if is_success else "FAIL"
                
                # Optional: Add error message if failed
                if not is_success:
                    metric["error_message"] = getattr(result, "error_message", "Unknown")

            filled_metrics.append(metric)

        # Wrap in a root object
        response = {
            "metrics": filled_metrics
        }
        
        return json.dumps(response, indent=2)

    # -------------------------------------------------------------------------
    # Specific Parsing Logic
    # -------------------------------------------------------------------------

    def _parse_operator_case(self, config_data: Dict[str, Any]) -> Tuple[str, TestCase]:
        """Parses Operator-specific JSON structure (Inputs, Outputs, Modes)."""
        op_name = config_data.get("operator")
        kwargs = {}

        # A. Parse Attributes (List/Dict compatibility)
        raw_attrs = config_data.get("attributes", [])
        if isinstance(raw_attrs, list):
            for attr in raw_attrs:
                if isinstance(attr, dict):
                    kwargs[attr.get("name")] = attr.get("value")
        elif isinstance(raw_attrs, dict):
            kwargs.update(raw_attrs)

        # B. Parse Inputs (TensorSpecs)
        inputs_list = []
        input_name_to_index = {}
        for idx, inp in enumerate(config_data.get("inputs", [])):
            # Use Base Method: _parse_dtype
            spec = TensorSpec.from_tensor(
                shape=tuple(inp["shape"]),
                strides=tuple(inp["strides"]) if inp.get("strides") else None,
                dtype=self._parse_dtype(inp["dtype"]), 
                name=inp.get("name")
            )
            inputs_list.append(spec)
            input_name_to_index[inp.get("name")] = idx

        # C. Parse Outputs & Execution Mode
        raw_outputs = config_data.get("outputs", [])
        parsed_output_specs = []
        comparison_target = None
        
        # Start constructing description
        desc_parts = [op_name]

        if raw_outputs:
            out_def = raw_outputs[0]
            
            # --- Inplace Logic ---
            if "inplace" in out_def:
                target_name = out_def['inplace']
                desc_parts.append(f"INPLACE({target_name})")
                
                # Map inplace target name -> input index -> kwargs['out']
                if target_name in input_name_to_index:
                    target_idx = input_name_to_index[target_name]
                    
                    # Store Index (int) so the runner can retrieve the Tensor object
                    # TestCase.__str__ will display this as "out=name"
                    kwargs['out'] = target_idx 
                    
                    # Comparison target needs index
                    comparison_target = target_idx
                else:
                    print(f"⚠️ Warning: Inplace target '{target_name}' not found in inputs.")
            
            # --- Explicit Out Logic ---
            elif out_def.get("name"):
                kwargs['out_name'] = out_def["name"]

            # Construct specs (Only if NOT inplace)
            if "inplace" not in out_def:
                for out_def_item in raw_outputs:
                    spec = TensorSpec.from_tensor(
                        shape=tuple(out_def_item["shape"]),
                        strides=tuple(out_def_item["strides"]) if out_def_item.get("strides") else None,
                        dtype=self._parse_dtype(out_def_item["dtype"]),
                        name=out_def_item.get("name")
                    )
                    parsed_output_specs.append(spec)

        # Detect Complex Strides for description
        if any(inp.strides is not None for inp in inputs_list):
            desc_parts.append("(Complex Strides)")
        
        full_desc = " - ".join(desc_parts)

        # D. Parse Tolerance (Optional, with default)
        default_tolerance = {"atol": 1e-3, "rtol": 1e-3}
        tolerance = config_data.get("tolerance", default_tolerance)

        # E. Construct TestCase
        output_count = len(parsed_output_specs)
        
        # If Inplace (kwargs has 'out'), output_count is logically 1
        if "out" in kwargs:
            output_count = 1
            final_output_spec = None
            final_output_specs = None
        else:
            # Standard logic
            final_count_arg = output_count if output_count > 0 else 1
            final_output_specs = parsed_output_specs if output_count > 1 else None
            final_output_spec = parsed_output_specs[0] if output_count == 1 else None

        test_case = TestCase(
            inputs=inputs_list,
            kwargs=kwargs,
            output_spec=final_output_spec,
            output_specs=final_output_specs,
            output_count=output_count if output_count > 0 else 1,
            comparison_target=comparison_target,
            description=full_desc,
            tolerance=tolerance
        )
        return op_name, test_case


# ==========================================
# Execution Entry Point
# ==========================================
if __name__ == "__main__":
    
    def run_test_with_file(adapter, file_path, description):
        print("=" * 60)
        print(f"🧪 TEST SCENARIO: {description}")
        print(f"📂 File: {file_path}")
        print("-" * 60)

        try:
            op_name, case, runtime_args = adapter.parseRequest(file_path)
            
            print(f"✅ [Parse Result] Target Name: {op_name}")
            print(f"   🔹 Description: {case.description}")
            if 'out' in case.kwargs:
                 print(f"   🔹 Execution Mode: INPLACE (Mapped to Index: {case.kwargs['out']})")
            else:
                 print(f"   🔹 Execution Mode: Standard / Explicit")
            print(f"   🔹 Kwargs: {case.kwargs}")
            
            # Simulate a result for formatResponse testing
            class MockResult:
                def __init__(self):
                    self.success = True
                    self.infini_device_time = 12.3456
                    self.error_message = ""
            
            # Test formatResponse
            print("-" * 60)
            print("📝 Testing formatResponse (Simulated Result):")
            mock_results = [MockResult()]
            json_response = adapter.formatResponse(mock_results)
            print(json_response)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n❌ Error: {e}")

    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, nargs="?", help="Path to input JSON")
    args = parser.parse_args()
    
    adapter = InfiniCoreAdapter()

    if args.file_path:
        run_test_with_file(adapter, args.file_path, "Manual Input File")
    else:
        print(f"🛠️  Mode: No input provided. Running auto-generated scenarios.")
        
        scenarios = [
            {
                "desc": "Scenario 1: Operator with Metrics",
                "filename": "temp_test_metrics.json",
                "content": {
                    "config": {
                        "operator": "Mul",
                        "attributes": [],
                        "inputs": [{"name": "a", "dtype": "float32", "shape": [4,4]}],
                        "outputs": [],
                        "device": "cpu",
                        "measured_iterations": 100
                    },
                    "metrics": [
                        {
                            "name": "operator.latency",
                            "type": "timeseries",
                            "raw_data_url": "./operator/${run_id}_latency.csv",
                            "unit": "ms"
                        },
                        {
                            "name": "operator.tensor_accuracy",
                            "type": "tensor_diff",
                            "unit": ""
                        }
                    ]
                }
            }
        ]

        for scen in scenarios:
            fname = scen["filename"]
            try:
                with open(fname, 'w') as f:
                    json.dump(scen["content"], f, indent=2)
                run_test_with_file(adapter, fname, scen["desc"])
            finally:
                if os.path.exists(fname):
                    os.remove(fname)
        
        print("\n" + "="*60)
        print("✅ All scenarios completed.")
