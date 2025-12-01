import abc
import argparse
import json
import os
import sys
import infinicore
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from framework import TestCase, TensorSpec, TestConfig

from framework.config import get_args

class ExternalSystemAdapter(abc.ABC):
    """
    Abstract Adapter Class (Interface)
    Defines standard interfaces for parsing external requests and formatting external responses
    """
    
    @abc.abstractmethod
    def parseRequest(self, json_file_path: str) -> Any:
        """Parse external JSON string into internal standard object"""
        pass

    @abc.abstractmethod
    def formatResponse(self) -> Any:
        """Format internal result object into JSON string required by external system"""
        pass


class InfiniCoreAdapter(ExternalSystemAdapter):
    def parseRequest(self, json_file_path: str) -> Tuple[str, TestCase, argparse.Namespace]:
        """
        Parse JSON request from a file and map to TestCase and Arguments.
        
        Returns:
            op_name (str): Operator name (e.g., "Conv")
            test_case (TestCase): Constructed TestCase object
            args (argparse.Namespace): Argument object containing runtime config (hardware, iterations, etc.)
        """
        # 1. Check if file exists
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"JSON file not found at: {json_file_path}")

        # 2. Read and parse JSON file
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                req = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {json_file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {e}")

        config_data = req.get("config", {})
        
        # ========================================================
        # 3. Construct Args Object (Config)
        # ========================================================
        
        # A. Get clean default arguments (by temporarily clearing sys.argv)
        # This is done to prevent get_args() from reading the current script's command-line arguments and raising an error
        original_argv = sys.argv
        sys.argv = [sys.argv[0]] 
        args = get_args()
        sys.argv = original_argv

        # B. Map basic configuration (Override defaults with JSON values)
        if "warmup_iterations" in config_data:
            args.num_prerun = int(config_data["warmup_iterations"])
        
        if "measured_iterations" in config_data:
            args.num_iterations = int(config_data["measured_iterations"])
            # If measurement iterations are specified, enable benchmark mode by default
            if args.num_iterations > 0 and not args.bench:
                args.bench = "both"

        # C. Hardware device mapping (Map 'device' string to boolean flags)
        # Prioritize reading 'device'; if not present, try 'target_device', default fallback to 'cpu'
        target_device = config_data.get("device", config_data.get("target_device", "cpu")).lower()
        
        # Reset all hardware flags (Ensure they are unaffected by the default environment, set all to False)
        args.cpu = False
        args.nvidia = False
        args.cambricon = False
        args.ascend = False
        args.iluvatar = False
        args.metax = False
        args.moore = False
        args.kunlun = False
        args.hygon = False
        args.qy = False

        # Set the corresponding Flag based on the JSON string (Only one will be set to True)
        if "cpu" in target_device:
            args.cpu = True
        elif "cuda" in target_device or "nvidia" in target_device:
            args.nvidia = True
        elif "mlu" in target_device or "cambricon" in target_device:
            args.cambricon = True
        elif "npu" in target_device or "ascend" in target_device:
            args.ascend = True
        elif "iluvatar" in target_device:
            args.iluvatar = True
        elif "metax" in target_device:
            args.metax = True
        elif "musa" in target_device or "moore" in target_device:
            args.moore = True
        elif "xpu" in target_device or "kunlun" in target_device:
            args.kunlun = True
        elif "dcu" in target_device or "hygon" in target_device:
            args.hygon = True
        else:
            print(f"⚠️ Warning: Unknown device '{target_device}' in JSON. Fallback to CPU.")
            args.cpu = True

        # ========================================================
        # 4. Extract Basic Op Info
        # ========================================================
        op_name = config_data.get("operator")  # e.g., "Conv"
        
        # 5. Parse Attributes -> Kwargs
        kwargs = {}
        attributes = config_data.get("attributes", [])
        for attr in attributes:
            name = attr.get("name")
            value = attr.get("value")
            kwargs[name] = value

        # 6. Parse Inputs -> List[TensorSpec]
        inputs_list = []
        input_name_to_index = {} 
        
        raw_inputs = config_data.get("inputs", [])
        for idx, inp in enumerate(raw_inputs):
            spec = TensorSpec.from_tensor(
                shape=tuple(inp["shape"]),
                strides=tuple(inp["strides"]) if inp.get("strides") else None,
                dtype=self._parse_dtype(inp["dtype"]),
                name=inp.get("name")
            )
            inputs_list.append(spec)
            input_name_to_index[inp.get("name")] = idx

        # 7. Parse Outputs & Inplace logic
        raw_outputs = config_data.get("outputs", [])
        comparison_target = None
        output_spec = None
        
        if raw_outputs:
            out_def = raw_outputs[0]
            
            output_spec = TensorSpec.from_tensor(
                shape=tuple(out_def["shape"]),
                strides=tuple(out_def["strides"]) if out_def.get("strides") else None,
                dtype=self._parse_dtype(out_def["dtype"]),
                name=out_def.get("name")
            )

            inplace_target_name = out_def.get("inplace")
            if inplace_target_name:
                if inplace_target_name in input_name_to_index:
                    comparison_target = input_name_to_index[inplace_target_name]
                    output_spec = None 
                else:
                    print(f"⚠️ Warning: Output specifies inplace='{inplace_target_name}', but no such input found.")
            else:
                comparison_target = None 
                output_spec = None 

        # 8. Construct TestCase object
        test_case = TestCase(
            inputs=inputs_list,
            kwargs=kwargs,
            output_spec=output_spec,
            comparison_target=comparison_target,
            tolerance={"atol": 1e-3, "rtol": 1e-3}, 
            description=f"Auto-generated test for {op_name} (from {os.path.basename(json_file_path)})"
        )

        return op_name, test_case, args

    def _parse_dtype(self, dtype_str: str):
        """Helper function: Convert string dtype to infinicore.dtype"""
        dtype_map = {
            "float16": infinicore.float16,
            "float32": infinicore.float32,
            "bfloat16": infinicore.bfloat16,
            "int8": infinicore.int8,
            "int32": infinicore.int32,
        }
        return dtype_map.get(dtype_str, infinicore.float32)

    def formatResponse(self) -> str:
        """
        Temporary logic: Returns a hardcoded JSON string without input parameters.
        """
        print("    [Adapter] Executing temporary formatResponse logic (No input args)...")
        
        # Construct a dummy response
        response_dict = {
            "header": {
                "source": "InfiniCore",
                "status": "OK"
            },
            "payload": {
                "task_id": "DUMMY_TASK_ID",
                "execution_status": "PASS",
                "metrics": {
                    "latency_ms": 12.34
                },
                "system_logs": "Simulated log: Execution finished successfully."
            }
        }
        return json.dumps(response_dict, indent=2)


# ==========================================
# Execution Entry Point
# ==========================================
if __name__ == "__main__":
    
    # 1. Configure command line parser
    parser = argparse.ArgumentParser(description="InfiniMetrics Adapter Test Runner")
    parser.add_argument("file_path", type=str, nargs="?", help="Path to the input JSON file")
    
    args = parser.parse_args()
    
    # 2. Determine which file to use
    json_file_to_use = ""
    is_temp_file = False

    if args.file_path:
        # A. Use provided file
        print(f"📂 Mode: Using external file -> {args.file_path}")
        json_file_to_use = args.file_path
    else:
        # B. Use hardcoded default file
        print(f"🛠️ Mode: No input provided. Using hardcoded default file.")
        json_file_to_use = "default_hardcoded_request.json"
        is_temp_file = True 
        
        # Hardcoded JSON content
        hardcoded_json = """
        {
            "config": {
                "operator": "Conv",
                "attributes": [
                    {"name": "kernel_shape", "value": [3, 3]},
                    {"name": "strides", "value": [1, 1]}
                ],
                "inputs": [
                    {
                        "name": "X", "dtype": "float16", "shape": [1, 64, 256, 256]
                    },
                    {
                        "name": "W", "dtype": "float16", "shape": [128, 64, 3, 3]
                    }
                ],
                "outputs": [
                    {
                        "name": "Y", "dtype": "float16", "shape": [1, 128, 254, 254]
                    }
                ],
                "warmup_iterations": 10,
                "measured_iterations": 50,
                "device": "cuda" 
            }
        }
        """
        with open(json_file_to_use, 'w') as f:
            f.write(hardcoded_json)

    # 3. Run Adapter Flow
    adapter = InfiniCoreAdapter()
    
    try:
        # --- Step A: Parse Request ---
        print("-" * 40)
        print(f"1. Parsing Request from: {json_file_to_use}")
        
        # Note: The third value returned here is now runtime_args (argparse.Namespace)
        op_name, case, runtime_args = adapter.parseRequest(json_file_to_use)
        
        # 🖨️ Print detailed return values
        print(f"   ✅ [Result 1] Operator Name: {op_name}")
        
        print(f"   ✅ [Result 2] TestCase Object:")
        print(f"      ├── Description: {case.description}")
        print(f"      ├── Inputs ({len(case.inputs)}):")
        for i, inp in enumerate(case.inputs):
            print(f"      │   [{i}] Name: {inp.name}")
            print(f"      │       Shape: {inp.shape}")
        print(f"      ├── Kwargs: {case.kwargs}")

        print(f"   ✅ [Result 3] Runtime Args (Config):")
        # Print all attributes in args
        for key, value in vars(runtime_args).items():
             print(f"      ├── {key:<15}: {value}")

        # --- Step B: Format Response (No Args) ---
        print("-" * 40)
        print("2. Formatting Response...")
        final_json_response = adapter.formatResponse()
        
        print("\n[Final Output JSON]:")
        print(final_json_response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Error: {e}")
    
    finally:
        # Cleanup temp file
        if is_temp_file and os.path.exists(json_file_to_use):
            print(f"\n🧹 Cleaning up temporary file: {json_file_to_use}")
            os.remove(json_file_to_use)
