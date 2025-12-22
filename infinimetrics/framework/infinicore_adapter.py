import json
import copy
from datetime import datetime


class InfiniCoreAdapter:
    def __init__(self):
        # Operator mapping: Legacy (lower) -> InfiniCore (PascalCase)
        self.op_mapping = {
            "mul": "Mul",
            "add": "Add",
            "sub": "Sub",
            "div": "Div",
            "matmul": "MatMul",
        }

    def convert_to_infinicore_request(self, legacy_json: dict) -> list:
        """
        Convert legacy add.json to InfiniCore new_add.json request format.
        Focuses on extracting 'config' to build the test args.
        """
        config = legacy_json.get("config", {})

        # 1. Map Operator
        legacy_op = config.get("operator", "").lower()
        infinicore_op = self.op_mapping.get(legacy_op, legacy_op.capitalize())

        # 2. Build Inputs
        infinicore_inputs = []
        for inp in config.get("inputs", []):
            new_inp = {
                "name": inp.get("name"),
                "shape": inp.get("shape"),
                "dtype": inp.get("dtype"),
                "strides": inp.get("strides"),
            }
            infinicore_inputs.append(new_inp)

        # 3. Build Testcase
        testcase = {
            "description": f"{infinicore_op} - Adapter Generated",
            "inputs": infinicore_inputs,
            "kwargs": {"out": "b"},  # Assuming inplace/output logic
            "comparison_target": 1,
            "tolerance": {"atol": 1e-5, "rtol": 1e-5},
        }

        # 4. Build Top-Level Request
        infinicore_req = {
            "operator": infinicore_op,
            "device": config.get("device", "cuda").upper(),
            "torch_op": f"torch.{legacy_op}",
            "infinicore_op": f"infinicore.{legacy_op}",
            "args": {
                "bench": config.get("bench", "both"),
                "num_prerun": config.get("warmup_iterations", 10),
                "num_iterations": config.get("measured_iterations", 100),
                "verbose": False,
                "debug": False,
            },
            "testcases": [testcase],
        }

        return [infinicore_req]

    def run_infinicore_test(self, infinicore_req: list) -> list:
        """
        Mock execution of the InfiniCore backend.
        """
        print(">> [System] Executing InfiniCore Backend...")

        response = copy.deepcopy(infinicore_req)

        # Mocking a SUCCESSFUL result
        # Change 'success': False to simulate a failure
        mock_result = {
            "status": {"success": True, "error": ""},
            "perf_ms": {
                "torch": {"host": 10.5, "device": 100.0},
                "infinicore": {"host": 25.0, "device": 25.3},
            },
        }

        # Inject result
        response[0]["testcases"][0]["result"] = mock_result
        return response

    def convert_from_infinicore_response(
        self, infinicore_response: list, original_legacy_json: dict
    ) -> dict:
        """
        Reconstruct the legacy add.json format.
        Requirements:
        1. Preserve run_id, testcase.
        2. Generate current time.
        3. Map success status (Bool -> Int).
        4. Fill metrics.
        """
        # Deep copy to preserve original structure (run_id, testcase, etc.)
        final_json = copy.deepcopy(original_legacy_json)

        # 1. Update Time (Generate Current Time)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        final_json["time"] = current_time

        try:
            # Extract result block
            result_block = infinicore_response[0]["testcases"][0]["result"]

            # 2. Update Success Field
            # InfiniCore uses boolean (True/False), Legacy uses int (1/0)
            is_success = result_block["status"]["success"]
            final_json["success"] = 1 if is_success else 0

            # 3. Fill Metrics (Latency)
            latency_val = result_block["perf_ms"]["infinicore"]["device"]

            if "metrics" in final_json:
                for metric in final_json["metrics"]:
                    if metric["name"] == "operator.latency":
                        # We inject the actual value here.
                        # Note: The raw_data_url remains pointing to the CSV as per legacy format.
                        metric["value"] = latency_val

        except (KeyError, IndexError, TypeError) as e:
            print(f"Error parsing results: {e}")
            final_json["success"] = 0  # Mark as failed if parsing fails

        return final_json


# ==========================================
# Demo Execution
# ==========================================

if __name__ == "__main__":
    # The specific input provided in the prompt
    input_legacy_data = {
        "run_id": "train.infiniTrain.SFT.a8b4c9e1-5b1a-4f8a-9e3b-b2c1d0f8e3a4",
        "time": "2025-10-11 14:50:50",
        "testcase": "train.InfiniTrain.SFT",
        "success": 0,
        "config": {
            "model_source": "Manual_Test",
            "opset_version": 11,
            "operator": "mul",
            "attributes": [],
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
            "outputs": [
                {
                    "name": "c",
                    "dtype": "float32",
                    "shape": [13, 4, 4],
                    "strides": [20, 4, 1],
                }
            ],
            "warmup_iterations": 10,
            "measured_iterations": 100,
            "command": "python base_op_bench.py ...",
            "device": "cuda",
            "bench": "both",
        },
        "metrics": [
            {
                "name": "operator.latency",
                "type": "timeseries",
                "raw_data_url": "./operator/${run_id}_operator_latency.csv",
                "unit": "ms",
            }
            # ... other metrics omitted for brevity in demo output
        ],
    }

    adapter = InfiniCoreAdapter()

    # 1. Convert
    print("--- [1] Request to InfiniCore ---")
    req = adapter.convert_to_infinicore_request(input_legacy_data)
    # print(json.dumps(req, indent=2))

    # 2. Run
    print("\n--- [2] Running Test ---")
    res = adapter.run_infinicore_test(req)

    # 3. Reconstruct
    print("\n--- [3] Final Legacy Output ---")
    final_output = adapter.convert_from_infinicore_response(res, input_legacy_data)

    # Validation
    print(json.dumps(final_output, indent=4))

    print("\n[Verification]")
    print(f"Original Time: {input_legacy_data['time']}")
    print(f"New Time:      {final_output['time']}")
    print(f"Success Status: {final_output['success']}")
    print(f"Run ID Preserved: {final_output['run_id'] == input_legacy_data['run_id']}")
