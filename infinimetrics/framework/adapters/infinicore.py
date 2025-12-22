import copy
import time
from datetime import datetime
from .base import BaseAdapter


# 确保类名完全一致
class InfiniCoreAdapter(BaseAdapter):
    def __init__(self):
        self.op_mapping = {"mul": "Mul", "add": "Add", "sub": "Sub", "div": "Div"}

    def _convert_to_request(self, legacy_json: dict) -> list:
        # Reuse the previous conversion logic
        config = legacy_json.get("config", {})
        legacy_op = config.get("operator", "").lower()
        infinicore_op = self.op_mapping.get(legacy_op, legacy_op.capitalize())

        infinicore_inputs = [
            {
                "name": i.get("name"),
                "shape": i.get("shape"),
                "dtype": i.get("dtype"),
                "strides": i.get("strides"),
            }
            for i in config.get("inputs", [])
        ]

        return [
            {
                "operator": infinicore_op,
                "device": config.get("device", "cuda").upper(),
                "args": {
                    "bench": "both",
                    "num_iterations": config.get("measured_iterations", 100),
                },
                "testcases": [
                    {
                        "description": "Adapter Generated",
                        "inputs": infinicore_inputs,
                        "result": None,  # Placeholder
                    }
                ],
            }
        ]

    def _mock_execute_backend(self, infinicore_req: list) -> list:
        # Simulate backend call
        # print(f"   [InfiniCoreAdapter] Calling underlying InfiniCore... (Simulating 200ms latency)")
        time.sleep(0.2)

        resp = copy.deepcopy(infinicore_req)
        # Simulate successful result
        resp[0]["testcases"][0]["result"] = {
            "status": {"success": True},
            "perf_ms": {"infinicore": {"device": 25.3}},
        }
        return resp

    def _convert_from_response(self, infinicore_resp: list, original_req: dict) -> dict:
        final_json = copy.deepcopy(original_req)
        final_json["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        result_data = infinicore_resp[0]["testcases"][0]["result"]
        final_json["success"] = 0 if result_data["status"]["success"] else 1

        # Fill Metrics
        val = result_data["perf_ms"]["infinicore"]["device"]
        if "metrics" in final_json:
            for m in final_json["metrics"]:
                if m["name"] == "operator.latency":
                    m["value"] = val
        return final_json

    def process(self, legacy_data: dict) -> dict:
        """
        The only public method exposed: Input Legacy -> Output Legacy
        """
        core_req = self._convert_to_request(legacy_data)
        core_resp = self._mock_execute_backend(core_req)
        final_res = self._convert_from_response(core_resp, legacy_data)
        return final_res
