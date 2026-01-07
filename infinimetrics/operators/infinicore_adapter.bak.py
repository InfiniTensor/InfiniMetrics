#!/usr/bin/env python3
"""
InfiniCore Operator Adapter
"""

import copy
import time
import logging
from datetime import datetime
from typing import Any, Dict, Optional, List

from infinimetrics.adapter import BaseAdapter


logger = logging.getLogger(__name__)


class InfiniDtype:
    """InfiniCore dtype constants"""

    float16 = "float16"
    float32 = "float32"
    bfloat16 = "bfloat16"
    int8 = "int8"
    int32 = "int32"
    int64 = "int64"
    bool = "bool"


class InfiniCoreAdapter(BaseAdapter):
    """
    Adapter for InfiniCore operator tests (Conv, MatMul, etc.)

    Converts legacy operator test requests to InfiniCore format,
    executes tests, and converts responses back to legacy format.
    """

    # Metric name constants
    METRIC_LATENCY = "operator.latency"
    METRIC_ACCURACY = "operator.tensor_accuracy"
    METRIC_FLOPS = "operator.flops"
    METRIC_BANDWIDTH = "operator.bandwidth"

    def __init__(self):
        """Initialize InfiniCore adapter"""
        self.req_metrics_template = []
        logger.info("InfiniCoreAdapter initialized")

    def process(self, test_input: Any) -> Dict[str, Any]:
        """
        Execute the operator test.

        Args:
            test_input: TestInput object or dict with testcase, config, metrics, etc.

        Returns:
            Dict with:
                - 'success': int (0 = success, non-zero = failure)
                - 'metrics': list (performance metrics)
                - 'time': str (timestamp)
                - 'error_msg': str (optional, if failed)
        """
        # Convert TestInput object to dict if needed
        if hasattr(test_input, "to_dict"):
            test_input_dict = test_input.to_dict()
        elif isinstance(test_input, dict):
            test_input_dict = test_input
        else:
            return {
                "success": 1,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error_msg": f"Invalid test_input type: {type(test_input)}",
                "metrics": [],
            }

        testcase = test_input_dict.get("testcase", "unknown")
        logger.info(f"InfiniCoreAdapter: Processing {testcase}")

        try:
            # Convert request to InfiniCore format
            core_req = self._convert_to_request(test_input_dict)

            # Execute test (currently mocked)
            core_resp = self._mock_execute_backend(core_req)

            # Convert response back to legacy format
            return self._convert_from_response(core_resp, test_input_dict)

        except Exception as e:
            logger.error(f"InfiniCoreAdapter: {testcase} failed: {e}", exc_info=True)
            return {
                "success": 1,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error_msg": str(e),
                "metrics": [],
            }

    # =========================================================================
    # Request Conversion
    # =========================================================================

    def _parse_dtype(self, dtype_str: str) -> str:
        """Parse dtype string to InfiniCore format"""
        return getattr(InfiniDtype, dtype_str.lower(), InfiniDtype.float32)

    def _parse_device(self, device_str: str) -> str:
        """Parse device string to uppercase format"""
        return device_str.upper() if device_str else "CPU"

    def _parse_runtime_args(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Parse runtime arguments from config"""
        args = {
            "bench": "both",
            "num_prerun": 5,
            "num_iterations": 100,
            "verbose": False,
            "debug": False,
        }
        if "warmup_iterations" in config:
            args["num_prerun"] = int(config["warmup_iterations"])
        if "measured_iterations" in config:
            args["num_iterations"] = int(config["measured_iterations"])
        if "backend_args" in config:
            args.update(config["backend_args"])
        return args

    def _convert_to_request(self, test_input: Dict[str, Any]) -> list:
        """
        Convert test input to InfiniCore request format

        Args:
            test_input: Test input dict

        Returns:
            List of InfiniCore request objects
        """
        config = test_input.get("config", {})
        self.req_metrics_template = test_input.get("metrics", [])

        # 1. Operator mapping
        legacy_op = config.get("operator", "").lower()
        infinicore_op = legacy_op.capitalize()

        # 2. Runtime args
        run_args = self._parse_runtime_args(config)

        # 3. Inputs
        infinicore_inputs = []
        for inp in config.get("inputs", []):
            input_spec = {
                "name": inp.get("name"),
                "shape": inp.get("shape"),
                "dtype": self._parse_dtype(inp.get("dtype", "float32")),
                "strides": inp.get("strides"),
            }
            infinicore_inputs.append(input_spec)

        # 4. Kwargs construction
        infinicore_kwargs = {}

        # Attribute parsing
        for attr in config.get("attributes", []):
            infinicore_kwargs[attr["name"]] = attr["value"]

        # Output parsing
        outputs = config.get("outputs", [])
        if outputs:
            out_cfg = outputs[0]
            if "inplace" in out_cfg:
                infinicore_kwargs["out"] = out_cfg["inplace"]
            else:
                arg_name = out_cfg.get("arg_name", "out")
                infinicore_kwargs[arg_name] = {
                    "name": out_cfg.get("name"),
                    "shape": out_cfg.get("shape"),
                    "dtype": self._parse_dtype(out_cfg.get("dtype", "float32")),
                    "strides": out_cfg.get("strides"),
                }

        # Override with op_kwargs
        if "op_kwargs" in config:
            infinicore_kwargs.update(config["op_kwargs"])

        return [
            {
                "operator": infinicore_op,
                "device": self._parse_device(config.get("device", "NVIDIA")),
                "args": run_args,
                "testcases": [
                    {
                        "description": f"Auto-Gen: {infinicore_op}",
                        "inputs": infinicore_inputs,
                        "kwargs": infinicore_kwargs,
                        "tolerance": config.get(
                            "tolerance", {"atol": 1e-3, "rtol": 1e-3}
                        ),
                        "result": None,
                    }
                ],
            }
        ]

    # =========================================================================
    # Response Conversion
    # =========================================================================

    def _fill_scalar_metric(self, metric: Dict[str, Any], value: float):
        """Helper to format scalar metrics"""
        metric["value"] = round(value, 4) if value > 0 else 0.0
        metric["type"] = "scalar"
        metric["raw_data_url"] = ""

    def _handle_latency(self, metric: Dict[str, Any], context: Dict[str, Any]):
        """Handle latency metric"""
        if context["latency_ms"] is not None:
            metric["value"] = context["latency_ms"]
            metric["type"] = "scalar"
            metric["raw_data_url"] = ""

    def _handle_accuracy(self, metric: Dict[str, Any], context: Dict[str, Any]):
        """Handle accuracy metric"""
        metric["value"] = "PASS" if context["success"] else "FAIL"

    def _handle_flops(self, metric: Dict[str, Any], context: Dict[str, Any]):
        """Handle FLOPS metric (mocked)"""
        # TODO: Implement real TFLOPS calculation
        self._fill_scalar_metric(metric, context["tflops"])

    def _handle_bandwidth(self, metric: Dict[str, Any], context: Dict[str, Any]):
        """Handle bandwidth metric (mocked)"""
        # TODO: Implement real bandwidth calculation
        self._fill_scalar_metric(metric, context["bandwidth"])

    def _convert_from_response(
        self, infinicore_resp: list, original_req: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert InfiniCore response to legacy format

        Args:
            infinicore_resp: Response from InfiniCore backend
            original_req: Original test input

        Returns:
            Legacy format response dict
        """
        final_json = copy.deepcopy(original_req)
        final_json["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            # 1. Parse result
            tc_result = infinicore_resp[0]["testcases"][0]["result"]
            config = original_req.get("config", {})

            is_success = tc_result.get("status", {}).get("success", False)
            final_json["success"] = 0 if is_success else 1

            if not is_success:
                final_json["error_msg"] = tc_result.get("status", {}).get(
                    "error", "Unknown"
                )
                return final_json

            # 2. Calculate workload metrics (mocked for now)
            latency_ms = (
                tc_result.get("perf_ms", {}).get("infinicore", {}).get("device")
            )
            bandwidth_gbs = 0.0  # Mock value
            tflops = 0.0  # Mock value

            # TODO: Uncomment when workload estimation is implemented
            # if latency_ms and latency_ms > 0:
            #     latency_sec = latency_ms / 1000.0
            #     total_bytes, total_flops = self._estimate_workload(config)
            #     bandwidth_gbs = (total_bytes / latency_sec) / 1e9
            #     tflops = (total_flops / latency_sec) / 1e12

            # 3. Metrics dispatcher map
            metric_handlers = {
                self.METRIC_LATENCY: self._handle_latency,
                self.METRIC_ACCURACY: self._handle_accuracy,
                self.METRIC_FLOPS: self._handle_flops,
                self.METRIC_BANDWIDTH: self._handle_bandwidth,
            }

            # Context object to pass data to handlers
            context = {
                "success": is_success,
                "latency_ms": latency_ms,
                "tflops": tflops,
                "bandwidth": bandwidth_gbs,
            }

            # Fill metrics
            if "metrics" in final_json and self.req_metrics_template:
                for i, metric_template in enumerate(self.req_metrics_template):
                    if i < len(final_json["metrics"]):
                        metric = final_json["metrics"][i]
                        name = metric.get("name")

                        handler = metric_handlers.get(name)
                        if handler:
                            handler(metric, context)

        except Exception as e:
            logger.error(f"Error parsing response: {e}", exc_info=True)
            final_json["success"] = 1
            final_json["error_msg"] = str(e)

        return final_json

    # =========================================================================
    # Mock Backend Execution
    # =========================================================================

    def _mock_execute_backend(self, infinicore_req: list) -> list:
        """
        Mock InfiniCore backend execution

        TODO: Replace with real backend call
        """
        time.sleep(0.01)  # Simulate some processing time
        resp = copy.deepcopy(infinicore_req)
        resp[0]["testcases"][0]["result"] = {
            "status": {"success": True, "error": ""},
            "perf_ms": {
                "torch": {"host": 6.1, "device": 77.5},
                "infinicore": {"host": 13.1, "device": 19.2},
            },
        }
        return resp
