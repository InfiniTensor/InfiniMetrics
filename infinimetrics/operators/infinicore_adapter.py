#!/usr/bin/env python3
"""InfiniCore Operator Adapter"""

import logging
import copy
import json
from datetime import datetime
from typing import Any, Dict, Union

from infinicore.test.framework import TestManager
from infinicore.test.framework.devices import InfiniDeviceNames

from infinimetrics.adapter import BaseAdapter

logger = logging.getLogger(__name__)


class InfiniCoreAdapter(BaseAdapter):
    """Adapter for InfiniCore operator tests (Conv, MatMul, etc.)."""

    # Metric name constants
    METRIC_LATENCY = "operator.latency"
    METRIC_ACCURACY = "operator.tensor_accuracy"
    METRIC_FLOPS = "operator.flops"
    METRIC_BANDWIDTH = "operator.bandwidth"

    # Device names from InfiniCore (uppercase for matching)
    DEVICE_NAMES = [v.upper() for v in InfiniDeviceNames.values()]

    def __init__(self):
        """Initialize adapter."""
        self.req_metrics_template = []

    def process(self, test_input: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """Execute the operator test."""
        if hasattr(test_input, "to_dict"):
            test_input = test_input.to_dict()
        elif not isinstance(test_input, dict):
            return self._create_error_response(
                test_input if isinstance(test_input, dict) else {},
                f"Invalid test_input type: {type(test_input)}"
            )

        testcase = test_input.get("testcase", "unknown")
        logger.info(f"InfiniCoreAdapter: Processing {testcase}")

        try:
            core_req = self._convert_to_request(test_input)
            core_resp = self._execute_backend(core_req)
            result = self._convert_from_response(core_resp, test_input)
            return result
        except Exception as e:
            logger.error(f"InfiniCoreAdapter: Error processing {testcase}: {e}")
            import traceback
            traceback.print_exc()
            return self._create_error_response(test_input, str(e))

    def _create_error_response(self, test_input: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """Create error response with full context."""
        return {
            "result_code": 1,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error_msg": error_msg,
            "metrics": [],
            "run_id": test_input.get("run_id", ""),
            "testcase": test_input.get("testcase", ""),
            "config": test_input.get("config", {}),
        }

    def _convert_to_request(self, legacy_json: dict) -> list:
        """Convert legacy JSON format to InfiniCore request format."""
        config = legacy_json.get("config", {})
        self.req_metrics_template = legacy_json.get("metrics", [])

        operator = config.get("operator", "").capitalize()
        infinicore_op = config.get("infinicore_op", f"infinicore.{config.get('operator', '')}")
        torch_op = config.get("torch_op", f"torch.{config.get('operator', '')}")

        run_args = self._parse_runtime_args(config)
        device_str = config.get("device", "NVIDIA").upper() if config.get("device") else "CPU"

        # Set device flags
        for device in self.DEVICE_NAMES:
            run_args[device.lower()] = device in device_str

        # Build inputs
        infinicore_inputs = [
            {k: inp[k] for k in ("name", "shape", "dtype") if k in inp}
            for inp in config.get("inputs", [])
        ]
        for inp, spec in zip(config.get("inputs", []), infinicore_inputs):
            if "strides" in inp:
                spec["strides"] = inp["strides"]

        # Build kwargs
        infinicore_kwargs = {attr["name"]: attr["value"] for attr in config.get("attributes", [])}

        outputs = config.get("outputs", [])
        if outputs:
            out_cfg = outputs[0]
            if "inplace" in out_cfg:
                infinicore_kwargs["out"] = out_cfg["inplace"]
            else:
                arg_name = out_cfg.get("arg_name", "out")
                infinicore_kwargs[arg_name] = {k: out_cfg[k] for k in ("name", "shape", "dtype") if k in out_cfg}
                if "strides" in out_cfg:
                    infinicore_kwargs[arg_name]["strides"] = out_cfg["strides"]

        if "op_kwargs" in config:
            infinicore_kwargs.update(config["op_kwargs"])

        return [{
            "operator": operator,
            "device": device_str,
            "torch_op": torch_op,
            "infinicore_op": infinicore_op,
            "args": run_args,
            "testcases": [{
                "description": f"Auto-Gen: {operator}",
                "inputs": infinicore_inputs,
                "kwargs": infinicore_kwargs,
                "tolerance": config.get("tolerance", {"atol": 1e-3, "rtol": 1e-3}),
                "result": None
            }]
        }]

    def _parse_runtime_args(self, config: dict) -> dict:
        """Parse runtime arguments from config."""
        args = {
            "bench": "both",
            "num_prerun": config.get("warmup_iterations", 5),
            "num_iterations": config.get("measured_iterations", 100),
            "verbose": False,
            "debug": False,
            "eq_nan": False,
            "save": "test_report.json",
            **{device.lower(): False for device in self.DEVICE_NAMES}
        }
        if "backend_args" in config:
            args.update(config["backend_args"])
        return args

    def _handle_latency(self, metric: dict, context: dict):
        """Handle latency metric."""
        if context.get('latency_ms') is not None:
            metric.update({
                "value": context['latency_ms'],
                "type": "scalar",
                "raw_data_url": ""
            })

    def _handle_accuracy(self, metric: dict, context: dict):
        """Handle accuracy metric (mock)."""
        metric["value"] = "PASS"

    def _handle_mock_metric(self, metric: dict, context: dict):
        """Handle mock metrics (FLOPS, bandwidth)."""
        metric.update({
            "value": 0.0,
            "type": "scalar",
            "raw_data_url": ""
        })

    def _convert_from_response(self, saved_files: list, original_req: dict) -> dict:
        """Convert InfiniCore saved files to legacy format."""
        final_json = copy.deepcopy(original_req)
        final_json["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            if not saved_files:
                raise ValueError("No saved files returned from TestManager")

            with open(saved_files[0], 'r') as f:
                saved_data = json.load(f)

            tc_result = saved_data[0]["testcases"][0]["result"]
            is_success = tc_result.get("status", {}).get("success", False)
            final_json["result_code"] = 0 if is_success else 1

            if not is_success:
                final_json["error_msg"] = tc_result.get("status", {}).get("error", "Unknown")
                return final_json

            perf_data = tc_result.get("perf_ms", {}).get("infinicore", {})
            context = {
                'latency_ms': perf_data.get("device"),
                'tflops': tc_result.get("metrics", {}).get("tflops"),
                'bandwidth_gbs': tc_result.get("metrics", {}).get("bandwidth_gbs")
            }

            metric_handlers = {
                self.METRIC_LATENCY: self._handle_latency,
                self.METRIC_ACCURACY: self._handle_accuracy,
                self.METRIC_FLOPS: self._handle_mock_metric,
                self.METRIC_BANDWIDTH: self._handle_mock_metric,
            }

            if "metrics" in final_json and self.req_metrics_template:
                for i, metric in enumerate(final_json["metrics"][:len(self.req_metrics_template)]):
                    handler = metric_handlers.get(metric.get("name"))
                    if handler:
                        handler(metric, context)

        except Exception as e:
            logger.error(f"[Adapter] Parsing Error: {e}")
            final_json["result_code"] = 1
            final_json["error_msg"] = str(e)

        return final_json

    def _execute_backend(self, infinicore_req: list) -> list:
        """Execute InfiniCore backend API using TestManager."""
        test_manager = TestManager(verbose=False, bench_mode=True)
        all_passed, saved_files = test_manager.test(
            target_ops=None,
            json_cases_list=infinicore_req,
            global_exec_args=None
        )
        return saved_files
