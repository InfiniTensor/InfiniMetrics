import copy
import time
from datetime import datetime
from .base import BaseAdapter


# 确保类名完全一致
class InfiniCoreAdapter(BaseAdapter):
    def __init__(self):

    def _convert_to_request(self, legacy_json: dict) -> list:
        config = legacy_json.get("config", {})
        
        # 缓存 Metrics 模板
        self.req_metrics_template = legacy_json.get("metrics", [])

        # 1. 算子映射
        legacy_op = config.get("operator", "").lower()
        infinicore_op = legacy_op.capitalize()

        # 2. 运行时参数解析
        run_args = self._parse_runtime_args(config)

        # 3. Inputs 解析
        infinicore_inputs = []
        for inp in config.get("inputs", []):
            input_spec = {
                "name": inp.get("name"),
                "shape": inp.get("shape"),
                "dtype": self._parse_dtype(inp.get("dtype", "float32")),
                "strides": inp.get("strides"),
            }
            if "init" in inp: input_spec["init"] = inp["init"]
            if "value" in inp: input_spec["init"] = {"type": "constant", "value": inp["value"]}
            if inp.get("requires_grad"): input_spec["requires_grad"] = True
            infinicore_inputs.append(input_spec)

        # 4. Kwargs / Outputs / Attributes 解析
        infinicore_kwargs = {}
        
        # [补丁 1] 解析 attributes 列表 (关键！处理 Conv 的 pads/strides)
        # JSON 格式: "attributes": [{"name": "pads", "value": [...]}, ...]
        for attr in config.get("attributes", []):
            infinicore_kwargs[attr["name"]] = attr["value"]

        # 处理 Outputs
        outputs = config.get("outputs", [])
        if outputs:
            out_cfg = outputs[0]
            if "inplace" in out_cfg:
                # 直接透传 inplace 的目标变量名 (例如 "A")
                infinicore_kwargs["out"] = out_cfg["inplace"]
            else:
                arg_name = out_cfg.get("arg_name", "out")
                infinicore_kwargs[arg_name] = {
                    "name": out_cfg.get("name"),
                    "shape": out_cfg.get("shape"),
                    "dtype": self._parse_dtype(out_cfg.get("dtype", "float32"))
                }
        
        # 允许 op_kwargs 覆盖 attributes
        if "op_kwargs" in config:
            infinicore_kwargs.update(config["op_kwargs"])

        # 5. 组装
        return [{
            "operator": infinicore_op,
            "device": self._parse_device(config.get("device", "cuda")),
            "args": run_args,
            "testcases": [{
                "description": f"Auto-Gen: {infinicore_op}",
                "inputs": infinicore_inputs,
                "kwargs": infinicore_kwargs,
                "tolerance": config.get("tolerance", {"atol": 1e-3, "rtol": 1e-3}),
                "result": None
            }]
        }]

    def _mock_execute_backend(self, infinicore_req: list) -> list:
        time.sleep(0.1)
        resp = copy.deepcopy(infinicore_req)
        resp[0]["testcases"][0]["result"] = {
            "status": {"success": True},
            "perf_ms": {"infinicore": {"device": 0.045}}, 
            "tflops": 128.5,           # 对应 operator.flops
            "bandwidth_gb_s": 900.2    # 对应 operator.bandwidth
        }
        return resp

    def _convert_from_response(self, infinicore_resp: list, original_req: dict) -> dict:
        final_json = copy.deepcopy(original_req)
        final_json["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            result_data = infinicore_resp[0]["testcases"][0]["result"]
            
            # 状态处理
            is_success = result_data.get("status", {}).get("success", False)
            final_json["success"] = 0 if is_success else 1
            if not is_success:
                final_json["error_msg"] = result_data.get("status", {}).get("error", "Unknown")
                return final_json

            # Metrics 动态回填
            if "metrics" in final_json and self.req_metrics_template:
                filled_metrics = []
                for metric_template in self.req_metrics_template:
                    metric = copy.deepcopy(metric_template)
                    name = metric.get("name")
                    
                    # 1. Latency
                    if name == "operator.latency":
                        val = result_data.get("perf_ms", {}).get("infinicore", {}).get("device")
                        if val is not None: metric["value"] = val
                    
                    # 2. Accuracy
                    elif name == "operator.tensor_accuracy":
                        metric["value"] = "PASS" if is_success else "FAIL"
                    
                    # 3. [补丁 2] FLOPS 支持
                    elif name == "operator.flops":
                        # 假设后端返回字段叫 tflops 或 flops
                        val = result_data.get("tflops") or result_data.get("flops")
                        if val is not None: metric["value"] = val
                            
                    # 4. [补丁 3] Bandwidth 支持
                    elif name == "operator.bandwidth":
                        # 假设后端返回字段叫 bandwidth_gb_s
                        val = result_data.get("bandwidth_gb_s") or result_data.get("bandwidth")
                        if val is not None: metric["value"] = val
                            
                    filled_metrics.append(metric)
                
                final_json["metrics"] = filled_metrics
                
        except Exception as e:
            print(f"[Adapter] Parsing Error: {e}")
            final_json["success"] = 1
            final_json["error_msg"] = str(e)

        return final_json

    def _parse_runtime_args(self, config: dict) -> dict:
        """
        解析运行时参数：合并默认值、历史字段(measured_iterations)和通用配置
        """
        # 1. 基础默认值
        args = {
            "bench": "both",
            "num_prerun": 5,
            "num_iterations": 100,
            "verbose": False,
            "seed": 42  # 默认随机种子
        }
        
        # 2. 映射历史字段 (兼容旧的 add.json 写法)
        if "warmup_iterations" in config:
            args["num_prerun"] = int(config["warmup_iterations"])
        if "measured_iterations" in config:
            args["num_iterations"] = int(config["measured_iterations"])
            
        # 3. 映射高级配置
        if "seed" in config:
            args["seed"] = int(config["seed"])
        
        # 4. 如果显式指定了 backend_args，进行覆盖
        if "backend_args" in config:
            args.update(config["backend_args"])
            
        return args
        
    def process(self, legacy_data: dict) -> dict:
        """
        The only public method exposed: Input Legacy -> Output Legacy
        """
        core_req = self._convert_to_request(legacy_data)
        core_resp = self._mock_execute_backend(core_req)
        final_res = self._convert_from_response(core_resp, legacy_data)
        return final_res
