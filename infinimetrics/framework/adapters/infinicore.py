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

    # =========================================================================
    # 新增辅助方法：计算理论数据量和计算量
    # =========================================================================
    def _get_dtype_bytes(self, dtype_str: str) -> int:
        """简单的 dtype 字节数映射"""
        d = dtype_str.lower()
        if "float32" in d or "int32" in d: return 4
        if "float16" in d or "bfloat16" in d or "int16" in d: return 2
        if "int8" in d or "uint8" in d or "bool" in d: return 1
        if "float64" in d or "int64" in d: return 8
        return 4 # 默认按 4 字节算

    def _estimate_workload(self, config: dict) -> tuple[float, float]:
        """
        根据 config 解析输入输出，估算:
        1. total_bytes (用于计算 Bandwidth)
        2. total_flops (用于计算 FLOPS)
        
        返回: (total_bytes, total_flops)
        """
        total_bytes = 0.0
        total_flops = 0.0
        
        # 1. 计算带宽数据量 (Sum of Inputs + Outputs)
        # -------------------------------------------------
        tensors_to_count = config.get("inputs", []) + config.get("outputs", [])
        
        for tensor in tensors_to_count:
            shape = tensor.get("shape", [])
            dtype = tensor.get("dtype", "float32")
            
            # 计算元素个数 (volume)
            if shape:
                volume = 1
                for dim in shape:
                    volume *= dim
            else:
                volume = 0
                
            # 累加字节数
            total_bytes += volume * self._get_dtype_bytes(dtype)

        # 2. 估算 FLOPS (根据算子类型使用不同公式)
        # -------------------------------------------------
        op_type = config.get("operator", "").lower()
        
        # [示例] Matmul: 2 * M * N * K
        if op_type == "matmul":
            try:
                inputs = config.get("inputs", [])
                # 假设 Input A=[M, K], B=[K, N] (简化处理，未考虑Batch)
                shape_a = inputs[0]["shape"]
                shape_b = inputs[1]["shape"]
                
                M = shape_a[-2]
                K = shape_a[-1]
                N = shape_b[-1]
                
                total_flops = 2.0 * M * N * K
            except:
                total_flops = 0.0
        
        # [示例] Conv: 2 * Batch * OutCh * OutH * OutW * InCh * KH * KW
        elif op_type == "conv":
            # 这里的计算比较复杂，需要结合 attributes (stride, padding) 推导
            # 为了演示，这里先留空，或者你可以写一个简化估算
            # 如果你有 attributes 中的 kernel_shape 等信息，可以在这里算
            pass 
        
        # [兜底] 如果没实现特定算子公式，FLOPS 设为 0 (避免除以零报错)
        
        return total_bytes, total_flops

    # =========================================================================
    # 修改后的核心方法
    # =========================================================================
    def _convert_from_response(self, infinicore_resp: list, original_req: dict) -> dict:
        final_json = copy.deepcopy(original_req)
        final_json["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            # 1. 提取基础数据
            testcase_res = infinicore_resp[0]["testcases"][0]
            result_data = testcase_res["result"]
            config = original_req.get("config", {})

            # 2. 状态处理
            is_success = result_data.get("status", {}).get("success", False)
            final_json["success"] = 0 if is_success else 1
            if not is_success:
                final_json["error_msg"] = result_data.get("status", {}).get("error", "Unknown")
                # 失败时直接返回，不再计算指标
                return final_json

            # 3. 准备计算基础值
            latency_ms = result_data.get("perf_ms", {}).get("infinicore", {}).get("device")
            
            # 只有当 Latency 有效且大于0时才进行计算
            if latency_ms and latency_ms > 0:
                latency_sec = latency_ms / 1000.0
                
                # 调用上面的辅助方法获取理论工作量
                total_bytes, total_flops = self._estimate_workload(config)
                
                # 计算结果
                # GB/s = Bytes / Seconds / 10^9
                bandwidth_gbs = (total_bytes / latency_sec) / 1e9
                # TFLOPS = Flops / Seconds / 10^12
                tflops = (total_flops / latency_sec) / 1e12
            else:
                bandwidth_gbs = 0.0
                tflops = 0.0

            # 4. Metrics 动态回填
            if "metrics" in final_json and self.req_metrics_template:
                filled_metrics = []
                for metric_template in self.req_metrics_template:
                    metric = copy.deepcopy(metric_template)
                    name = metric.get("name")
                    
                    # --- 1. Latency (直接获取) ---
                    if name == "operator.latency":
                        if latency_ms is not None:
                            metric["value"] = latency_ms
                    
                    # --- 2. Accuracy (Mock) ---
                    elif name == "operator.tensor_accuracy":
                        # 只要 success=0 就认为是 PASS
                        metric["value"] = "PASS" if is_success else "FAIL"
                        # 如果需要 Mock 文件路径，这里已经保留了 original_req 里的 reference/candidate 结构
                    
                    # --- 3. FLOPS (计算得出) ---
                    elif name == "operator.flops":
                        # 只有当算子公式已实现且 Latency 有效时才填
                        if tflops > 0:
                            metric["value"] = round(tflops, 4)
                        else:
                            # 如果没算出，给个 0 或者 null，或者不填 value
                            metric["value"] = 0.0
                            
                    # --- 4. Bandwidth (计算得出) ---
                    elif name == "operator.bandwidth":
                        if bandwidth_gbs > 0:
                            metric["value"] = round(bandwidth_gbs, 4)
                        else:
                            metric["value"] = 0.0
                            
                    filled_metrics.append(metric)
                
                final_json["metrics"] = filled_metrics
                
        except Exception as e:
            print(f"[Adapter] Parsing Error: {e}")
            final_json["success"] = 1
            final_json["error_msg"] = str(e)
            import traceback
            traceback.print_exc()

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
