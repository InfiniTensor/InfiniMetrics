#!/usr/bin/env python3
"""
Executor - Universal Test Execution Framework
"""

import logging
import json
import subprocess
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

from infinimetrics.adapter import BaseAdapter
from infinimetrics.input import TestInput

logger = logging.getLogger(__name__)

NVIDIA_SMI_GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,memory.total,driver_version",
    "--format=csv,noheader",
]

def _which(cmd: str) -> Optional[str]:
    try:
        from shutil import which
        return which(cmd)
    except Exception:
        return None

@dataclass
class TestResult:
    """
    Standardized test result structure.

    Used throughout the execution lifecycle and returned to Dispatcher.

    Note:
        result_code: 0 = success, non-zero = error code (following Linux convention)
    """

    run_id: str
    testcase: str
    result_code: int  # 0 = success, non-zero = error code
    result_file: Optional[str] = None
    skipped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to lightweight dictionary format for Dispatcher aggregation."""
        return {
            "run_id": self.run_id,
            "testcase": self.testcase,
            "result_code": self.result_code,
            "result_file": self.result_file,
            "skipped": self.skipped,
        }


class Executor:
    """
    Universal test executor for all test types.

    Responsibilities:
        1. Manage adapter lifecycle (setup -> process -> teardown)
        2. Save results to disk
        3. Return result summary
    """

    def __init__(self, payload: Dict[str, Any], adapter: BaseAdapter):
        """
        Initialize executor.

        Args:
            payload: Test payload with testcase, config, etc.
            adapter: Configured adapter instance
        """
        self.payload = payload
        self.adapter = adapter
        self.testcase = payload.get("testcase", "unknown")
        self.run_id = payload.get("run_id", "")
        self.test_input = None

        # Setup output directory from config
        config = payload.get("config", {})
        output_dir = config.get("output_dir", "./output")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Executor initialized: testcase={self.testcase}")

    def setup(self) -> None:
        """
        Setup phase - initialize adapter.

        This should be called before execute().
        """
        config = self.payload.get("config", {})

        # Convert payload to TestInput object
        self.test_input = TestInput.from_dict(self.payload)

        self.adapter.setup(config)

        logger.debug(f"Executor: Setup complete for {self.testcase}")

    def teardown(self, result: Any) -> str:
        """
        Teardown phase - cleanup adapter, collect metrics, and save results.

        This should be called after process() completes.

        Args:
            result:

        Returns:
            Path to saved result file
        """
        # Always cleanup adapter
        try:
            self.adapter.teardown()
        except Exception as teardown_error:
            logger.warning(
                f"Executor: Teardown failed for {self.testcase}: {teardown_error}"
            )

        # TODO: Add metrics calculation method

        # Save result to disk
        result_file = self._save_result(result)

        logger.debug(f"Executor: Teardown complete for {self.testcase}")
        return result_file

    def execute(self) -> TestResult:
        """
        Execute the complete test with proper lifecycle management.

        Lifecycle:
            1. adapter.setup(config)
            2. adapter.process(payload)
            3. adapter.teardown() - includes saving results
            4. Return TestResult

        Returns:
            TestResult object with result_code and file path.
        """
        logger.info(f"Executor: Running {self.testcase}")

        # Initialize TestResult directly (default: result_code=0)
        test_result = TestResult(
            run_id=self.run_id,
            testcase=self.testcase,
            result_code=0,  # Default to success
            result_file=None,
        )

        try:
            # Phase 1: Setup
            self.setup()

            # Phase 2: Process
            logger.debug(f"Executor: Calling adapter.process()")
            response = self.adapter.process(self.test_input)

            # Process response (0 = success, non-zero = error code)
            test_result.result_code = int(response.get("result_code", 1)) if isinstance(response, dict) else 1
            if test_result.result_code != 0:
                logger.warning(f"Executor: Adapter failed with error code {test_result.result_code}")
            
            # Enrich environment ONLY if missing
            if isinstance(response, dict) and "environment" not in response:
                env = self._build_environment(response)

                # rebuild ordered dict (py3.7+ preserves insertion order)
                ordered = {}
                for k in ["run_id", "time", "testcase", "success"]:
                    if k in response:
                        ordered[k] = response[k]
                ordered["environment"] = env

                # append remaining keys in original order (skip those already set)
                for k, v in response.items():
                    if k not in ordered:
                        ordered[k] = v

                response = ordered

            # Phase 3: Teardown (cleanup, save result)
            result_file = self.teardown(response)
            test_result.result_file = result_file

            logger.info(
                f"Executor: {self.testcase} completed with code={test_result.result_code}"
            )

            return test_result

        except Exception as e:
            logger.error(f"Executor: {self.testcase} failed: {e}", exc_info=True)

            # Still run teardown on failure
            self._save_result(None)
            test_result.result_code = 1  # Failure

            return test_result
    
    def _build_environment(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a unified environment block
        """
        resolved = response.get("resolved", {}) if isinstance(response, dict) else {}
        nodes = int(resolved.get("nodes", 1) or 1)
        device_used = int(resolved.get("device_used", 0) or 0)
        gpn = int(resolved.get("gpus_per_node", 0) or 0)

        # Fallback to config hints if adapter didn't provide
        cfg = (self.payload.get("config", {}) or {})

        accel_type = (cfg.get("accelerator_type") or cfg.get("device_type") or "").strip().lower()  # optional
        device_ids = cfg.get("device_ids")

        if device_ids is None and isinstance(cfg.get("single_node"), dict):
            device_ids = cfg["single_node"].get("device_ids")

        if device_used <= 0:
            try:
                device_used = int(cfg.get("device_involved", 0) or 0)
            except Exception:
                device_used = 0

        if nodes <= 1:
            topo = f"{device_used}x1 ring mesh"
        else:
            topo = f"{nodes}x{(gpn or max(1, device_used // nodes))} ring mesh"

        hw = self._collect_static_hw(accel_type=accel_type, device_ids=device_ids)

        return {
            "cluster_scale": nodes,
            "topology": topo,
            "cluster": [
                {
                    "machine": {
                        "cpu_model": hw.get("cpu_model", "Unknown"),
                        "memory_gb": hw.get("memory_gb", 0),
                        "accelerators": [
                            {
                                "model": hw.get("gpu_model", "Unknown"),
                                "count": device_used,
                                "memory_gb_per_card": hw.get("gpu_memory_gb", 0),
                                "driver": hw.get("driver_version", "Unknown"),
                                "cuda": hw.get("cuda_version", "Unknown"),
                                # reserved: type of platform
                                "type": hw.get("accelerator_type", "generic"),
                            }
                        ],
                    },
                    "framework": [{"name": "unknown", "version": "unknown"}],
                }
            ],
        }

    def _collect_static_hw(self, accel_type: str = "", device_ids: Any = None) -> Dict[str, Any]:
        """
        Best-effort static HW collector (CPU/mem/GPU model/driver/CUDA).
        """
        hw: Dict[str, Any] = {
            "cpu_model": "Unknown",
            "memory_gb": 0,
            "gpu_model": "Unknown",
            "gpu_count": 0,
            "gpu_memory_gb": 0,
            "driver_version": "Unknown",
            "cuda_version": "Unknown",
            "accelerator_type": "generic",
        }

        # CPU
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        hw["cpu_model"] = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

        # Mem
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        mem_kb = int(line.split()[1])
                        hw["memory_gb"] = mem_kb // (1024 * 1024)
                        break
        except Exception:
            pass
        
        hint = (accel_type or "").lower().strip()

        probes: List[str] = []
        if hint in ("nvidia", "amd", "ascend", "cambricon", "generic"):
            probes = [hint]
        else:
            probes = []

        # add auto-detect order
        if "nvidia" not in probes:
            probes.append("nvidia")
        if "amd" not in probes:
            probes.append("amd")
        if "ascend" not in probes:
            probes.append("ascend")
        if "cambricon" not in probes:
            probes.append("cambricon")
        if "generic" not in probes:
            probes.append("generic")
        
        for p in probes:
            if p == "nvidia" and self._probe_nvidia(hw):
                hw["accelerator_type"] = "nvidia"
                hw["cuda_version"] = self._collect_cuda_version() or hw["cuda_version"]
                return hw

            if p == "amd" and self._probe_amd(hw):
                hw["accelerator_type"] = "amd"
                return hw

            if p == "ascend" and self._probe_ascend(hw):
                hw["accelerator_type"] = "ascend"
                return hw

            if p == "cambricon" and self._probe_cambricon(hw):
                hw["accelerator_type"] = "cambricon"
                return hw

            if p == "generic":
                hw["accelerator_type"] = "generic"
                return hw

        return hw
    def _probe_nvidia(self, hw: Dict[str, Any]) -> bool:
        try:
            r = subprocess.run(NVIDIA_SMI_GPU_QUERY, capture_output=True, text=True, timeout=5)
            if r.returncode != 0 or not r.stdout.strip():
                return False

            lines = [x.strip() for x in r.stdout.strip().splitlines() if x.strip()]
            hw["gpu_count"] = len(lines)

            p = [x.strip() for x in lines[0].split(",")]
            if len(p) >= 3:
                hw["gpu_model"] = p[0]
                hw["driver_version"] = p[2]
                mm = re.search(r"(\d+)\s*MiB", p[1])
                if mm:
                    hw["gpu_memory_gb"] = int(mm.group(1)) // 1024
            return True
        except Exception:
            return False


    def _probe_amd(self, hw: Dict[str, Any]) -> bool:
        """
        Try amd-smi or rocm-smi (best-effort).
        We only fill model/count if possible; otherwise return False.
        """
        try:
            tool = None
            for c in AMD_SMI_CANDIDATES:
                if _which(c):
                    tool = c
                    break
            if not tool:
                return False

            # Try a light command. Different environments output differently.
            # amd-smi: `amd-smi list` ; rocm-smi: `rocm-smi -i`
            if tool == "amd-smi":
                cmd = ["amd-smi", "list"]
            else:
                cmd = ["rocm-smi", "-i"]

            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if r.returncode != 0 or not r.stdout.strip():
                return False

            # Minimal parse: count devices by "GPU" markers
            txt = r.stdout
            # heuristic: count lines containing "GPU" and an index
            lines = [x for x in txt.splitlines() if re.search(r"\bGPU\b", x, re.IGNORECASE)]
            hw["gpu_count"] = max(hw["gpu_count"], len(lines)) if lines else hw["gpu_count"]
            hw["gpu_model"] = hw["gpu_model"] if hw["gpu_model"] != "Unknown" else "AMD GPU"
            return True
        except Exception:
            return False


    def _probe_ascend(self, hw: Dict[str, Any]) -> bool:
        """
        Ascend: best-effort using npu-smi if present.
        """
        try:
            if not _which("npu-smi"):
                return False
            r = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=5)
            if r.returncode != 0 or not r.stdout.strip():
                return False

            txt = r.stdout
            # heuristic: count device lines with "NPU" or "Device"
            cnt = len([x for x in txt.splitlines() if re.search(r"\bNPU\b|\bDevice\b", x)])
            hw["gpu_count"] = max(hw["gpu_count"], cnt) if cnt else hw["gpu_count"]
            hw["gpu_model"] = hw["gpu_model"] if hw["gpu_model"] != "Unknown" else "Ascend NPU"
            return True
        except Exception:
            return False


    def _probe_cambricon(self, hw: Dict[str, Any]) -> bool:
        """
        Cambricon: best-effort using cnmon if present.
        """
        try:
            if not _which("cnmon"):
                return False
            r = subprocess.run(["cnmon", "info"], capture_output=True, text=True, timeout=5)
            if r.returncode != 0 or not r.stdout.strip():
                return False

            txt = r.stdout
            cnt = len([x for x in txt.splitlines() if re.search(r"\bMLU\b|\bDevice\b", x)])
            hw["gpu_count"] = max(hw["gpu_count"], cnt) if cnt else hw["gpu_count"]
            hw["gpu_model"] = hw["gpu_model"] if hw["gpu_model"] != "Unknown" else "Cambricon MLU"
            return True
        except Exception:
            return False

    def _collect_cuda_version(self) -> Optional[str]:
        try:
            r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=2)
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    if "release" in line:
                        m = re.search(r"release\s+(\d+\.\d+)", line)
                        if m:
                            return m.group(1)
        except Exception:
            pass
        return None

    def _save_result(self, result: Dict[str, Any]) -> str:
        """
        Save detailed result to disk as JSON.

        Args:
            result: Complete result dict with data and metrics

        Returns:
            Absolute path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = self.testcase.replace(".", "_").replace("/", "_")
        filename = f"{safe_name}_{timestamp}_results.json"
        output_file = self.output_dir / filename

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.debug(f"Executor: Results saved to {output_file}")
        return str(output_file)
