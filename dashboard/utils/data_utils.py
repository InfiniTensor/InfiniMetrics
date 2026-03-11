#!/usr/bin/env python3
"""Data utility functions for InfiniMetrics dashboard."""
import json
import logging

from pathlib import Path
from typing import Any, Dict, List


def load_summary_file(summary_path: str = "../summary_output") -> List[Dict[str, Any]]:
    """Load dispatcher summary files."""

    logger = logging.getLogger(__name__)
    summaries = []
    summary_dir = Path(summary_path)

    if summary_dir.exists():
        for json_file in sorted(
            summary_dir.glob("dispatcher_summary_*.json"), reverse=True
        ):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["file"] = json_file.name
                data["timestamp"] = json_file.stem.replace("dispatcher_summary_", "")
                summaries.append(data)
            except Exception as e:
                logger.warning(f"Failed to load summary {json_file}: {e}")

    return summaries


def get_friendly_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def extract_accelerator_types(result_json: dict) -> list[str]:
    """Extract the accelerator card type from result_json."""
    types = set()
    try:
        clusters = result_json.get("environment", {}).get("cluster", [])
        for node in clusters:
            accs = node.get("machine", {}).get("accelerators", [])
            for acc in accs:
                if "type" in acc:
                    types.add(acc["type"])
    except Exception:
        pass
    return list(types)


def extract_test_type(testcase: str) -> str:
    """Extract test type from testcase string (e.g., 'comm.nccl.allreduce' -> 'comm')."""
    parts = testcase.split(".")
    return parts[0] if parts else "unknown"


def extract_operation(testcase: str) -> str:
    """Extract operation from testcase string (e.g., 'comm.nccl.allreduce' -> 'allreduce')."""
    parts = testcase.split(".")
    return parts[2] if len(parts) > 2 else "unknown"


def extract_run_info(data: Dict[str, Any], path: Path = None) -> Dict[str, Any]:
    """
    Extract run info from test result data.

    Args:
        data: Test result JSON data
        path: Optional file path (for file-based sources)

    Returns:
        Dictionary with extracted run information
    """
    config = data.get("config", {})
    resolved = data.get("resolved", {})

    device_used = (
        resolved.get("device_used")
        or config.get("device_used")
        or config.get("device_involved", 1)
    )

    nodes = resolved.get("nodes") or data.get("environment", {}).get("cluster_scale", 1)

    result_code = data.get("result_code", 1)
    success = result_code == 0

    metrics = data.get("metrics", [])
    metric_types = [m.get("name", "").split(".")[0] for m in metrics if m.get("name")]

    testcase = data.get("testcase", "")

    return {
        "path": path,
        "testcase": testcase,
        "run_id": data.get("run_id", "unknown"),
        "time": data.get("time", ""),
        "success": success,
        "result_code": result_code,
        "test_type": extract_test_type(testcase),
        "operation": extract_operation(testcase),
        "config": config,
        "resolved": resolved,
        "device_used": device_used,
        "nodes": nodes,
        "metrics_count": len(metrics),
        "metric_types": list(set(metric_types)),
    }
