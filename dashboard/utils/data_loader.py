#!/usr/bin/env python3
"""Data loading utilities for InfiniMetrics dashboard."""

import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class InfiniMetricsDataLoader:
    """Load and parse InfiniMetrics test results."""

    def __init__(self, results_dir: str = "./test_output"):
        self.results_dir = Path(results_dir)

    def list_test_runs(self, test_type: str = None) -> List[Dict[str, Any]]:
        """List all test runs, filtering out summary files."""
        runs = []

        # Search for JSON result files
        for json_file in self.results_dir.rglob("*.json"):
            try:
                # Skip summary files and dispatcher files
                if (
                    "summary" in json_file.name.lower()
                    or "dispatcher" in json_file.name.lower()
                ):
                    continue

                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Filter: must be a test result file, not a summary file
                if not self._is_test_result_file(data):
                    continue

                # Filter by test type if specified
                testcase = data.get("testcase", "")
                if test_type and not testcase.startswith(test_type):
                    continue

                # Extract basic info
                run_info = self._extract_run_info(data, json_file)

                # Extract the accelerator card type
                run_info["accelerator_types"] = extract_accelerator_types(data)
                runs.append(run_info)

            except Exception as e:
                logger.debug(f"Skipping file {json_file}: {e}")

        # Sort by time (newest first)
        runs.sort(key=lambda x: x["time"], reverse=True)
        return runs

    def load_test_result(self, json_path: Path) -> Dict[str, Any]:
        """Load a single test result with all data."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load associated CSV files
        for metric in data.get("metrics", []):
            csv_url = metric.get("raw_data_url")
            if csv_url and not csv_url.startswith("http"):
                # Get the correct base directory
                base_dir = self._get_csv_base_dir(data, json_path)
                csv_path = self._resolve_csv_path(csv_url, base_dir)

                if csv_path and csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        metric["data"] = df
                        metric["data_columns"] = list(df.columns)
                        metric["csv_path"] = str(csv_path)
                    except Exception as e:
                        logger.warning(f"Failed to load CSV {csv_path}: {e}")
                        metric["data"] = None
                else:
                    logger.debug(f"CSV not found: {csv_url} (base: {base_dir})")
                    metric["data"] = None

        return data

    def load_csv_data(
        self, csv_url: str, json_data: Dict[str, Any], json_path: Path
    ) -> Optional[pd.DataFrame]:
        """Load CSV data file using proper path resolution."""
        try:
            if csv_url.startswith("http"):
                return None

            base_dir = self._get_csv_base_dir(json_data, json_path)
            csv_path = self._resolve_csv_path(csv_url, base_dir)

            if csv_path and csv_path.exists():
                return pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Failed to load CSV {csv_url}: {e}")
        return None

    def _is_test_result_file(self, data: Dict[str, Any]) -> bool:
        """Check if JSON file is a test result (not a summary)."""
        # Must have these fields
        required = ["run_id", "testcase", "config"]
        if not all(key in data for key in required):
            return False

        # Should have metrics
        if "metrics" not in data:
            return False

        return True

    def _extract_run_info(
        self, data: Dict[str, Any], json_path: Path
    ) -> Dict[str, Any]:
        """Extract run info from test result data."""
        config = data.get("config", {})
        resolved = data.get("resolved", {})

        # Device used: try resolved first, then config
        device_used = (
            resolved.get("device_used")
            or config.get("device_used")
            or config.get("device_involved", 1)
        )

        # Nodes: try resolved first, then environment
        nodes = resolved.get("nodes") or data.get("environment", {}).get(
            "cluster_scale", 1
        )

        # Success: use result_code if available, fallback to success field
        result_code = data.get("result_code", 1)
        success = result_code == 0

        # Extract metrics count and types
        metrics = data.get("metrics", [])
        metric_types = [
            m.get("name", "").split(".")[0] for m in metrics if m.get("name")
        ]

        return {
            "path": json_path,
            "testcase": data.get("testcase", "unknown"),
            "run_id": data.get("run_id", "unknown"),
            "time": data.get("time", ""),
            "success": success,
            "result_code": result_code,
            "test_type": self._extract_test_type(data.get("testcase", "")),
            "operation": self._extract_operation(data.get("testcase", "")),
            "config": config,
            "resolved": resolved,
            "device_used": device_used,
            "nodes": nodes,
            "metrics_count": len(metrics),
            "metric_types": list(set(metric_types)),
        }

    def _get_csv_base_dir(self, json_data: Dict[str, Any], json_path: Path) -> Path:
        """Get the correct base directory for CSV files."""
        # First try: use output_dir from config
        config = json_data.get("config", {})
        output_dir = config.get("output_dir")

        if output_dir:
            output_path = Path(output_dir)
            if output_path.is_absolute():
                return output_path
            # Relative path: resolve relative to JSON file location
            return json_path.parent / output_dir

        # Second try: use JSON file's parent directory
        return json_path.parent

    def _resolve_csv_path(self, csv_url: str, base_dir: Path) -> Optional[Path]:
        """
        Resolve CSV path from raw_data_url and base_dir.

        Handles cases like:
        - base_dir/output/communication + "./comm/xxx.csv" but file is actually base_dir/"xxx.csv"
        - base_dir/output/infer + "./infer/xxx.csv" but file is base_dir/"xxx.csv"
        """
        try:
            if not csv_url:
                return None

            # strip leading "./"
            rel = csv_url[2:] if csv_url.startswith("./") else csv_url
            rel_path = Path(rel)

            candidates = []

            # 1) base_dir / rel
            candidates.append(base_dir / rel_path)

            # 2) base_dir / basename (most common fallback for your current layout)
            candidates.append(base_dir / rel_path.name)

            # 3) base_dir.parent / rel (just in case)
            candidates.append(base_dir.parent / rel_path)

            # 4) base_dir.parent / basename
            candidates.append(base_dir.parent / rel_path.name)

            for p in candidates:
                if p.exists():
                    return p

            return None
        except Exception:
            return None

    def _extract_test_type(self, testcase: str) -> str:
        """Extract test type from testcase string."""
        parts = testcase.split(".")
        if len(parts) > 0:
            return parts[0]  # comm, infer, operator, etc.
        return "unknown"

    def _extract_operation(self, testcase: str) -> str:
        """Extract operation from testcase string."""
        parts = testcase.split(".")
        if len(parts) > 2:
            return parts[2]  # AllReduce, Direct, Conv, etc.
        return "unknown"


def load_summary_file(summary_path: str = "./summary_output") -> List[Dict[str, Any]]:
    """Load dispatcher summary files."""
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
    """
    Extract the accelerator card type from result_json
    """
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
