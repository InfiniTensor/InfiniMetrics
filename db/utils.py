#!/usr/bin/env python3
"""Utility functions for file handling and data processing."""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ==================== CSV Utilities ====================


def load_csv_data(csv_path: Path) -> List[Dict[str, Any]]:
    """Load CSV file as list of dictionaries."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [convert_csv_row(row) for row in reader]


def convert_csv_row(row: Dict[str, str]) -> Dict[str, Any]:
    """Convert CSV row values to appropriate types."""
    result = {}
    for key, value in row.items():
        try:
            if "." in value:
                result[key] = float(value)
            else:
                result[key] = int(value)
        except (ValueError, TypeError):
            result[key] = value
    return result


def resolve_csv_path(csv_url: str, base_dir: Path) -> Optional[Path]:
    """Resolve CSV path with fallback strategies."""
    if not csv_url:
        return None

    rel = csv_url[2:] if csv_url.startswith("./") else csv_url
    rel_path = Path(rel)

    candidates = [
        base_dir / rel_path,
        base_dir / rel_path.name,
        base_dir.parent / rel_path,
        base_dir.parent / rel_path.name,
    ]

    for p in candidates:
        if p.exists():
            return p
    return None


def get_csv_base_dir(data: Dict[str, Any], json_path: Path) -> Path:
    """Get base directory for CSV resolution."""
    config = data.get("config", {})
    output_dir = config.get("output_dir")
    if output_dir:
        output_path = Path(output_dir)
        if output_path.is_absolute():
            return output_path
        return json_path.parent / output_dir
    return json_path.parent


# ==================== File Type Detection ====================


def is_valid_test_result(data: Dict[str, Any]) -> bool:
    """Check if data is a valid test result."""
    required = ["run_id", "testcase", "config"]
    return all(k in data for k in required) and "metrics" in data


def is_dispatcher_summary(data: Dict[str, Any]) -> bool:
    """Check if data is a dispatcher summary file."""
    return "results" in data and "total_tests" in data


def is_dispatcher_summary_file(path: Path) -> bool:
    """Check if file is a dispatcher summary based on name."""
    name_lower = path.name.lower()
    return "dispatcher_summary" in name_lower or (
        "summary" in name_lower and "dispatcher" in name_lower
    )


def should_skip_file(path: Path) -> bool:
    """Check if file should be skipped."""
    name_lower = path.name.lower()
    if "summary" in name_lower and not is_dispatcher_summary_file(path):
        return True
    return False


# ==================== Path Resolution ====================


# Directory name aliases for flexible path resolution
_DIR_ALIASES = {
    "comm": "communication",
    "communication": "comm",
    "infer": "inference",
    "inference": "infer",
    "hw": "hardware",
    "hardware": "hw",
    "op": "operators",
    "operators": "op",
}


def _get_path_variants(result_path: Path) -> List[Path]:
    """Generate path variants with directory name aliases."""
    variants = [result_path]
    parts = list(result_path.parts)

    for i, part in enumerate(parts):
        if part in _DIR_ALIASES:
            new_parts = parts.copy()
            new_parts[i] = _DIR_ALIASES[part]
            variants.append(Path(*new_parts))

    # Also try test_output <-> output mapping
    result_str = str(result_path)
    if result_str.startswith("test_output/"):
        variants.append(Path(result_str.replace("test_output/", "output/", 1)))
    elif result_str.startswith("output/"):
        variants.append(Path(result_str.replace("output/", "test_output/", 1)))

    return variants


def resolve_result_file_path(
    result_file: str, summary_path: Path, base_dir: Path
) -> Optional[Path]:
    """Resolve result file path from dispatcher summary reference."""
    result_path = Path(result_file)

    if result_path.is_absolute():
        return result_path

    # Generate path variants with directory name aliases
    path_variants = _get_path_variants(result_path)

    for variant in path_variants:
        candidates = [
            base_dir / variant,
            summary_path.parent.parent / variant,
            summary_path.parent / variant,
        ]
        for p in candidates:
            if p.exists():
                return p

    # Last resort: search by filename in base_dir subdirectories
    filename = result_path.name
    for p in base_dir.rglob(filename):
        return p

    return None
