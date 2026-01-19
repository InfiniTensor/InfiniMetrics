#!/usr/bin/env python3
"""Constants for Hardware Test Adapter."""

# Test type mappings
TEST_TYPE_MAP = {
    "MemSweep": "memory",
    "MemBw": "bandwidth",
    "Stream": "stream",
    "Cache": "cache",
    "Comprehensive": "all",
}

# Memory direction mappings
MEMORY_DIRECTIONS = [
    ("Host to Device", "h2d"),
    ("Device to Host", "d2h"),
    ("Device to Device", "d2d"),
]

# STREAM operations
STREAM_OPERATIONS = ["copy", "scale", "add", "triad"]

# Log file mappings
LOG_FILES_MAP = {
    "MemBw": ["bandwidth_test.log"],
    "Stream": ["stream_test.log"],
    "Cache": ["cache_test.log"],
    "MemSweep": ["memory_test.log", "bandwidth_test.log"],
    "Comprehensive": ["comprehensive_test.log"],
}

# Regex patterns
SWEEP_PATTERN = r"{direction}.*?Size \(MB\)\s+Time \(ms\)Bandwidth \(GB/s\)\s+CV \(%\)\s*-+\s*(.*?)\s*(?=={direction}|Device to Device|Cache|STREAM|\Z)"
BANDWIDTH_PATTERN = r"{direction}.*?Bandwidth Test.*?Transfer Size \(Bytes\)\s+Bandwidth\(GB/s\)\s*-+\s*(.*?)\s*(?=={direction}|Device to Device|Cache|STREAM|\Z)"

L1_CACHE_PATTERN = r"L1 Cache Bandwidth Sweep Test.*?Eff\. bw\s*-+\s*\n(.*?)(?=L2 Cache|\Z)"
L2_CACHE_PATTERN = r"L2 Cache Bandwidth Sweep Test.*?Eff\. bw\s*-+\s*\n(.*?)(?=\Z)"

STREAM_PATTERN_TEMPLATE = r"STREAM_{op}\s+(\d+\.\d+)"

# CSV field names
MEMORY_CSV_FIELDS = ["size_mb", "bandwidth_gbps"]
L1_CACHE_CSV_FIELDS = ["data_set", "exec_time", "spread", "eff_bw"]
L2_CACHE_CSV_FIELDS = ["data_set", "exec_data", "exec_time", "spread", "eff_bw"]

# Test timeouts (seconds)
CACHE_TEST_TIMEOUT = 1800
DEFAULT_TEST_TIMEOUT = 600

# Metric prefixes
METRIC_PREFIX_MEM_SWEEP = "hardware.mem_sweep"
METRIC_PREFIX_MEM_BW = "hardware.mem_bw"
