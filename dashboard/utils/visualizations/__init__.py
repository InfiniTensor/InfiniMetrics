#!/usr/bin/env python3
"""Visualization functions for InfiniMetrics dashboard.

This package provides visualization utilities organized by test type:
- base: Common/legacy visualization functions
- hardware: Hardware test visualizations (memory sweep, cache bandwidth)
- (future) communication: Communication test visualizations
- (future) inference: Inference test visualizations
- (future) operator: Operator test visualizations
"""

from .base import (
    plot_metric_vs_size,
    plot_comparison_matrix,
    create_summary_table,
    create_gauge_chart,
    plot_timeseries_auto,
    create_summary_table_infer,
    create_summary_table_ops,
)
from .hardware import (
    create_summary_table_hw,
    plot_hw_mem_sweep,
    plot_hw_cache,
)

__all__ = [
    # Base (common/legacy)
    "plot_metric_vs_size",
    "plot_comparison_matrix",
    "create_summary_table",
    "create_gauge_chart",
    "plot_timeseries_auto",
    "create_summary_table_infer",
    "create_summary_table_ops",
    # Hardware
    "create_summary_table_hw",
    "plot_hw_mem_sweep",
    "plot_hw_cache",
]
