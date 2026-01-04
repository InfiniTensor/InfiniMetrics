#!/usr/bin/env python3
"""
Unified Metrics Collector for All Adapters

Provides standardized metrics collection and reporting across different adapters.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MetricData:
    """
    Generic metric data container.

    Represents a single metric with its value, unit, and type.

    Attributes:
        name: Metric name (e.g., "latency", "throughput")
        value: Metric value (can be scalar, string, or file path for timeseries)
        unit: Unit of measurement (e.g., "ms", "GB/s", "tokens/s")
        metric_type: Type of metric ("scalar", "timeseries", "string")
        metadata: Additional metadata about the metric
    """

    name: str
    value: Any
    unit: Optional[str] = None
    metric_type: str = "scalar"  # scalar, timeseries, string
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metric to dictionary format.

        Returns:
            Dictionary representation suitable for JSON serialization

        Examples:
            >>> metric = MetricData("latency", 123.45, "ms", "scalar")
            >>> metric.to_dict()
            {'name': 'latency', 'type': 'scalar', 'value': 123.45, 'unit': 'ms'}
        """
        result = {
            "name": self.name,
            "type": self.metric_type,
        }

        if self.unit is not None:
            result["unit"] = self.unit

        if self.metric_type == "scalar":
            result["value"] = self.value
        elif self.metric_type == "timeseries":
            result["raw_data_url"] = str(self.value)
        elif self.metric_type == "string":
            result["value"] = str(self.value)

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    def __repr__(self) -> str:
        return f"MetricData(name='{self.name}', value={self.value}, unit='{self.unit}', type='{self.metric_type}')"


class MetricsCollector:
    """
    Unified metrics collector for all adapters.

    Provides a consistent interface for collecting, storing, and exporting
    performance metrics across different adapters and use cases.

    Examples:
        >>> collector = MetricsCollector()
        >>> collector.add_metric("latency", 123.45, unit="ms")
        >>> collector.add_metric("accuracy", 0.95, unit="ratio")
        >>> collector.add_metric("status", "PASS", metric_type="string")
        >>> metrics = collector.to_dict_list()
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize metrics collector.

        Args:
            name: Optional name for this collector (useful for debugging)
        """
        self.name = name or "collector"
        self._metrics: Dict[str, MetricData] = {}

    def add_metric(
        self,
        name: str,
        value: Any,
        unit: Optional[str] = None,
        metric_type: str = "scalar",
        **metadata
    ) -> None:
        """
        Add a metric to the collection.

        Args:
            name: Metric name (will overwrite existing metric with same name)
            value: Metric value
            unit: Unit of measurement
            metric_type: Type of metric ("scalar", "timeseries", "string")
            **metadata: Additional metadata key-value pairs

        Examples:
            >>> collector.add_metric("latency", 100.5, unit="ms")
            >>> collector.add_metric(
            ...     "throughput_timeseries",
            ...     "./data/throughput.csv",
            ...     metric_type="timeseries"
            ... )
            >>> collector.add_metric(
            ...     "test_status",
            ...     "PASS",
            ...     metric_type="string"
            ... )
        """
        metric = MetricData(
            name=name,
            value=value,
            unit=unit,
            metric_type=metric_type,
            metadata=metadata
        )
        self._metrics[name] = metric
        logger.debug(f"[{self.name}] Added metric: {name} = {value} {unit or ''}")

    def get_metric(self, name: str) -> Optional[MetricData]:
        """
        Get a metric by name.

        Args:
            name: Metric name

        Returns:
            MetricData object or None if not found

        Examples:
            >>> metric = collector.get_metric("latency")
            >>> if metric:
            ...     print(f"Latency: {metric.value} {metric.unit}")
        """
        return self._metrics.get(name)

    def has_metric(self, name: str) -> bool:
        """
        Check if a metric exists.

        Args:
            name: Metric name

        Returns:
            True if metric exists
        """
        return name in self._metrics

    def remove_metric(self, name: str) -> bool:
        """
        Remove a metric by name.

        Args:
            name: Metric name

        Returns:
            True if metric was removed, False if not found
        """
        if name in self._metrics:
            del self._metrics[name]
            logger.debug(f"[{self.name}] Removed metric: {name}")
            return True
        return False

    def get_all_metrics(self) -> List[MetricData]:
        """
        Get all metrics as a list.

        Returns:
            List of all MetricData objects

        Examples:
            >>> for metric in collector.get_all_metrics():
            ...     print(f"{metric.name}: {metric.value}")
        """
        return list(self._metrics.values())

    def get_metric_names(self) -> List[str]:
        """
        Get all metric names.

        Returns:
            List of metric names

        Examples:
            >>> names = collector.get_metric_names()
            >>> print(names)
            ['latency', 'throughput', 'accuracy']
        """
        return list(self._metrics.keys())

    @property
    def count(self) -> int:
        """Get number of metrics collected."""
        return len(self._metrics)

    def clear(self) -> None:
        """Clear all metrics."""
        self._metrics.clear()
        logger.debug(f"[{self.name}] Cleared all metrics")

    def update_metric(self, name: str, value: Any = None, **kwargs) -> bool:
        """
        Update an existing metric.

        Args:
            name: Metric name
            value: New value (None to keep existing)
            **kwargs: Other attributes to update (unit, metric_type, metadata)

        Returns:
            True if metric was updated, False if not found

        Examples:
            >>> collector.update_metric("latency", value=150.0)
            >>> collector.update_metric("latency", unit="ms")
        """
        metric = self._metrics.get(name)
        if not metric:
            logger.warning(f"[{self.name}] Cannot update non-existent metric: {name}")
            return False

        if value is not None:
            metric.value = value

        for key, val in kwargs.items():
            if hasattr(metric, key):
                setattr(metric, key, val)

        logger.debug(f"[{self.name}] Updated metric: {name}")
        return True

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """
        Convert all metrics to a list of dictionaries.

        Returns:
            List of metric dictionaries suitable for JSON serialization

        Examples:
            >>> metrics_dict = collector.to_dict_list()
            >>> import json
            >>> print(json.dumps(metrics_dict, indent=2))
        """
        return [metric.to_dict() for metric in self._metrics.values()]

    def to_dict_by_name(self) -> Dict[str, Dict[str, Any]]:
        """
        Convert all metrics to a dictionary keyed by metric name.

        Returns:
            Dictionary mapping metric names to their dict representations

        Examples:
            >>> metrics_dict = collector.to_dict_by_name()
            >>> latency_info = metrics_dict.get("latency", {})
        """
        return {name: metric.to_dict() for name, metric in self._metrics.items()}

    def merge(self, other: 'MetricsCollector', prefix: str = "") -> None:
        """
        Merge metrics from another collector.

        Args:
            other: Another MetricsCollector instance
            prefix: Optional prefix for merged metric names

        Examples:
            >>> collector1 = MetricsCollector()
            >>> collector2 = MetricsCollector()
            >>> collector2.add_metric("extra_metric", 42, unit="count")
            >>> collector1.merge(collector2, prefix="adapter2_")
            >>> # Now collector1 has "adapter2_extra_metric"
        """
        for metric in other.get_all_metrics():
            new_name = f"{prefix}{metric.name}"
            # Create a copy to avoid shared references
            new_metric = MetricData(
                name=new_name,
                value=metric.value,
                unit=metric.unit,
                metric_type=metric.metric_type,
                metadata=metric.metadata.copy()
            )
            self._metrics[new_name] = new_metric

        logger.debug(f"[{self.name}] Merged {other.count} metrics from {other.name}")

    def filter_by_type(self, metric_type: str) -> List[MetricData]:
        """
        Get metrics filtered by type.

        Args:
            metric_type: Type to filter by ("scalar", "timeseries", "string")

        Returns:
            List of metrics of the specified type

        Examples:
            >>> scalar_metrics = collector.filter_by_type("scalar")
            >>> for metric in scalar_metrics:
            ...     print(f"{metric.name}: {metric.value}")
        """
        return [
            metric for metric in self._metrics.values()
            if metric.metric_type == metric_type
        ]

    def get_scalar_values(self) -> Dict[str, float]:
        """
        Get all scalar metrics as a name-value dictionary.

        Returns:
            Dictionary mapping scalar metric names to float values

        Examples:
            >>> scalars = collector.get_scalar_values()
            >>> print(f"Average latency: {scalars.get('latency', 0):.2f} ms")
        """
        return {
            metric.name: float(metric.value)
            for metric in self._metrics.values()
            if metric.metric_type == "scalar" and isinstance(metric.value, (int, float))
        }

    def summary(self) -> str:
        """
        Generate a human-readable summary of collected metrics.

        Returns:
            Summary string

        Examples:
            >>> print(collector.summary())
            MetricsCollector (5 metrics):
              - latency: 123.45 ms (scalar)
              - throughput: 1000.0 tokens/s (scalar)
              - accuracy: 0.95 (scalar)
              - data: ./data/timeseries.csv (timeseries)
              - status: PASS (string)
        """
        lines = [f"MetricsCollector '{self.name}' ({self.count} metrics):"]

        if not self._metrics:
            lines.append("  (no metrics)")
        else:
            for metric in self._metrics.values():
                value_str = str(metric.value)
                if len(value_str) > 50:
                    value_str = value_str[:47] + "..."
                lines.append(f"  - {metric.name}: {value_str} {metric.unit or ''} ({metric.metric_type})")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"MetricsCollector(name='{self.name}', count={self.count})"

    def __len__(self) -> int:
        """Allow len() on collector."""
        return self.count
