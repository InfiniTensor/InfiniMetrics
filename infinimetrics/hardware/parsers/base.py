"""Base class for hardware test output parsers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseOutputParser(ABC):
    """Abstract base class for parsing hardware test output."""

    def __init__(self, metric_prefix: str = ""):
        """
        Initialize parser.

        Args:
            metric_prefix: Prefix to add to all metric names
        """
        self.metric_prefix = metric_prefix

    @abstractmethod
    def parse(self, output: str, run_id: str) -> List[Dict[str, Any]]:
        """
        Parse test output and extract metrics.

        Args:
            output: Raw test output string
            run_id: Test run identifier

        Returns:
            List of metric dictionaries
        """
        pass

    def create_metric(
        self,
        name: str,
        value: float,
        metric_type: str = "scalar",
        unit: str = "",
    ) -> Dict[str, Any]:
        """
        Create a standardized metric dictionary.

        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric (scalar or timeseries)
            unit: Unit of measurement

        Returns:
            Metric dictionary
        """
        return {
            "name": f"{self.metric_prefix}.{name}" if self.metric_prefix else name,
            "value": value,
            "type": metric_type,
            "unit": unit,
        }

    def create_timeseries_metric(
        self,
        name: str,
        data_url: str,
        unit: str = "",
    ) -> Dict[str, Any]:
        """
        Create a timeseries metric with external data file.

        Args:
            name: Metric name
            data_url: Path to data file (relative)
            unit: Unit of measurement

        Returns:
            Timeseries metric dictionary
        """
        return {
            "name": f"{self.metric_prefix}.{name}" if self.metric_prefix else name,
            "type": "timeseries",
            "raw_data_url": data_url,
            "unit": unit,
        }
