"""
JSON Configuration Generator Module

Generates different JSON configuration formats for InfiniMetrics.
This module separates the JSON generation logic from the batch generation logic.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime


class JSONConfigGenerator:
    """Base class for JSON config generation"""

    def __init__(self,
                 operator: str,
                 device: str = "nvidia",
                 data_base_dir: str = ".",
                 warmup_iterations: int = 10,
                 measured_iterations: int = 100,
                 tolerance_atol: float = 1e-3,
                 tolerance_rtol: float = 1e-3):
        """
        Initialize JSON generator with common settings.

        Args:
            operator: Operator name
            device: Target device (nvidia, cambricon, etc.)
            data_base_dir: Base directory for data files
            warmup_iterations: Number of warmup iterations
            measured_iterations: Number of measured iterations
            tolerance_atol: Absolute tolerance
            tolerance_rtol: Relative tolerance
        """
        self.operator = operator
        self.device = device
        self.data_base_dir = data_base_dir
        self.warmup_iterations = warmup_iterations
        self.measured_iterations = measured_iterations
        self.tolerance_atol = tolerance_atol
        self.tolerance_rtol = tolerance_rtol

    def generate(self,
               run_id: str,
               inputs: List[Dict[str, Any]],
               output_shape: List[int],
               attributes: Optional[List[Dict[str, Any]]] = None,
               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate complete JSON configuration.

        Args:
            run_id: Unique run identifier
            inputs: List of input configurations with file_path
            output_shape: Output tensor shape
            attributes: Optional operator attributes
            metadata: Optional metadata to include

        Returns:
            Complete JSON configuration dictionary
        """
        config = {
            "run_id": run_id,
            "testcase": f"operator.InfiniCore.{self.operator.capitalize()}",
            "config": {
                "operator": self.operator,
                "device": self.device,
                "data_base_dir": self.data_base_dir,
                "inputs": inputs,
                "attributes": attributes or [],
                "outputs": [
                    {
                        "name": "output",
                        "shape": output_shape,
                        "dtype": inputs[0]["dtype"]  # Use first input's dtype
                    }
                ],
                "warmup_iterations": self.warmup_iterations,
                "measured_iterations": self.measured_iterations,
                "tolerance": {
                    "atol": self.tolerance_atol,
                    "rtol": self.tolerance_rtol
                }
            },
            "metrics": [
                {"name": "operator.latency"},
                {"name": "operator.tensor_accuracy"}
            ]
        }

        # Add metadata if provided
        if metadata:
            config["metadata"] = metadata

        return config


class InfiniMetricsJSONGenerator(JSONConfigGenerator):
    """
    Standard InfiniMetrics JSON format generator.

    Generates JSON configs compatible with InfiniMetrics framework.
    """

    def generate_batch_config(self,
                             index: int,
                             operator: str,
                             inputs: List[Dict[str, Any]],
                             output_shape: List[int],
                             **kwargs) -> Dict[str, Any]:
        """
        Generate JSON config for a batch test case.

        Args:
            index: Batch index
            operator: Operator name
            inputs: Input configurations
            output_shape: Output tensor shape
            **kwargs: Additional parameters

        Returns:
            Complete JSON configuration
        """
        run_id = f"batch.{operator}.{index:04d}"

        return self.generate(
            run_id=run_id,
            inputs=inputs,
            output_shape=output_shape
        )


class MinimalJSONGenerator(JSONConfigGenerator):
    """
    Minimal JSON format generator.

    Generates simplified JSON configs for quick testing.
    """

    def generate(self,
               run_id: str,
               inputs: List[Dict[str, Any]],
               output_shape: List[int],
               attributes: Optional[List[Dict[str, Any]]] = None,
               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate minimal JSON configuration.

        Minimal format removes optional fields like tolerance and metrics.
        """
        config = {
            "run_id": run_id,
            "operator": self.operator,
            "device": self.device,
            "data_base_dir": self.data_base_dir,
            "inputs": inputs,
            "outputs": [
                {
                    "name": "output",
                    "shape": output_shape,
                    "dtype": inputs[0]["dtype"]
                }
            ]
        }

        if attributes:
            config["attributes"] = attributes

        if metadata:
            config["metadata"] = metadata

        return config


class JSONFileWriter:
    """Handles writing JSON configs to files"""

    @staticmethod
    def write_config(config: Dict[str, Any],
                    file_path: str,
                    indent: int = 2) -> None:
        """
        Write configuration to JSON file.

        Args:
            config: Configuration dictionary
            file_path: Path to output file
            indent: JSON indentation level
        """
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=indent)

    @staticmethod
    def write_configs(configs: List[Dict[str, Any]],
                     base_dir: str,
                     filename_template: str = "format_input_{operator}_batch.{index:04d}.json") -> List[str]:
        """
        Write multiple configurations to files.

        Args:
            configs: List of configuration dictionaries
            base_dir: Base directory for files
            filename_template: Template for filenames with {index} placeholder

        Returns:
            List of written file paths
        """
        import os
        os.makedirs(base_dir, exist_ok=True)

        written_paths = []
        for i, config in enumerate(configs):
            filename = filename_template.format(
                operator=config.get("config", {}).get("operator", "unknown"),
                index=i
            )
            file_path = f"{base_dir}/{filename}"
            JSONFileWriter.write_config(config, file_path)
            written_paths.append(file_path)

        return written_paths


def create_json_generator(format_type: str = "standard",
                         **kwargs) -> JSONConfigGenerator:
    """
    Factory function to create JSON generator.

    Args:
        format_type: Type of JSON format ("standard", "minimal")
        **kwargs: Arguments passed to generator constructor

    Returns:
        JSONConfigGenerator instance
    """
    if format_type == "minimal":
        return MinimalJSONGenerator(**kwargs)
    else:
        return InfiniMetricsJSONGenerator(**kwargs)
