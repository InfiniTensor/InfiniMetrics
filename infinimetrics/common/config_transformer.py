#!/usr/bin/env python3
"""
Unified Configuration Transformer

Handles conversion between different configuration formats,
simplifying the adapter implementations.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TensorSpec:
    """Simplified tensor specification."""
    name: Optional[str]
    shape: List[int]
    dtype: str
    strides: Optional[List[int]] = None
    requires_grad: bool = False


@dataclass
class OperatorSpec:
    """Simplified operator specification."""
    name: str
    device: str
    inputs: List[TensorSpec]
    outputs: List[TensorSpec]
    attributes: Dict[str, Any]
    tolerance: Dict[str, float]


class ConfigTransformer:
    """
    Unified transformer for converting between config formats.

    Simplifies the complex conversion logic in InfiniCoreAdapter.
    """

    @staticmethod
    def parse_tensor_specs(tensors: List[Dict], default_dtype: str = "float32") -> List[TensorSpec]:
        """
        Parse list of tensor specifications.

        Args:
            tensors: List of tensor dicts
            default_dtype: Default dtype if not specified

        Returns:
            List of TensorSpec objects
        """
        from common.dtype_utils import DtypeHandler

        specs = []
        for tensor in tensors:
            spec = TensorSpec(
                name=tensor.get("name"),
                shape=tensor.get("shape", []),
                dtype=DtypeHandler.normalize_dtype(tensor.get("dtype", default_dtype)),
                strides=tensor.get("strides"),
                requires_grad=tensor.get("requires_grad", False)
            )
            specs.append(spec)

        return specs

    @staticmethod
    def build_inference_config(
        operator: str,
        device: str,
        inputs: List[Dict],
        outputs: List[Dict],
        attributes: List[Dict],
        tolerance: Optional[Dict] = None
    ) -> OperatorSpec:
        """
        Build standardized operator specification.

        Args:
            operator: Operator name
            device: Device type
            inputs: Input tensor specs
            outputs: Output tensor specs
            attributes: Operator attributes
            tolerance: Tolerance config

        Returns:
            OperatorSpec object
        """
        from common.device_utils import DeviceHandler
        from common.dtype_utils import DtypeHandler

        input_specs = ConfigTransformer.parse_tensor_specs(inputs)
        output_specs = ConfigTransformer.parse_tensor_specs(outputs)

        attrs = {attr["name"]: attr["value"] for attr in attributes}

        device_normalized = DeviceHandler.normalize_device_name(device)

        return OperatorSpec(
            name=operator,
            device=device_normalized,
            inputs=input_specs,
            outputs=output_specs,
            attributes=attrs,
            tolerance=tolerance or {"atol": 1e-3, "rtol": 1e-3}
        )

    @staticmethod
    def build_runtime_args(
        config: Dict,
        default_warmup: int = 5,
        default_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Build runtime arguments from config.

        Args:
            config: Configuration dict
            default_warmup: Default warmup iterations
            default_iterations: Default measured iterations

        Returns:
            Runtime arguments dict
        """
        args = {
            "bench": "both",
            "num_prerun": int(config.get("warmup_iterations", default_warmup)),
            "num_iterations": int(config.get("measured_iterations", default_iterations)),
            "verbose": False,
            "debug": False
        }

        # Merge backend args if present
        if "backend_args" in config:
            args.update(config["backend_args"])

        return args
