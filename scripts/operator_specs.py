"""
Operator Specifications Module

Defines input/output formats and validation rules for each operator.
This module makes it easy to add new operators by simply adding a new spec.
"""

from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class OperatorSpec:
    """Specification for an operator's input/output format"""
    name: str
    display_name: str
    min_inputs: int
    max_inputs: int
    # Shape requirements: each element is (input_index, dimension_index, constraint_description)
    shape_constraints: List[str]
    description: str

    def get_input_shapes(self, base_shape: List[int], index: int = 0) -> List[List[int]]:
        """
        Generate input shapes based on base shape and index.

        Args:
            base_shape: Base shape template from CLI
            index: Test case index for variation

        Returns:
            List of input shapes
        """
        raise NotImplementedError("Subclasses must implement get_input_shapes")

    def get_output_shape(self, input_shapes: List[List[int]]) -> List[int]:
        """
        Calculate output shape based on input shapes.

        Args:
            input_shapes: List of input shapes

        Returns:
            Output shape
        """
        raise NotImplementedError("Subclasses must implement get_output_shape")

    def get_input_names(self) -> List[str]:
        """Get default input names for this operator"""
        raise NotImplementedError("Subclasses must implement get_input_names")


class MatMulSpec(OperatorSpec):
    """Matrix Multiplication operator specification"""

    def __init__(self):
        super().__init__(
            name="matmul",
            display_name="Matrix Multiplication",
            min_inputs=2,
            max_inputs=2,
            shape_constraints=[
                "Input 0 (A): [M, K] - Left matrix",
                "Input 1 (B): [K, N] - Right matrix",
                "Constraint: A's last dimension must equal B's second-to-last dimension"
            ],
            description="Matrix multiplication: C = A @ B"
        )

    def get_input_shapes(self, base_shape: List[int], index: int = 0) -> List[List[int]]:
        """
        MatMul shape interpretation:
        - 2 params: [M, K] -> both matrices use [M, K] and [K, M]
        - 3 params: [M, K, N] -> first is [M, K], second is [K, N]
        """
        if len(base_shape) == 2:
            m, k = base_shape
            n = k  # Square matrices by default
            return [[m, k], [k, n]]
        elif len(base_shape) == 3:
            m, k, n = base_shape
            return [[m, k], [k, n]]
        else:
            raise ValueError(f"MatMul requires 2 or 3 dimensions, got {len(base_shape)}")

    def get_output_shape(self, input_shapes: List[List[int]]) -> List[int]:
        """Output shape: [M, N]"""
        shape_a = input_shapes[0]
        shape_b = input_shapes[1]
        return [shape_a[0], shape_b[-1]]

    def get_input_names(self) -> List[str]:
        return ["a", "b"]


class AddSpec(OperatorSpec):
    """Element-wise Addition operator specification"""

    def __init__(self):
        super().__init__(
            name="add",
            display_name="Element-wise Addition",
            min_inputs=2,
            max_inputs=2,
            shape_constraints=[
                "Both inputs must have identical shapes",
                "Supports broadcasting for NumPy-style operations"
            ],
            description="Element-wise addition: C = A + B"
        )

    def get_input_shapes(self, base_shape: List[int], index: int = 0) -> List[List[int]]:
        """Both inputs use the same shape"""
        if len(base_shape) < 2:
            raise ValueError(f"Add requires at least 2D shape, got {len(base_shape)}")
        return [base_shape.copy(), base_shape.copy()]

    def get_output_shape(self, input_shapes: List[List[int]]) -> List[int]:
        """Output shape same as inputs"""
        return input_shapes[0].copy()

    def get_input_names(self) -> List[str]:
        return ["a", "b"]


class SubSpec(OperatorSpec):
    """Element-wise Subtraction operator specification"""

    def __init__(self):
        super().__init__(
            name="sub",
            display_name="Element-wise Subtraction",
            min_inputs=2,
            max_inputs=2,
            shape_constraints=[
                "Both inputs must have identical shapes"
            ],
            description="Element-wise subtraction: C = A - B"
        )

    def get_input_shapes(self, base_shape: List[int], index: int = 0) -> List[List[int]]:
        """Both inputs use the same shape"""
        if len(base_shape) < 2:
            raise ValueError(f"Sub requires at least 2D shape, got {len(base_shape)}")
        return [base_shape.copy(), base_shape.copy()]

    def get_output_shape(self, input_shapes: List[List[int]]) -> List[int]:
        """Output shape same as inputs"""
        return input_shapes[0].copy()

    def get_input_names(self) -> List[str]:
        return ["a", "b"]


class MulSpec(OperatorSpec):
    """Element-wise Multiplication operator specification"""

    def __init__(self):
        super().__init__(
            name="mul",
            display_name="Element-wise Multiplication",
            min_inputs=2,
            max_inputs=2,
            shape_constraints=[
                "Both inputs must have identical shapes"
            ],
            description="Element-wise multiplication: C = A * B"
        )

    def get_input_shapes(self, base_shape: List[int], index: int = 0) -> List[List[int]]:
        """Both inputs use the same shape"""
        if len(base_shape) < 2:
            raise ValueError(f"Mul requires at least 2D shape, got {len(base_shape)}")
        return [base_shape.copy(), base_shape.copy()]

    def get_output_shape(self, input_shapes: List[List[int]]) -> List[int]:
        """Output shape same as inputs"""
        return input_shapes[0].copy()

    def get_input_names(self) -> List[str]:
        return ["a", "b"]


class DivSpec(OperatorSpec):
    """Element-wise Division operator specification"""

    def __init__(self):
        super().__init__(
            name="div",
            display_name="Element-wise Division",
            min_inputs=2,
            max_inputs=2,
            shape_constraints=[
                "Both inputs must have identical shapes"
            ],
            description="Element-wise division: C = A / B"
        )

    def get_input_shapes(self, base_shape: List[int], index: int = 0) -> List[List[int]]:
        """Both inputs use the same shape"""
        if len(base_shape) < 2:
            raise ValueError(f"Div requires at least 2D shape, got {len(base_shape)}")
        return [base_shape.copy(), base_shape.copy()]

    def get_output_shape(self, input_shapes: List[List[int]]) -> List[int]:
        """Output shape same as inputs"""
        return input_shapes[0].copy()

    def get_input_names(self) -> List[str]:
        return ["a", "b"]


# Operator registry
OPERATOR_REGISTRY: Dict[str, OperatorSpec] = {
    "matmul": MatMulSpec(),
    "add": AddSpec(),
    "sub": SubSpec(),
    "mul": MulSpec(),
    "div": DivSpec(),
}


def get_operator_spec(operator_name: str) -> OperatorSpec:
    """
    Get operator specification by name.

    Args:
        operator_name: Name of the operator

    Returns:
        OperatorSpec instance

    Raises:
        ValueError: If operator not found
    """
    operator_name = operator_name.lower()
    if operator_name not in OPERATOR_REGISTRY:
        available = ", ".join(OPERATOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown operator: {operator_name}. "
            f"Available operators: {available}"
        )
    return OPERATOR_REGISTRY[operator_name]


def list_supported_operators() -> List[str]:
    """Get list of supported operator names"""
    return list(OPERATOR_REGISTRY.keys())
