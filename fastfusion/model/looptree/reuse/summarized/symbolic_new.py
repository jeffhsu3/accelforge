from dataclasses import dataclass, field

from fastfusion.frontend.mapping import Mapping
from fastfusion.frontend.workload import (
    Workload,
    get_rank_variable_bounds,
    get_tensor_size
)

import sympy


@dataclass
class SummarizedAnalysisOutput:
    ops: dict = field(default_factory=dict)
    fills: dict = field(default_factory=dict)
    occupancy: dict = field(default_factory=dict)
    op_occupancy: dict = field(default_factory=dict)
    reads_to_peer: dict = field(default_factory=dict)
    reads_to_parent: dict = field(default_factory=dict)
    temporal_steps: dict = field(default_factory=dict)
    fanout: dict = field(default_factory=dict)
    op_intensity: dict = field(default_factory=dict)


def analyze_reuse(
    mapping: Mapping,
    workload: Workload
) -> SummarizedAnalysisOutput:
    mapping = mapping.nodes
    einsum_name = mapping[-1]['einsum']
    einsum_shape = get_rank_variable_bounds(workload, einsum_name)

    all_tensors = (
        workload.tensors_read_by_einsum(einsum_name)
        |
        workload.tensors_written_by_einsum(einsum_name)
    )

    tensor_size = {
        tensor: get_tensor_size(workload, tensor) for tensor in all_tensors
    }
    original_tensor_size = tensor_size.copy()

    symbols = add_symbols_if_needed(mapping)

    for node in mapping:
        if node.type == "temporal":
            stride_and_shape = get_stride_and_tile_shape(node, full_shape, n)
            # TODO
        elif node.type == "spatial":
            stride_and_shape = get_stride_and_tile_shape(node, full_shape, n)
            # TODO
        elif node.type == "storage":
            pass
        elif node.type == "compute":
            pass
        else:
            raise NotImplementedError(f'unsupported node type {node.type}')


@dataclass
class RepeatedValue[T]:
    value: T
    repeats: int


@dataclass
class SequenceOfRepatedvalues[T]:
    sequence: list[RepeatedValue[T]]


@dataclass
class StrideAndShape:
    stride: any
    shape: any


def get_stride_and_tile_shape(node, full_shape, n: int):
    if "tile_shape" in node:
        tile_shape = node.tile_shape

        assume_perfect_factor = \
            "assume_perfect" in node and node["assume_perfect"]

        if assume_perfect_factor or known_perfect_factor(tile_shape, full_shape):
            factor = full_shape // tile_shape
            return StrideAndShape(tile_shape, RepeatedValue(tile_shape, factor))
        else:
            factor = sympy.ceiling(full_shape / tile_shape)
            return make_possibly_different_last(tile_shape, factor, full_shape)

    elif "factor" in node:
        factor = node.factor

        assume_perfect_factor = \
            "assume_perfect" in node and node["assume_perfect"]

        if assume_perfect_factor or known_perfect_factor(factor, full_shape):
            tile_shape = full_shape // factor
            return StrideAndShape(tile_shape, RepeatedValue(tile_shape, factor))
        else:
            tile_shape = sympy.ceiling(full_shape / factor)
            return make_possibly_different_last(tile_shape, factor, full_shape)
    
    elif "tile_pattern" in node:
        stride = node["tile_pattern"]["stride"]

        if "first_shape" in node["tile_pattern"]:
            first_shape = node["tile_pattern"]["first_shape"]

            middle_shape_factor = sympy.floor((full_shape - first_shape)/stride)

            last_shape = full_shape - first_shape - stride*middle_shape_factor

            return StrideAndShape(
                stride,
                SequenceOfRepatedvalues([
                    RepeatedValue(first_shape, 1),
                    RepeatedValue(stride, middle_shape_factor),
                    RepeatedValue(last_shape, 1)
                ])
            )
        elif "shape" in node["tile_pattern"]:
            shape = node["tile_pattern"]["shape"]

            factor = sympy.ceiling(full_shape / stride)

            common_case_factor = sympy.floor((full_shape - shape)/stride)

            iterationvar = sympy.symbols(f"iteration{n}")
            last_shapes = full_shape - iterationvar*stride
            last_case_factor = factor - common_case_factor

            return StrideAndShape(
                stride,
                SequenceOfRepatedvalues([
                    RepeatedValue(shape, common_case_factor),
                    RepeatedValue(last_shapes, last_case_factor)
                ])
            )


def known_perfect_factor(divisor, full_shape):
    return (
        isinstance(divisor, int) and isinstance(full_shape, int)
        and full_shape % divisor == 1
    )


def make_possibly_different_last(common_tile_shape, factor, full_shape):
    last_shape = full_shape - common_tile_shape*(factor-1)
    all_shapes = SequenceOfRepatedvalues([
        RepeatedValue(common_tile_shape, factor-1),
        RepeatedValue(last_shape, 1)
    ])
    return StrideAndShape(common_tile_shape, all_shapes)
