from typing import Any
from dataclasses import dataclass, field

from fastfusion.frontend.mapping import Mapping
from fastfusion.frontend.workload import (
    Workload,
    get_rank_variable_bounds,
    get_tensor_size
)

import sympy


@dataclass(eq=True, frozen=True)
class Buffet:
    tensor: str
    einsum: str
    level: str


@dataclass
class Compute:
    einsum: str
    level: str


@dataclass
class BuffetStats:
    fills: Any = field(default=None)
    reads_to_peer: Any = field(default=None)
    reads_to_parent: Any = field(default=None)
    occupancy: Any = field(default=None)
    fanout: Any = field(default=None)


@dataclass
class SummarizedAnalysisOutput:
    per_einsum_ops: dict = field(default_factory=dict)
    op_occupancy: dict = field(default_factory=dict)

    buffet_stats: dict[Buffet, BuffetStats] = field(default_factory=dict)

    temporal_steps: dict = field(default_factory=dict)
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

    # symbols = add_symbols_if_needed(mapping)

    # insert_reservation_nodes(mapping)

    result = SummarizedAnalysisOutput()

    analyze_node(result, 0, einsum_shape, mapping)


def analyze_node(result_accumulator, node_idx, current_shape, mapping):
    for cur_idx in range(node_idx, len(mapping)):
        node = mapping[cur_idx]
        print(node)
        if node['type'] == 'temporal':
            analyze_temporal(result_accumulator, node_idx, current_shape, mapping)
            break
        elif node['type'] == 'spatial':
            analyze_spatial(result_accumulator, node_idx, current_shape, mapping)
            break
        elif node['type'] == 'storage':
            analyze_storage(result_accumulator, node_idx, current_shape, mapping)
        elif node['type'] == 'reservation':
            pass
            # analyze_reservation(result_accumulator, node_idx, current_shape, mapping)
        elif node['type'] == 'compute':
            # analyze_compute(result_accumulator, node_idx, current_shape, mapping)
            break


def analyze_temporal(result_accumulator, node_idx, current_shape, mapping):
    node = mapping[node_idx]
    stride_and_shape = get_stride_and_tile_shape(node, current_shape, node_idx)

    analyze_node(result_accumulator, node_idx+1, current_shape, mapping)


def analyze_spatial(result_accumulator, node_idx, current_shape, mapping):
    analyze_node(result_accumulator, node_idx+1, current_shape, mapping)


def analyze_storage(result_accumulator, node_idx, current_shape, mapping):
    node = mapping[node_idx]

    for tensor in node['tensor']:
        buffet = Buffet(tensor, mapping[-1]['einsum'], node['level'])
        buffet_stats = BuffetStats()
        buffet_stats.reads_to_parent = None # TODO
        result_accumulator.buffet_stats[buffet] = buffet_stats


def recurse_next_loop(node_idx, last_shape, stride_and_shape, mapping):
    stride = stride_and_shape.stride
    shape = stride_and_shape.shape
    if isinstance(shape, SequenceOfRepatedvalues):
        for repeated_shape in shape.sequence:
            assert isinstance(repeated_shape, RepeatedValue)
            shape_value = repeated_shape.value
            shape_repeats = repeated_shape.repeats
            full_shape = update_full_shape(full_shape)
            result = analyze_loop(mapping, node_idx, full_shape)
            full_shape = reset_full_shape()

            consolidate_result(accumulated_result, result, repeated_shape)
    elif isinstance(shape, RepeatedValue):
        shape_value = shape.value
        shape_repeats = shape.repeats

        full_shape = update_full_shape(full_shape)
        result = analyze_loop(mapping, node_idx, full_shape)
        full_shape = reset_full_shape()

        consolidate_result(accumulated_result, result, repeated_shape)

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
    rank = node['rank']
    rank_shape = full_shape[rank]

    if "tile_shape" in node:
        tile_shape = node['tile_shape']

        assume_perfect_factor = \
            "assume_perfect" in node and node["assume_perfect"]

        if assume_perfect_factor or known_perfect_factor(tile_shape, rank_shape):
            factor = rank_shape // tile_shape
            return StrideAndShape(tile_shape, RepeatedValue(tile_shape, factor))
        else:
            factor = sympy.ceiling(rank_shape / tile_shape)
            return make_possibly_different_last(tile_shape, factor, rank_shape)

    elif "factor" in node:
        factor = node['factor']

        assume_perfect_factor = \
            "assume_perfect" in node and node["assume_perfect"]

        if assume_perfect_factor or known_perfect_factor(factor, rank_shape):
            tile_shape = rank_shape // factor
            return StrideAndShape(tile_shape, RepeatedValue(tile_shape, factor))
        else:
            tile_shape = sympy.ceiling(rank_shape / factor)
            return make_possibly_different_last(tile_shape, factor, rank_shape)
    
    elif "tile_pattern" in node:
        stride = node["tile_pattern"]["stride"]

        if "first_shape" in node["tile_pattern"]:
            first_shape = node["tile_pattern"]["first_shape"]

            middle_shape_factor = sympy.floor((rank_shape - first_shape)/stride)

            last_shape = rank_shape - first_shape - stride*middle_shape_factor

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

            factor = sympy.ceiling(rank_shape / stride)

            common_case_factor = sympy.floor((rank_shape - shape)/stride)

            iterationvar = sympy.symbols(f"iteration{n}")
            last_shapes = rank_shape - iterationvar*stride
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
