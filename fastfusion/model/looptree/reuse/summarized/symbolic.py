from collections import defaultdict
from dataclasses import dataclass, field
from functools import reduce
from operator import mul

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

    tile_shapes = []

    output = SummarizedAnalysisOutput()

    latency = 1
    potential_tensor_access_multiplier = defaultdict(lambda: 1)
    actual_tensor_access_multiplier = defaultdict(lambda: 1)
    fill_multicast_factor = defaultdict(lambda: 1)
    fanout = {}
    cur_fanout = [1]
    for node in mapping:
        if node['type'] == 'temporal':
            rank_name = node['rank']
            if isinstance(rank_name, int):
                rank_id = rank_name
            else:
                rank_id = rank_name_to_id[rank_name]
            group_id = rank_groups.rank_to_group_id[rank_id]

            if 'tile_shape' not in node:
                tile_shape = sympy.symbols(f'tileshape{len(tile_shapes)}')
                tile_shapes.append(tile_shape)
            else:
                tile_shape = node['tile_shape']
            factor = sympy.ceiling(einsum_shape[group_id] / tile_shape)
            tile_shape = einsum_shape[group_id] / factor
            einsum_shape[group_id] = tile_shape

            latency *= factor

            for tensor in workload.tensors:
                relevant_ranks = tensor_to_relevant_ranks[tensor]
                if group_id in relevant_ranks:
                    actual_tensor_access_multiplier[tensor] = \
                        potential_tensor_access_multiplier[tensor]
                    tensor_size[tensor] /= factor
                else:
                    potential_tensor_access_multiplier[tensor_id] *= factor
        elif node['type'] == 'sequential':
            for tensor in workload.tensors:
                actual_tensor_access_multiplier[tensor] = \
                    potential_tensor_access_multiplier[tensor]
        elif node['type'] == 'spatial':
            rank_name = node['rank']
            if isinstance(rank_name, int):
                rank_id = rank_name
            else:
                rank_id = rank_name_to_id[rank_name]
            group_id = rank_groups.rank_to_group_id[rank_id]

            if 'tile_shape' not in node:
                tile_shape = sympy.symbols(f'tileshape{len(tile_shapes)}')
                tile_shapes.append(tile_shape)
            else:
                tile_shape = node['tile_shape']
            factor = sympy.ceiling(einsum_shape[group_id] / tile_shape)
            tile_shape = einsum_shape[group_id] / factor
            einsum_shape[group_id] = tile_shape
 
            for tensor_id in tensors:
                relevant_ranks = tensor_to_relevant_ranks[tensor_id]
                if group_id in relevant_ranks:
                    tensor_size[tensor_id] /= factor
                else:
                    fill_multicast_factor[tensor_id] *= factor

            if 'spatial' not in node:
                spatial = 0
            else:
                spatial = node['spatial']

            if spatial >= len(cur_fanout):
                cur_fanout += [1]*(spatial + 1 - len(cur_fanout))
            cur_fanout[spatial] *= factor
        elif node['type'] == 'storage':
            target = node['target']
            tensor_names = node['dspace']
            exploits_reuse = 'exploits_reuse' not in node or node['exploits_reuse']
            for tensor_id in tensor_names:
                if isinstance(tensor_id, int):
                    tensor_id = tensor_id
                else:
                    tensor_id = tensor_name_to_id[tensor_id]

                if tensor_id not in tensors:
                    continue

                if cur_fanout is None:
                    cur_fanout = [1]

                output.op_intensity[target] = (
                    sum(tensor_size.values())
                    /
                    reduce(mul, einsum_shape.values(), 1)
                )

                output.occupancy[(target, tensor_id)] = tensor_size[tensor_id]

                if not exploits_reuse:
                    actual_tensor_access_multiplier[tensor_id] = \
                        potential_tensor_access_multiplier[tensor_id]

                output.fills[(target, tensor_id, einsum_id)] = (
                    None,
                    (
                        original_tensor_size[tensor_id]
                        *
                        actual_tensor_access_multiplier[tensor_id]
                        *
                        fill_multicast_factor[tensor_id]
                    )
                )
                output.reads_to_parent[(target, tensor_id, einsum_id)] = (
                    None,
                    (
                        original_tensor_size[tensor_id]
                        *
                        actual_tensor_access_multiplier[tensor_id]
                    )
                )

                actual_tensor_access_multiplier[tensor_id] *= \
                    fill_multicast_factor[tensor_id]
                potential_tensor_access_multiplier[tensor_id] *= \
                    fill_multicast_factor[tensor_id]
                fill_multicast_factor[tensor_id] = 1

                if target not in fanout:
                    fanout[target] = cur_fanout
                cur_fanout = [1]
        elif node['type'] == 'compute':
            target = node['target']
            for tensor_id in tensors:
                output.occupancy[(target, tensor_id)] = tensor_size[tensor_id]

                output.fills[(target, tensor_id, einsum_id)] = (
                    None,
                    (
                        original_tensor_size[tensor_id]
                        *
                        potential_tensor_access_multiplier[tensor_id]
                        *
                        fill_multicast_factor[tensor_id]
                    )
                )
                output.reads_to_parent[(target, tensor_id, einsum_id)] = (
                    None,
                    (
                        original_tensor_size[tensor_id]
                        *
                        potential_tensor_access_multiplier[tensor_id]
                    )
                )
            fanout[target] = cur_fanout

    output.ops[einsum_id] = \
        (None, workload.get_operation_space_volume(einsum_id))
    output.temporal_steps[einsum_id] = latency
    output.fanout = fanout

    return tile_shapes, output
