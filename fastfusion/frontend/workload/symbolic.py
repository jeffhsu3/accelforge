from dataclasses import dataclass
from typing import Any
from functools import reduce
from operator import mul

import sympy

from .workload import TensorName, Einsum, Workload
from .isl import get_rank_variable_bounds


def get_projection_expr(einsum: Einsum, tensor: TensorName) -> dict[str, sympy.Expr]:
    projection = einsum.tensor_accesses[tensor].projection
    return {
        rank_name: sympy.parsing.sympy_parser.parse_expr(proj_str)
        for rank_name, proj_str in projection.items()
    }


class Irrelevant:
    pass

@dataclass
class Relevant:
    rank: Any

@dataclass
class PartiallyRelevant:
    rank: Any


def get_rank_variable_relevancy(einsum: Einsum,
                                tensor: TensorName):
    relevancy = {}
    projection = einsum.tensor_accesses[tensor].projection
    for rank_variable in einsum.rank_variables:
        relevancy[rank_variable] = Irrelevant()
        for rank_name, projection_str in projection.items():
            projection_expr = sympy.parsing.sympy_parser.parse_expr(projection_str)
            is_simple = len(sympy.Add.make_args(projection_expr)) == 1
            is_relevant = sympy.symbols(f'{rank_variable}') in projection_expr.free_symbols

            if not is_relevant:
                continue

            if is_simple:
                relevancy[rank_variable] = Relevant(rank=rank_name)
            else:
                relevancy[rank_variable] = PartiallyRelevant(rank=rank_name)

            break
    return relevancy


def compute_dense_tile_occupancy(
    projection_expr: dict[str, sympy.Expr],
    rank_variable_shapes: dict
):
    substitutions = [
        (rank_variable, rank_variable_shape - 1)
        for rank_variable, rank_variable_shape in rank_variable_shapes.items()
    ]
    return reduce(
        mul,
        [
            index_expr.subs(substitutions) + 1
            for index_expr in projection_expr.values()
        ]
    )


def compute_rank_occupancy(
    projection_expr: sympy.Expr,
    rank_variable_shapes: dict
):
    substitutions = [
        (rank_variable, rank_variable_shape - 1)
        for rank_variable, rank_variable_shape in rank_variable_shapes.items()
    ]
    return projection_expr.subs(substitutions) + 1


def get_stride_and_halo(
    workload: Workload
) -> dict[str, dict[tuple[str, str], int]]:
    """
    Get stride and halo (initial delta) for Einsums in workload.

    Assumes each (producer, consumer) pair shares only one tensor.

    Returns dictionary mapping (producer, consumer) to another dictionary mapping
    (producer rank var, consumer rank var) to the stride and halo.
    """
    deltas = {}
    for producer in workload.einsums:
        output_tensors = producer.output_tensors()
        if len(output_tensors) > 1:
            raise ValueError('Does not support more output tensors than one.')

        tensor = next(iter(output_tensors))

        prod_rank2rank_vars = producer.tensor_accesses[tensor].rank2rank_variables

        for consumer in workload.einsums_that_read_tensor(tensor):
            delta_for_pair = deltas.setdefault((producer.name, consumer.name), {})
            projection = get_projection_expr(consumer, tensor)
            cons_shape = get_rank_variable_bounds(workload, consumer.name)

            cons_rank2rank_vars = consumer.tensor_accesses[tensor].rank2rank_variables
            for cons_rank, cons_rank_vars in cons_rank2rank_vars.items():
                if cons_rank not in prod_rank2rank_vars:
                    continue
                prod_rank_vars = prod_rank2rank_vars[cons_rank]
                rank_projection = projection[cons_rank]
                if len(prod_rank_vars) != 1:
                    continue  # Unclear what to do in this case

                prod_rank_var = next(iter(prod_rank_vars))

                for cons_rank_var in cons_rank_vars:
                    stride = rank_projection.coeff(cons_rank_var)

                    # Careful: in-place mutation of cons_shape
                    original_shape = cons_shape[cons_rank_var]
                    cons_shape[cons_rank_var] = 1
                    halo = (
                        compute_rank_occupancy(rank_projection, cons_shape)
                        - 1
                    )
                    cons_shape[cons_rank_var] = original_shape

                    delta_for_pair[(prod_rank_var, cons_rank_var)] = (stride, halo)
    return deltas