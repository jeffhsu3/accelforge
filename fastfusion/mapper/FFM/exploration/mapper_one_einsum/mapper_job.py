from dataclasses import dataclass
from typing import Callable

import fastfusion.frontend.architecture as architecture
from fastfusion.frontend.mapping import (
    Iteration,
    Mapping,
    ModelOnlyNode,
    Storage,
    Spatial,
)
from fastfusion.frontend.specification import Specification
from fastfusion.frontend.workload.workload import (
    EinsumName,
    RankVariableName,
    TensorName,
    Workload,
)

from fastfusion.mapper.FFM.exploration import metrics
from fastfusion.mapper.FFM.joining.mappinginfo import (
    Compatibility,
    Loop,
    Reservation,
    TilePattern,
)
from fastfusion.mapper.FFM.exploration.contraints.constraints import (
    MappingConstraints,
)
from fastfusion.mapper.FFM.tags import Tags
from fastfusion.model.looptree.reuse.summarized.symbolic import (
    quick_insert_reservation_nodes,
)
from fastfusion.util.util import fzs
from fastfusion.util.itertools import first
from fastfusion.frontend.mapping import Reservation as ReservationNode


def make_compatibility(
    mapping: Mapping,
    intermediate_tensors: set[TensorName],
    workload: Workload
) -> Compatibility:
    einsum = workload.einsums[mapping.nodes[-1].einsum]
    fused_slice = mapping.get_fused_slice(intermediate_tensors)
    fused_loops: list[Iteration] = []
    reservations: dict[int, list[ReservationNode]] = {}
    for node in fused_slice.nodes:
        if isinstance(node, Iteration):
            fused_loops.append(node)
        elif isinstance(node, ReservationNode):
            reservations.setdefault(len(fused_loops), []).append(node)
        elif isinstance(node, ModelOnlyNode):
            continue
        elif isinstance(node, Storage):
            continue
        else:
            raise ValueError(f"Unexpected node type: {type(node)}")

    compatibility_reservations = []
    for above_loop_index, reservation_nodes in reservations.items():
        for reservation in reservation_nodes:
            tensor = reservation.tensor
            rank_var2ranks = einsum.tensor_accesses[tensor].rank_variable2ranks
            tensor_loops = []
            for loop_idx, loop in enumerate(fused_loops[:above_loop_index]):
                ranks = rank_var2ranks[loop.rank_variable]
                if len(ranks) > 1:
                    raise NotImplementedError('co-iteration of ranks with one rank var.')
                if len(ranks) == 0:
                    raise NotImplementedError('recomputation')

                rank = first(ranks)
                tensor_loops.append(Loop(rank, None, isinstance(loop, Spatial)))

            compatibility_reservations.append(
                Reservation(
                    name=reservation.tensor,
                    loops=tensor_loops,
                    resource_name=reservation.memory,
                    size=0,  # TODO: Get size
                )
            )

    compatibility = Compatibility(
        n_loops=len(fused_loops),
        storage=fzs(compatibility_reservations),
    )
    return compatibility


@dataclass
class Job:
    job_id: int
    mapping: Mapping | None = None
    constraints: MappingConstraints | None = None
    tensor2compatibilties: dict[TensorName, set[Compatibility]] | None = None
    tensor2boundless_compatibilities: dict[TensorName, set[Compatibility]] | None = None
    except_from_imperfect: set = frozenset()
    _compatibility: Compatibility | None = None

    @property
    def compatibility(self) -> Compatibility:
        if self._compatibility is None:
            with_reservations = quick_insert_reservation_nodes(
                self.mapping, self.spec.workload
            )
            self._compatibility = make_compatibility(
                with_reservations,
                self.intermediate_tensors,
                self.spec.workload,
            )
        return self._compatibility


@dataclass
class ListOfJobs:
    """
    Contains a list of jobs in `self.jobs_list`.
    
    Other attributes are common to all jobs.
    """
    jobs_list: list[Job]

    tagger: Callable[[Mapping], Tags]
    spec: Specification
    metrics: metrics.Metrics
    rank_variable_bounds: dict[RankVariableName, int]
    flattened_arch: list[architecture.Leaf] | None = None


@dataclass
class JobsOfSingleEinsum(ListOfJobs):
    """Jobs of a single Einsum."""
    intermediate_tensors: set[TensorName] | None = None
    einsum_name: EinsumName | None = None


@dataclass
class JobsWithSimilarCompatibility(JobsOfSingleEinsum):
    """
    Jobs with compatibilities that are identical before
    populating tile shape.
    """
    compatibility: Compatibility
    intermediate_tensors: set[TensorName]