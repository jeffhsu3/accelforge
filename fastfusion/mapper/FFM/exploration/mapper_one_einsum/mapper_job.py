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
from fastfusion.frontend.mapping import Reservation as ReservationNode


def make_compatibility(
    mapping: Mapping,
    intermediate_tensors: set[TensorName],
) -> Compatibility:
    fused_slice = mapping.get_fused_slice(intermediate_tensors)
    fused_loops = []
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

    compatibility_loops = []
    for loop in fused_loops:
        if loop.tile_shape is not None:
            bound = 0  # populated later, but type is important
        elif loop.tile_pattern is not None:
            bound = TilePattern(0, 0)
        else:
            raise RuntimeError("BUG")

        loop = Loop(
            rank_variable_names=fzs((loop.rank_variable,)),
            bound=bound,
            is_spatial=isinstance(loop, Spatial),
        )
        compatibility_loops.append(loop)
    compatibility_reservations = []
    for above_loop_index, reservation_nodes in reservations.items():
        for reservation in reservation_nodes:
            compatibility_reservations.append(
                Reservation(
                    name=reservation.tensor,
                    above_loop_index=above_loop_index,
                    resource_name=reservation.memory,
                    size=0,  # TODO: Get size
                )
            )

    compatibility = Compatibility(
        loops=tuple(compatibility_loops),
        storage=fzs(compatibility_reservations),
    )
    return compatibility


@dataclass
class Job:
    spec: Specification
    tagger: Callable[[Mapping], Tags]
    metrics: metrics.Metrics
    job_id: int
    rank_variable_bounds: dict[RankVariableName, int]
    mapping: Mapping | None = None
    constraints: MappingConstraints | None = None
    intermediate_tensors: set[TensorName] | None = None
    flattened_arch: list[architecture.Leaf] | None = None
    einsum_name: EinsumName | None = None
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
                with_reservations, self.intermediate_tensors
            )
        return self._compatibility
