from fastfusion.frontend.specification import Specification
from fastfusion.frontend.workload.workload import EinsumName, RankVariableName
from fastfusion.mapper.FFM.exploration.mapper_one_einsum import get_single_einsum_sims
from fastfusion.mapper.FFM.joining.mappinginfo import Compatibility
from fastfusion.mapper.FFM.joining.sim import SIM

def get_sims(spec: Specification, rank_variable_to_size: dict[RankVariableName, int]) -> dict[EinsumName, list[SIM]]:
    sims = {}
    arch_flattened = spec.get_flattened_architecture()
    for einsum_name in spec.workload.einsum_names:
        sims[einsum_name] = get_single_einsum_sims(spec, einsum_name, rank_variable_to_size, arch_flattened)
    return sims
