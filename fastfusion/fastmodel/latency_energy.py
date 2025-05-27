from collections import defaultdict
from functools import reduce
from operator import mul

from pytimeloop.looptree.energy import gather_actions, get_accesses


def calculate_latency_and_energy(result,
                                 reads,
                                 writes ,
                                 bandwidth_dict,
                                 energy_dict,
                                 compute_latency,
                                 mapping,
                                 workload,
                                 bindings):
    actions = gather_actions(
        result, {"type": "fused", "nodes": mapping}, workload, bindings, is_path=True
    )
    accesses = defaultdict(lambda: 0)
    reads, writes = get_accesses(
        result, {"type": "fused", "nodes": mapping}, workload, is_path=True
    )
    for k, v in reads.items():
        accesses[k] += v
    for k, v in writes.items():
        accesses[k] += v

    energy = sum(
        energy_dict[comp_action] * counts for comp_action, counts in actions.items()
    )

    memory_latency = max(
        (
            sum(
                read_count
                for (this_level, _, _), read_count in reads.items()
                if this_level == level
            )
            + sum(
                write_count
                for (this_level, _, _), write_count in writes.items()
                if this_level == level
            )
        )
        / (bandwidth_dict[level] * reduce(mul, fanout, 1))
        for level, fanout in result.fanout.items()
        if level in bandwidth_dict
    )

    return energy, max(compute_latency, memory_latency), memory_latency, compute_latency