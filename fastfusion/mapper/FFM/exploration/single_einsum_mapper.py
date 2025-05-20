from itertools import chain, combinations
import copy

from fastfusion.frontend.mapping import MappingNodeList, Storage, Temporal, Spatial
import fastfusion.frontend.arch as arch
from fastfusion.frontend.arch import Leaf
from fastfusion.frontend._set_parsing import InvertibleSet
from fastfusion.frontend.specification import Specification
from fastfusion.frontend.workload.workload_spec import Einsum, Tensor

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

# =================================================================================================
# Choose what data to store in each memory
# =================================================================================================

def make_storage_choices_one_level(
        node: Leaf,
        symbol_table: dict[str, InvertibleSet],
    ):
    assert "All" in symbol_table
    tensors = symbol_table["All"]

    if not isinstance(node, arch.Memory):
        yield [], symbol_table
        return

    new_symbol_table = copy.copy(symbol_table)
    storage_constraints = node.constraints.storage._parse_keep_bypass(symbol_table)
    must_keep = tensors.to_my_space(storage_constraints["keep"])
    must_bypass = tensors.to_my_space(storage_constraints["bypass"])

    if must_keep - tensors:
        raise KeyError(f"Keep constraint for {node.name} includes tensors that are "
                       f"not in the einsum: {must_keep - new_symbol_table['All']}")
    if must_bypass - tensors:
        raise KeyError(f"Bypass constraint for {node.name} includes tensors that are "
                       f"not in the einsum: {must_bypass - tensors.full_space}")
    if must_keep & must_bypass:
        raise KeyError(f"Keep and bypass constraints for {node.name} intersect: "
                       f"{must_keep & must_bypass}")
    
    may_keep = tensors - must_bypass - must_keep

    for subset in powerset(may_keep):
        # Make keep choice & update symbol table
        subset = tensors.to_my_space(set(subset))
        keep_choice = tensors.to_my_space(subset | must_keep)
        keep_choice.tensors = lambda: keep_choice # So users can do MainMemory().tensors(). Optional.
        new_symbol_table[node.name] = keep_choice
        
        # Make sure they're all tensors
        assert all(isinstance(k, Tensor) for k in keep_choice)
        keep_choice = keep_choice.to_my_space({copy.copy(t) for t in keep_choice})
        storage_nodes = []
        
        # Create storage nodes
        for t in keep_choice:
            storage_nodes.append(Storage(tensor=t, memory=node))
            storage_nodes[-1]._must_exist = t in must_keep
        yield storage_nodes, new_symbol_table

def make_storage_choices_all_levels(
    nodes: list[Storage], 
    symbol_table: dict[str, InvertibleSet],
):
    if len(nodes) == 0:
        yield dict(), symbol_table
        return
    for choice, symbol_table in make_storage_choices_one_level(nodes[0], symbol_table):
        for subchoices, symbol_table in make_storage_choices_all_levels(nodes[1:], symbol_table):
            yield {**subchoices, nodes[0].name: choice}, symbol_table
            

# =================================================================================================
# Order storage nodes (dataflow).
# =================================================================================================

def label_backing_storages(mapping: MappingNodeList):
    seen_tensors = set()
    for _, s in mapping.enumerate_type(Storage):
        s._backing = s.tensor not in seen_tensors
        seen_tensors.add(s.tensor)

def valid_storage_order(mapping: MappingNodeList, node_names: list[str]):
    for i in range(len(mapping)):
        for j in range(i, len(mapping)):
            t1, t2 = mapping[i].tensor, mapping[j].tensor
            if t1.name != t2.name:
                continue
            
            s1, s2 = mapping[i].memory.name, mapping[j].memory.name
            s1_idx, s2_idx = node_names.index(s1), node_names.index(s2)
            
            # If a tensor is stored in two levels back-to-back, then we
            # should have bypassed the outer storage if possible.
            if i == j or i == j - 1:
                if s1_idx < s2_idx and not (mapping[i]._must_exist or mapping[i]._backing):
                    return False
                if s2_idx < s1_idx and not (mapping[j]._must_exist or mapping[j]._backing):
                    return False
                
            # Ensure order
            if i < j and s2_idx < s1_idx:
                return False
    return True

def recursive_order_storage_choices(
    mapping: MappingNodeList,
    nodes: list[arch.Memory],
    remaining_choices: list,
):
    if not remaining_choices:
        yield mapping
        return

    for choice in list(remaining_choices):
        mapping.append(choice)
        remaining_choices.remove(choice)
        label_backing_storages(mapping)
        if valid_storage_order(mapping, [n.name for n in nodes]):
            yield from recursive_order_storage_choices(mapping, nodes, remaining_choices)
        mapping.pop()
        remaining_choices.append(choice)

def get_storage_choices(
    nodes: list[arch.Memory],
    symbol_table: dict[str, InvertibleSet],
):
    while not isinstance(nodes[0], arch.Memory):
        nodes = nodes[1:]
    first_storage = nodes[0]
    
    for choice, symbol_table in make_storage_choices_all_levels(nodes, symbol_table):
        all_storage_nodes = []
        for v in choice.values():
            all_storage_nodes.extend(v)
            
        base_mapping = MappingNodeList()
        for node in list(all_storage_nodes[::-1]):
            if node.memory.name == first_storage.name:
                all_storage_nodes.remove(node)
                base_mapping.append(node)
            
        for mapping in recursive_order_storage_choices(base_mapping, nodes, all_storage_nodes):
            yield mapping, symbol_table

# =================================================================================================
# Insert loops
# =================================================================================================

def insert_temporal_loops(
    mapping: MappingNodeList,
    einsum: Einsum,
    first_memory: arch.Memory,
):
    seen_tensors = set()
    seen_non_first_memory = False
    rank_variables = set(einsum.rank_variables)
    
    i = 0
    while i < len(mapping):
        if mapping[i].memory.name != first_memory.name:
            seen_non_first_memory = True
        if i < len(mapping) - 1 and mapping[i+1].memory.name != first_memory.name:
            seen_non_first_memory = True
        if not seen_non_first_memory:
            i += 1
            continue
        
        rank_vars = set(rank_variables)
        if not mapping[i]._must_be_here:
            rank_vars &= einsum.tensor2rank_variables[mapping[i].tensor]
        if i < len(mapping) - 1:
            if not mapping[i+1]._must_be_here:
                rank_vars &= einsum.tensor2rank_variables[mapping[i+1].tensor]
        seen_tensors.add(mapping[i].tensor)
        for t in einsum.tensors - seen_tensors:
            rank_vars &= einsum.tensor2rank_variables[t]
            
        if rank_vars:
            mapping.insert(i+1, Temporal(rank_variable=rank_vars))
            i += 1
        i += 1
    return mapping

def insert_spatial_loops(
    mapping: MappingNodeList,
    einsum: Einsum,
    arch_flattened: list[arch.Memory],
):
    nodes_with_fanout = [n for n in arch_flattened if n.spatial.get_fanout() > 1]
    arch_node_names = [n.name for n in arch_flattened]
    
    # Place spatials above the last instance of the first memory ABOVE each fanout
    for fanout in nodes_with_fanout:
        insertion_point = 0
        for i in range(len(mapping)):
            if not isinstance(mapping[i], Storage):
                continue
            memory_name = mapping[i].memory.name
            if arch_node_names.index(memory_name) < arch_node_names.index(fanout.name):
                insertion_point = i + 1

        rv = einsum.rank_variables
        if fanout.spatial.fanout_Y > 1:
            mapping.insert(insertion_point, Spatial(rank_variable=rv, dimension="Y"))
        if fanout.spatial.fanout_X > 1:
            mapping.insert(insertion_point, Spatial(rank_variable=rv, dimension="X"))

# =================================================================================================
# Iterate over mappings
# =================================================================================================

def iterate_mappings(
    spec: Specification,
    einsum_names: list[str] | str | None = None,
):
    if isinstance(einsum_names, str):
        einsum_names = [einsum_names]
    if einsum_names is None:
        einsum_names = [e.name for e in spec.workload.einsums]

    arch_flattened = spec.get_flattened_architecture()
    first_memory = None
    for node in arch_flattened:
        if isinstance(node, arch.Memory):
            first_memory = node
            break
    if first_memory is None:
        raise ValueError("No memory found in architecture")

    for einsum_name in einsum_names:
        symbol_table = spec.workload.get_constraint_symbol_table(einsum_name, spec.renames)
        einsum = spec.workload.einsums[einsum_name]
        for mapping, symbol_table in get_storage_choices(arch_flattened, symbol_table):
            mapping = MappingNodeList(mapping)
            insert_temporal_loops(mapping, einsum, first_memory)
            insert_spatial_loops(mapping, einsum, arch_flattened)
            yield mapping
