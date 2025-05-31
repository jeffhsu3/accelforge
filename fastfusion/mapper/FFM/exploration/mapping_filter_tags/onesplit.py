from fastfusion.frontend.mapping import Storage, Iteration


def get_one_split_tag(pmapping, intermediate_tensors, non_fused_memory):
    tensor_to_n_fused_loops = {t: None for t in intermediate_tensors}
    n_loops = 0
    for node in pmapping:
        if isinstance(node, Storage):
            for tensor in node.tensors:
                if (
                    tensor not in intermediate_tensors
                    or tensor in tensor_to_n_fused_loops
                ):
                    continue
                if node.memory == non_fused_memory:
                    tensor_to_n_fused_loops[tensor] = None
                else:
                    tensor_to_n_fused_loops[tensor] = n_loops
        elif isinstance(node, Iteration):
            n_loops += 1

    # Unfused
    if all(t is None for t in tensor_to_n_fused_loops.values()):
        return ("ONE_SPLIT",)

    # Fused with one side but not the other. We don't want to interfere with the
    # unfused side, so just go ONE_SPLIT. The number of loops will be enforced
    # by the tiling since it must match for the one fused tensor.
    if len(tensor_to_n_fused_loops) == 1:
        return ("ONE_SPLIT",)
    
    # Fused with both sides. Make sure that the number of loops is the same.
    unique_loops = set(t for t in tensor_to_n_fused_loops.values() if t is not None)
    if len(unique_loops) > 1:
        return ("NOT_ONE_SPLIT",)
    return ("ONE_SPLIT", f"FUSED_LOOPS={next(iter(unique_loops))}")