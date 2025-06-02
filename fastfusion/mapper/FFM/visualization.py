import copy
import re
from fastfusion.frontend.mapping import Iteration, Mapping, Nested, Sequential, Storage

def make_mapping(row, einsum_names):
    pmappings = []
    for einsum_name in einsum_names:
        pmapping = copy.deepcopy(row[f"{einsum_name}___MAPPING"])
        tile_shape_reg = einsum_name + r"___tile_shape\d+"
        tile_shapes = row[[c for c in row.index if re.match(tile_shape_reg, c)]]
        tile_shapes = list(tile_shapes)
        nodes = [n for n in pmapping.nodes if isinstance(n, (Iteration, Storage))]
        for node in nodes:
            if isinstance(node, Iteration):
                node.tile_shape = tile_shapes.pop(0)
        pmappings.append(Nested(nodes=nodes))
    newmapping = Mapping(nodes=[Sequential(nodes=pmappings)])
    return newmapping
