import copy
from pydantic import BaseModel, Discriminator, Tag, model_validator
from fastfusion.frontend import architecture
from typing import Callable, List, Union, Annotated, Literal, TypeVar, TypeAlias
from abc import ABC
from fastfusion.util.basetypes import ParsableModel, ParsableList, ParsesTo, get_tag
from fastfusion.version import assert_version, __version__

from fastfusion.frontend import architecture
from fastfusion.frontend.workload.workload import RankVariableName, TensorName
from typing import Callable, Iterator, List, Optional, Type, TypeVar, Union, Annotated
from abc import ABC
from fastfusion.util.basetypes import ParsableModel, ParsableList, ParsesTo, InferFromTag
from fastfusion.version import assert_version, __version__

T = TypeVar("T")

node_list: TypeAlias = ParsableList[Annotated[
        Union[
            Annotated["Split", Tag("Split")],
            Annotated["Compute", Tag("Compute")],
            Annotated["Storage", Tag("Storage")],
            Annotated["Temporal", Tag("Temporal")],
            Annotated["Spatial", Tag("Spatial")],
            Annotated["Sequential", Tag("Sequential")],
            Annotated["Pipeline", Tag("Pipeline")],
            Annotated["Nested", Tag("Nested")],
        ], 
    Discriminator(get_tag)
]]

# =============================================================================
# LoopTree Mapping Nodes
# =============================================================================

class MappingNode(ParsableModel, ABC):
    _constraint_lambdas: List[Callable[[], bool]] = []
    _must_be_here: bool = False  # Can the mapper move this node?
    _required: bool = False  # Must the mapper keep this node?
    # children: ParsableList["MappingNode"] = ParsableList()

    def _mermaid_graph_node_name(self) -> str:
        return f"{self.__class__.__name__}_{id(self)}"
    
    def _mermaid_graph_node_label(self) -> str:
        return f"[\"{self.__class__.__name__}\"]"

    def _mermaid_graph_node(self) -> str:
        return f"{self._mermaid_graph_node_name()}{self._mermaid_graph_node_label()}"
    
    def _parent2next(self) -> "MappingNode":
        return self
    
    def _parent2child(self, parent: "MappingNode") -> list[tuple["MappingNode", "MappingNode"]]:
        return []
    
    def _mermaid_make_children(self) -> list[str]:
        return []

class Pattern(ParsableModel):
    stride: ParsesTo[Literal['symbol'] | int]
    initial_tile_shape: ParsesTo[Literal['symbol'] | int | None] = None
    tile_shape: ParsesTo[Literal['symbol'] | int | None] = None


class Iteration(MappingNode):
    rank_variable: Union[set[RankVariableName], RankVariableName]
    loop_bound: ParsesTo[Union[Literal['symbol'], int, None]] = None
    tile_shape: ParsesTo[Union[Literal['symbol'], int, None]] = None
    tile_pattern: ParsesTo[Union[Pattern, None]] = None
    assume_perfect_factor: bool = True
    _fused: bool = False

    # @model_validator(mode='after')
    # def check_at_least_one_tiling_info(self):
    #     n_non_none = sum([
    #         self.loop_bound is not None,
    #         self.tile_shape is not None,
    #         self.tile_pattern is not None
    #     ])
    #     if n_non_none != 1:
    #         raise ValueError('Must give exactly one of loop_bound, tile_shape, or tile_pattern')
    #     return self
    
    def __str__(self) -> str:
        if self.tile_shape is not None:
            return f"for {self.rank_variable} shape {self.tile_shape}"
        elif self.tile_pattern is not None:
            return f"for {self.rank_variable} pattern {self.tile_pattern}"
        else:
            return f"for {self.rank_variable} in [0..{self.loop_bound})"

class Temporal(Iteration):
    def compact_string(self) -> str:
        return f"{self.rank_variable}-{self.loop_bound}"
    
    def _mermaid_graph_node_label(self) -> str:
        return f"[\"{str(self)}\"]"

class Spatial(Iteration):
    dimension: Union[int, str]
    across: str
    across_object: Optional[architecture.Leaf] = None

    def compact_string(self) -> str:
        return f"S{self.dimension}-{self.rank_variable}-{self.loop_bound}"

    def __str__(self) -> str:
        return f"S{self.dimension}" + super().__str__()

    def _mermaid_graph_node_label(self) -> str:
        return f"[\"{str(self)}\"]"

class Storage(MappingNode):
    tensors: ParsableList[TensorName]
    memory: str
    memory_object: Optional[architecture.Memory] = None # Reference to memory node for convenience
    _must_keep_tensors: ParsableList[TensorName] = ParsableList() # Must the mapper keep these tensors here?
    _backing: bool = False  # Is this node a backing storage?

    def compact_string(self) -> str:
        tname = ",".join(self.tensors)
        return f"[{self.memory} {tname}]"
    
    def __str__(self) -> str:
        tname = ", ".join(self.tensors)
        return f"{tname} in {self.memory}"
    
    def _mermaid_graph_node_label(self) -> str:
        tensors = ", ".join(self.tensors)
        return f"[(\"{tensors} in {self.memory}\")]"
    
    @property
    def tensor(self) -> TensorName:
        if len(self.tensors) != 1:
            raise ValueError(
                f"Storage node {repr(self)} has {len(self.tensors)} tensors. "
                f"Access the tensors property instead."
            )
        return self.tensors[0]



class Compute(MappingNode):
    einsum: str
    compute: str = "MAC"

    def compact_string(self) -> str:
        return f"Einsum {self.einsum}"
    
    def _mermaid_graph_node_label(self) -> str:
        return f"([\"Einsum {self.einsum}\"])"

class MappingNodeWithChildren(MappingNode):
    nodes: node_list = ParsableList()

    def _parent2child(self, parent: MappingNode) -> list[tuple[MappingNode, MappingNode]]:
        mine = [(self, node) for node in self.nodes]
        for child in self.nodes:
            mine.extend(child._parent2child(self))
        return mine

    def _parent2next(self) -> MappingNode:
        return None
    
    def _mermaid_make_children(self) -> list[str]:
        lines = []
        for child in self.nodes:
            lines.append(child._mermaid_graph_node())
            lines.extend(child._mermaid_make_children())
        return lines


class Split(MappingNodeWithChildren):
    pass

    def _mermaid_graph_node_label(self) -> str:
        return f"{{{{\"Split\"}}}}"

class Nested(MappingNodeWithChildren):
    def _parent2child(self, parent: MappingNode) -> list[tuple[MappingNode, MappingNode]]:
        parent2child = []
        for node in self.nodes:
            parent2child.append((parent, node))
            parent2child.extend(node._parent2child(parent))
            parent = node._parent2next()
        return parent2child
    
    def _parent2next(self) -> MappingNode:
        if not self.nodes:
            raise ValueError("Nested node has no children")
        return self.nodes[-1]._parent2next()
    
    # def _mermaid_connect_children(self, names_lines: list[tuple[str, str]], parent_name: str=None) -> list[str]:
    #     return super()._mermaid_connect_children(names_lines)
    
    def _mermaid_graph_node_label(self) -> str:
        if not self.nodes:
            raise ValueError("Nested node has no children")
        return self.nodes[0]._mermaid_graph_node_label()
    
    def _mermaid_graph_node_name(self) -> str:
        if not self.nodes:
            raise ValueError("Nested node has no children")
        return self.nodes[0]._mermaid_graph_node_name()
    
    
class Pipeline(Split):
    pass


class Sequential(Split):
    pass

# =============================================================================
# Nodes That May Only be Inserted by the Model
# =============================================================================

class ModelOnlyNode:
    pass

class Reservation(MappingNode, ModelOnlyNode):
    tensor: str
    memory: str


class Fill(MappingNode, ModelOnlyNode):
    tensor: str
    memory: str


# =============================================================================
# Top-level Mapping
# =============================================================================

MappingNodeTypes: TypeAlias = Union[
    Temporal,
    Spatial,
    Storage,
    Pipeline,
    Sequential,
    Compute,
    Reservation,
    Fill
]


class Mapping(Nested):
    version: Annotated[str, assert_version] = __version__

    def get_fused_slice(self, intermediate_tensors: set[TensorName]) -> "Mapping":
        """
        Return a mapping with:
        - All backing storage nodes for intermediate tensors
        - Loop nodes above any backing storage nodes
        """
        fused_slice = Mapping(nodes=[])
        seen_tensors = set()
        to_add = []
        for node in self.nodes:
            new_node = copy.deepcopy(node)
            if isinstance(new_node, Storage):
                tensors = set(new_node.tensors) & intermediate_tensors
                if tensors - seen_tensors:
                    fused_slice.nodes.extend(to_add + [new_node])
                    to_add = []
                    seen_tensors.update(tensors)
            else:
                to_add.append(new_node)
        return fused_slice
    
    @property
    def loops(self) -> list[Iteration]:
        return [node for node in self.nodes if isinstance(node, Iteration)]
    
    def _mermaid_graph_node_label(self) -> str:
        return f"Root"
    
    def mermaid_graph(self) -> str:
        import mermaid as md
        from mermaid.graph import Graph
        lines = []
        lines = [
            "graph TD",
            "%%{init: {'flowchart': {'nodeSpacing': 30, 'rankSpacing': 30, 'padding': 2}, 'themeVariables': {'fontFamily': 'Arial, sans-serif'}}}%%"
        ]
        lines.extend(self._mermaid_make_children())
        for parent, child in self._parent2child(None):
            if parent is not None:
                lines.append(f"{parent._mermaid_graph_node_name()} --> {child._mermaid_graph_node_name()}")
            # if _is_root:
        #     lines.extend([
        #         "",
        #         "classDef default fill:#fff,stroke:#000,stroke-width:1px,color:#000,font-family:Arial,font-size:12px,padding:2px;",
        #         "classDef compact fill:#fff,stroke:#000,stroke-width:1px,color:#000,font-family:Arial,font-size:12px,padding:2px;"
        #     ])

        # Create the graph with the flowchart script
        flowchart_script = "\n".join(lines)
        graph = Graph('Flowchart', flowchart_script)
        
        # Set the configuration for compact layout
        config = md.Config()
        config.theme = 'base'
        # config.theme_variables = {
        #     'primaryColor': '#ffffff',
        #     'primaryTextColor': '#000000', 
        #     'primaryBorderColor': '#000000',
        #     'lineColor': '#000000',
        #     'fontSize': '12px'
        # }
        # config.flowchart = {
        #     'nodeSpacing': 20,
        #     'rankSpacing': 10,
        #     'curve': 'linear'
        # }
        graph.config = config

        return md.Mermaid(graph)

class MappingTree(MappingNode): # TODO: Make this a full mapping
    version: Annotated[str, assert_version] = __version__

Split.model_rebuild()
Nested.model_rebuild()
Mapping.model_rebuild()