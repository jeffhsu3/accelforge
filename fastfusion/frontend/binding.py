from abc import abstractmethod
from typing import Tuple

from pydantic import BaseModel, StrictFloat, model_validator
import islpy as isl

from fastfusion.util.basetypes import ParsableDict, ParsableList, ParsableModel

class Domain(BaseModel):
    """
    Represents an architecture dangling reference of the binding.
    """
    name: str

    @abstractmethod
    def get_isl_space(self):
        raise NotImplementedError(f"{type(self)} has not implemented get_isl_space")

class LogicalDomain(Domain):
    """
    Represents the logical architecture domain space of logical dims × ranks.
    """
    ranks: Tuple[str] = (
        'C', 'H', 'W', 'P', 'Q', 'R', 'S'
    )
    dims: Tuple[str, ...]

    def get_isl_space(self):
        return isl.Space.create_from_names(
            isl.DEFAULT_CONTEXT,
            in_=self.ranks,
            out=self.dims
        )

class PhysicalDomain(Domain):
    """
    Represents the logical architecture domain space of physical dims.
    """
    p_dims: Tuple[str, ...]

    def get_isl_space(self):
        return isl.Space.create_from_names(
            isl.DEFAULT_CONTEXT,
            set=self.p_dims
        )

class BindingNode(BaseModel):
    """
    How a logical architecture is implemented on a particular physical architecture
    for a particular hardware level. Represents a injection relation between points 
    in logical to physical space.
    
    The logical space is defined as logical architecture dims × tensor dims.
    The physical space is defined as physical architecture dims × tensor dims.
    """
    logical: LogicalDomain
    physical: PhysicalDomain
    relations: ParsableDict[str, str]

class Binding(BaseModel):
    """
    A collection of binding nodes that fully specifies a relation between the
    logical and physical space.
    """
    version: StrictFloat
    nodes: ParsableList[BindingNode]

# now loads YAML
import yaml
yaml_str: str = """
binding:
  version: 0.4
  nodes:
  - logical:
      name: PE
      dims: [i]
    physical: 
      name: PE
      p_dims: [x, y]
    relations:
      tensorA: i = x + y * 2 # This is a dimension-major compression into the logical. It is bijective.
      tensorB: i = x + y * 2 # This is a dimension-major compression into the logical. It is bijective.
"""

binding = Binding.model_validate(yaml.safe_load(yaml_str)['binding'])
print(binding)
print(binding.nodes[0].logical.get_isl_space())
print(binding.nodes[0].physical.get_isl_space())