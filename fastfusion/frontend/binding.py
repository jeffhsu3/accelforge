import pyisl as isl

from pydantic_core import CoreSchema
from fastfusion.util.basetypes import ParsableDict, ParsableList, ParsableModel
from fastfusion.util.setexpressions import InvertibleSet, eval_set_expression
from fastfusion.frontend.renames import Renames
from fastfusion.version import assert_version, __version__
from typing import Annotated, Any, Callable
from pydantic_core.core_schema import CoreSchema, chain_schema, no_info_plain_validator_function

class BindingRelation(ParsableDict):
    """
    Represents a bijective relation between points in logical to physical space.
    The logical space is defined as logical architecture dims × tensor dims.
    The physical space is defined as physical architecture dims × tensor dims.
    """
    logical_space: isl.Space
    physical_space: isl.Space

some_dictionary = some_yaml_library.parse(your_yaml_file)
# Some dictionary is going to have str for logical_space and physical_space
BindingRelation.build_model(some_dictionary)
# This build_model will validate the types of values in some_dictionary

# This is using a library called Pydantic.
# There may be "type adapter" that tells the library to call a function when
# validating. In this case, we want to call isl.Space ctor.

class Binding(ParsableModel):
    """
    How a logical architecture is implemented on a particular physical architecture.
    """
    name: str
    logical_model = None
    physical_model = None
    nodes: ParsableList[BindingRelation] = ParsableList()

