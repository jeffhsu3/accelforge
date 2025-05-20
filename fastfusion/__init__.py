from fastfusion.yamlparse import Node
from fastfusion.frontend.specification import Specification
import fastfusion.frontend.arch as arch
import fastfusion.frontend.area_table as area_table
import fastfusion.frontend.components as components
import fastfusion.frontend.config as config
import fastfusion.frontend.constraints as constraints
import fastfusion.frontend.energy_table as energy_table
import fastfusion.frontend.mapping as mapping
import fastfusion.frontend.renames as renames
import fastfusion.frontend.specification as specification
import fastfusion.frontend.variables as variables
import fastfusion.frontend.version as version
import fastfusion.frontend.workload as workload

for d in Node._needs_declare_attrs:
    if hasattr(d, "declare_attrs"):
        d.declare_attrs(d)
