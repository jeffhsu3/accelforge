import os
import pickle
from fastfusion import Specification
from fastfusion.mapper.FFM.exploration.mapper_multi_einsum import get_sims
from fastfusion.mapper.FFM.exploration.mapper_one_einsum import get_single_einsum_jobs
from fastfusion.mapper.FFM.joining.simexplore import compress, decompress, join_sims

spec = Specification.from_yaml(
    "architecture/four_level.arch.yaml",
    "workloads/mha_full.workload.yaml",
    "workloads/mha_full.renames.yaml",
)

# spec = Specification.from_yaml(
#     "architecture/snowcat.arch.yaml",
#     "workloads/matmuls8.workload.yaml",
# )


workload = spec.workload
renames = spec.renames

# pr = cProfile.Profile()
# pr.enable()

# sims = get_single_einsum_jobs(spec, "Q", rank_variable_bounds)

def cache(filename):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if os.path.exists(filename):
                return pickle.load(open(filename, "rb"))
            else:
                result = func(*args, **kwargs)
                pickle.dump(result, open(filename, "wb"))
                return result
        return wrapper
    return decorator

# @cache("sims.pkl")
def get_sims_with_cache():
    spec.estimate_energy_area()
    flattened_architecture = spec.get_flattened_architecture()
    sims = get_sims(spec, flattened_architecture)#, pkl_cache="sims.pkl")
    decompress_data = compress(sims)
    return sims, decompress_data, flattened_architecture

sims, decompress_data, flattened_architecture = get_sims_with_cache()
mappings = join_sims(sims, spec, flattened_architecture, drop_valid_reservations=False)
decompress(decompress_data, mappings, spec.workload.einsum_names)

# pr.disable()
# s = io.StringIO()
# ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
# ps.print_stats(30)  # Print top 30 time-consuming functions
# print(s.getvalue())

# TODO: Check for ranks not in the mapping and put them at the bottom
# TODO: What if there are no loops? 
# TODO: Set _must_exist for all backing storage nodes
# TODO: Constraint attacher
# TODO: Can't have tile size constraints on backing memory
# TODO: Einsum orders
# TODO: Copy Einsums
# TODO: Test dataflow constraints and order of storage nodes
# I'm doing the tile shape exploration now and I'm trying to understand this note. I think I understand what you're saying.
# Can I ask one thing from the constraint code? If the constraint is an equality, then just set the tile_shape attribute of the node (or factor or whatever is needed) to the value.
# The tile shape exploration assumes a particular mapspace (in most cases, tile shapes are factors of the full rank shape), so an equality may never be satisfied. E.g., if the constraint sets the tile shape equal to a non-factor value because you want a particular imperfect factorization, but that's never in the mapspace, then you'll get nothing.
# It's also a bit more efficient to just set the value and the explorer doesn't have to figure out the equality by trial-and-error. For other more complicated constraints, trial-and-error is better.