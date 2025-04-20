from collections import defaultdict
import copy
import itertools
import re

# Disable numba. We need user_has_package("numba") to be False
import sys
from typing import Iterable, Optional, Tuple, Union

from joblib import delayed

from fastfusion.joining.mappinginfo import TensorStorage
from fastfusion.util import fzs

sys.modules["numba"] = None


from paretoset import paretoset

import pandas as pd
import functools

LOGSTRING = "__Mappings"
MAPPING = "__LOOPNEST"
STATS = "__STATS"
OCCUPANCY = "__Occupancy"
TENSORS = "__TENSORS"
IN_PROGRESS_STATS = "__IN_PROGRESS_STATS"
MAPPING_HASH = "__MAPPING_HASH"
TAGS = "__TAGS"
VALID = "__VALID"
PER_COMPONENT_ACCESSES_ENERGY = "Per-Component Energy"

DICT_COLUMNS = set(
    [
        LOGSTRING,
        MAPPING,
        STATS,
        TENSORS,
        IN_PROGRESS_STATS,
        MAPPING_HASH,
        TAGS,
        PER_COMPONENT_ACCESSES_ENERGY,
    ]
)
RESERVED_COLUMNS = set([VALID]) | DICT_COLUMNS

TUPLABE_COLUMNS = set([MAPPING, TENSORS])

CHECK_CORRECTNESS = True

_resource_name_nloops_reg = re.compile(r"RESOURCE_(.+?)(?:_LEFT)?_LEVEL_(-?\d+)")


def dict_cached(func):
    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, fzs(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return wrapper


# TODO: Make these tuples?


@dict_cached
def col2nameloop(x):
    m = _resource_name_nloops_reg.match(x)
    return (m.group(1), int(m.group(2))) if m is not None else None


@dict_cached
def nameloop2col(name, nloops, left: bool = False):
    if left:
        return f"RESOURCE_{name}_LEFT_LEVEL_{nloops}"
    return f"RESOURCE_{name}_LEVEL_{nloops}"

@dict_cached
def col2nameloopleft(x):
    m = _resource_name_nloops_reg.match(x)
    return (m.group(1), int(m.group(2)), is_left_col(x)) if m is not None else None


@dict_cached
def is_left_col(x):
    return "_LEFT_LEVEL_" in x


MERGE_SUFFIXES = ["", "_RIGHT_MERGE"]


def is_merge_col(c):
    return any(c.endswith(s) for s in MERGE_SUFFIXES)


def add_to_col(df, target, source):
    if target in df:
        df.loc[:, target] = df[target] + df[source]
    else:
        df.loc[:, target] = df[source]


def max_to_col(df, target, source):
    if target in df:
        df.loc[:, target] = df[[target, source]].max(axis=1)
    else:
        df.loc[:, target] = df[source]


def is_special_col(c):
    return c in RESERVED_COLUMNS or col2nameloop(c) is not None


# Above index 0: Freed when Einsum fully terminates
# Above index 1: Freed after each iteration of the outermost loop

# -1 -> global resource
# 0 -> einsum only

# Shared index -1: Sum -1 resources, max everyone below
# Shared index 0: Sum 0 resources, max everyone below


def quick_pareto(df):
    df["chosen"] = False
    i = 0
    while i != len(df):
        i += 1
        # Pick the entry which is not chosen & has the lowest first columns
        idx = df.loc[~df["chosen"], df.columns[0]].idxmin()
        df.loc[idx, "chosen"] = True
        dominated = (df.iloc[:, :-1] >= df.loc[idx, df.columns[:-1]]).all(axis=1)
        df = df[~dominated]
    return df.drop(columns=["chosen"])


def makepareto(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        c for c in data.columns if c not in RESERVED_COLUMNS and not is_merge_col(c)
    ]
    if len(data) == 1:
        return data

    return data[paretoset(data[columns])].reset_index(drop=True)

class Pareto:
    def __init__(
            self, 
            data: pd.DataFrame, 
            skip_pareto: bool = False, 
            fill_reservation_cols: set | str = fzs(),
            check_above_subset_below: bool = CHECK_CORRECTNESS,
            max_right_to_left: bool = False,
        ):
        self._data: pd.DataFrame = data
        self.right_reservations: dict[set] = None
        self.left_reservations: dict[set] = None
        self.parents = []
        self._prev_free_to_loop_index = None
        self._make_reservations()
        
        if fill_reservation_cols: # Affects Pareto so must go before
            self.fill_reservation_cols(fill_reservation_cols)
        if check_above_subset_below:
            self.check_above_subset_below()
        if max_right_to_left: # Affects Pareto so must go before
            self.max_right_to_left()
        if check_above_subset_below:
            self.check_above_subset_below()

        if not skip_pareto:
            self.make_pareto()
            
        if check_above_subset_below:
            self.check_above_subset_below()

            
    def fill_reservation_cols(self, columns: set | str):
        targets = []
        if columns == "auto":
            for left, reservations_dict in [
                (True, self.left_reservations),
                (False, self.right_reservations),
            ]:
                for resource, reservations in reservations_dict.items():
                    for r in sorted(reservations):
                        above = self.get_reservation_or_parent(resource, r - 1)
                        if above is not None:
                            below = nameloop2col(resource, r, left=left)
                            targets.append((r, above, below))
        else:
            for below in columns:
                if (name_nloops := col2nameloop(below)) is None:
                    raise ValueError(f"{below} is not a valid reservation column")
                name, nloops = name_nloops
                above = self.get_reservation_or_parent(name, nloops - 1)
                if above is not None:
                    targets.append((nloops, above, below))

        # Sort so we go from top to bottom. Needed in case we have to max 0->1
        # then 1->2
        for _, above, below in sorted(targets, key=lambda x: x[0]):
            assert above in self.data.columns
            assert below in self.data.columns
            max_to_col(self.data, below, above)
            
    def max_right_to_left(self):
        for resource, reservations in self.left_reservations.items():
            for r in reservations:
                source = self.get_reservation_or_parent(resource, r)
                if source is None:
                    continue
                target = nameloop2col(resource, r, left=True)
                max_to_col(self.data, target, source)

    @property
    def data(self) -> pd.DataFrame:
        return self._data
        
    def _make_reservations(self) -> Tuple[dict[str, list], dict[str, list]]:
        """
        Create a dictionary of reservations for each resource.
        The dictionary keys are the resource names and the values are lists
        of column names for each loop index.
        """
        self.left_reservations, self.right_reservations = {}, {}
        for c in self.data.columns:
            if (name_nloops := col2nameloop(c)) is not None:
                name, nloops = name_nloops
                target = self.left_reservations if is_left_col(c) else self.right_reservations
                target.setdefault(name, set()).add(nloops)
                assert nloops >= 0
                
    def free_to_loop_index(
            self, 
            loop_index: int, 
            live_tensors: set[int] = None,
            check_correctness=CHECK_CORRECTNESS,
        ) -> bool:
        """
           A  B
            / | --- 0  
           C  D
            / | --- 1  < Shared Loop Index
           E  F
            / | --- 2
           G  H
        ->
           A  B
            / | --- 0
           C  D
              | --- 1  < Shared Loop Index
          max(E,G,H)
        We skip incorporating E into the max because its reservations are
        already incorporated into F and G.
        """
        if loop_index == self._prev_free_to_loop_index:
            return False
        self._prev_free_to_loop_index = loop_index
        
        if check_correctness:
            before = self.copy()

        drop_columns = []
        for resource in set(self.left_reservations) | set(self.right_reservations):
            max_columns = []
            left_reservations = self.left_reservations.get(resource, set())
            right_reservations = self.right_reservations.get(resource, set())
            left_big_enough = [l for l in left_reservations if l >= loop_index + 1]
            right_big_enough = [r for r in right_reservations if r >= loop_index + 2] # + 1 is target

            if len(right_big_enough) > 1: # All ones above the last are subsets
                right_biggest = max(right_big_enough)
                right_big_enough.remove(right_biggest)
                drop_columns += [nameloop2col(resource, r) for r in right_big_enough]
                right_big_enough = [right_biggest]

            max_columns = [nameloop2col(resource, r) for r in right_big_enough] + [nameloop2col(resource, l, left=True) for l in left_big_enough]

            if not max_columns:
                continue

            target = nameloop2col(resource, loop_index + 1)
            if target in self.data:
                max_columns.append(target)

            if len(max_columns) == 1:
                self.data.rename(columns={max_columns[0]: target}, inplace=True)
            else:
                for c in max_columns:
                    max_to_col(self.data, target, c)
                drop_columns += [m for m in max_columns if m != target]
        self.data.drop(columns=drop_columns, inplace=True)
        self._make_reservations()

        if check_correctness and live_tensors is not None:
            self.copy().check_reservations(live_tensors=live_tensors)

        return len(drop_columns) != 0
    
    def get_reservation_or_parent(
            self, 
            name: str, 
            level: int, 
            left: bool = False,
            return_name_level_left: bool = False,
        ) -> Optional[Union[str, Tuple[str, int, bool]]]:
        reservations = self.left_reservations if left else self.right_reservations
        if name not in reservations:
            return None
        reservations = reservations[name]
        while level >= 0:
            if level in reservations:
                if return_name_level_left:
                    return name, level, left
                return nameloop2col(name, level, left)
            # The parent of left nodes are right nodes, so if we don't find a
            # left node immediately then we're back on the right nodes
            reservations = self.right_reservations.get(name, set())
            left = False
            level -= 1
        return None

    def shift_bottom_reservation_left(self, shared_loop_index: int):
        for resource in self.right_reservations:
            if shared_loop_index + 1 not in self.right_reservations[resource]:
                continue
            self.left_reservations.setdefault(resource, set())
            self.right_reservations[resource].remove(shared_loop_index + 1)
            self.left_reservations[resource].add(shared_loop_index + 1)
            source = nameloop2col(resource, shared_loop_index + 1)
            target = nameloop2col(resource, shared_loop_index + 1, left=True)
            if target in self.data:
                max_to_col(self.data, target, source)
                self.data.drop(columns=[source], inplace=True)
            else:
                self.data.rename(columns={source: target}, inplace=True)
    
    @staticmethod
    def _get_target_path(suffix: str = None) -> str:
        import os
        f = "./images"
        os.makedirs(f, exist_ok=True)
        suffix = "" if suffix is None else f".{suffix}"
        i = 0
        while os.path.exists(os.path.join(f, f"test_{i}{suffix}.png")):
            i += 1
        assert i <= 50, "Too many images"
        return os.path.join(f, f"test_{i}{suffix}.png")
    
    def merge_next(
        self,
        right: "Pareto",
        shared_loop_index: int,
        live_tensors: set[int],
        shared_storage: set[TensorStorage],
        still_live_reservations: set[TensorStorage],
        duplicated_aliased_tensors: set[TensorStorage],
        resource2capacity: dict[str, int] = None,
        _raise_exceptions: bool = False,
    ) -> "Pareto":
        """
            A  B            A2
             / | --- 0      |
            C  D            C2
               | --- 1      |     < Shared Loop Index
               E            E2
                            |
                            F2
            -> 
            A  A+A2
             / | --- 0
         C+A2  C+C2
             / | --- 1  < Shared Loop Index
         E+C2  E2+D
               |
               F2+D
        """
        left = self
        prev_right = right
        self = copy.deepcopy(self)
        right = copy.deepcopy(right)

        # # These tensors will be re-added by the next Einsum and we don't want to
        # # double count them
        # for t in shared_storage:
        #     self.free(t)
        
        try:
            self.free_to_loop_index(shared_loop_index, live_tensors=live_tensors)
        except Exception as e:
            print(e)
            self.parents[0].merge_next(*self.parents[1:], _raise_exceptions=True).check_reservations(self.parents[3])
            raise e
        self.shift_bottom_reservation_left(shared_loop_index)
        self._make_reservations()
        right._make_reservations()
        
        # if duplicated_aliased_tensors:
        #     right = right.copy()
        #     right.free(duplicated_aliased_tensors)

        assert not right.left_reservations, f"{right.left_reservations} is not None"

        for resource, reservations in self.right_reservations.items():
            n_reservations = max(reservations, default=-1)
            assert n_reservations <= shared_loop_index, f"{resource}: {reservations} > {shared_loop_index}"

        for resource, reservations in self.left_reservations.items():
            n_reservations = max(reservations, default=-1)
            assert n_reservations <= shared_loop_index + 1, f"{resource}: {reservations} > {shared_loop_index}"
            
        max_nloops = max(
            shared_loop_index,
            max((max(r, default=-1) for r in self.right_reservations.values()), default=-1),
            max((max(r, default=-1) for r in self.left_reservations.values()), default=-1),
            max((max(r, default=-1) for r in right.right_reservations.values()), default=-1),
            max((max(r, default=-1) for r in right.left_reservations.values()), default=-1),
        )

        df = pd.merge(self.data, right.data, how="cross", suffixes=MERGE_SUFFIXES)

        # Make sure everything is done in increasing loop order so we don't have
        # read-after-write hazards
        for nloops in range(max_nloops, -1, -1):
            def iter_reservations(reservations_dict):
                for resource in reservations_dict:
                    if nloops in reservations_dict[resource]:
                        yield resource
            
            # For the RIGHT tree, RIGHT reservations: If there is no matching node in the left
            # tree, add the above-this-level reservation from the left tree. If there is a matching
            # node in the left tree, then we'll add this node to it in the next step.
            for resource in iter_reservations(right.right_reservations):
                if (source := self.get_reservation_or_parent(resource, nloops - 1)) is None:
                    continue
                target = nameloop2col(resource, nloops)
                # If there's a merged version column, then it's in both trees
                if target + "_RIGHT_MERGE" in df:
                    continue
                df.loc[:, target] += df[source]
            # For LEFT tree, LEFT reservations: Add the immediately-above
            # reservation from the right tree.
            for resource in iter_reservations(self.left_reservations):
                if (source := right.get_reservation_or_parent(resource, nloops - 1)) is None:
                    continue
                right_merge_source = source + "_RIGHT_MERGE"
                target = nameloop2col(resource, nloops, left=True)
                if source is not None:
                    df.loc[:, target] += df[right_merge_source if right_merge_source in df else source]
            # For LEFT tree, RIGHT reservations: Add the same-level reservation from
            # the right tree. This will double-count reservations that are in both branches,
            # so we remove them later.
            for resource in iter_reservations(self.right_reservations):
                if (source := right.get_reservation_or_parent(resource, nloops)) is None:
                    continue
                right_merge_source = source + "_RIGHT_MERGE"
                target = nameloop2col(resource, nloops)
                if source is not None:
                    df.loc[:, target] += df[right_merge_source if right_merge_source in df else source]

        # For everything else: Simple add
        dropcols = [c for c in df.columns if c.endswith("_RIGHT_MERGE")]
        for source in dropcols:
            target = source[:-len("_RIGHT_MERGE")]
            if col2nameloop(target) is not None:
                continue
            if target in DICT_COLUMNS:
                df[target] = (
                    df.apply(lambda row: {**row[target], **row[source]}, axis=1)
                    if len(df) > 0
                    else []
                )
            else:
                df.loc[:, target] += df[source]
        df = df.drop(columns=dropcols)
        try:
            result = Pareto(df, skip_pareto=True, check_above_subset_below=False)
            if not CHECK_CORRECTNESS:
                result.limit_capacity(resource2capacity)
            # result._draw_index(0, live_tensors, self._get_target_path())
            result.check_above_subset_below(live_tensors)
            # Remove tensors that were allocated in both branches and got added
            # together. We can do this after pareto calculation because it affects
            # all mappings equally.
            # print(f"Freeing {[s for s in shared_storage if s.above_loop_index <= shared_loop_index]}")
            # print(f"Freeing {list(duplicated_aliased_tensors)}")
            # print(f"Allocating {[s for s in still_live_reservations if s.above_loop_index > shared_loop_index]}")
            
            shared_to_free = [s for s in shared_storage if s.above_loop_index <= shared_loop_index]
            live_to_alloc = [ s for s in still_live_reservations if s.above_loop_index > shared_loop_index]
            # self.adjust_reservations(
            #     alloc=live_to_alloc,
            #     free=itertools.chain(shared_to_free, duplicated_aliased_tensors),
            # )
            
            result.free((s for s in shared_storage if s.above_loop_index <= shared_loop_index))
            result.free(duplicated_aliased_tensors)
            result.alloc(s for s in still_live_reservations if s.above_loop_index > shared_loop_index)
            result.max_right_to_left()
            if not CHECK_CORRECTNESS:
                result.make_pareto()

        except Exception as e:
            if _raise_exceptions:
                raise e
            print(e)
            self.merge_next(
                right,
                shared_loop_index,
                live_tensors,
                shared_storage,
                still_live_reservations,
                duplicated_aliased_tensors,
                _raise_exceptions=True,
            )


        if CHECK_CORRECTNESS:
            try:
                result.check_above_subset_below(live_tensors)
                result.check_reservations(live_tensors)
                # result.parents = [self, right, shared_loop_index, live_tensors, shared_storage, still_live_reservations, duplicated_aliased_tensors]
            except Exception as e:
                if _raise_exceptions:
                    raise e
                # raise e
                print(e)
                # result.free(shared_storage)
                # x = self.parents[0].merge_next(*self.parents[1:])
                # y = x.merge_next(right, shared_loop_index, live_tensors, shared_storage)
                left.merge_next(
                    prev_right,
                    shared_loop_index,
                    live_tensors,
                    shared_storage=shared_storage,
                    still_live_reservations=still_live_reservations,
                    duplicated_aliased_tensors=duplicated_aliased_tensors,
                    _raise_exceptions=True,
                )
            result.make_pareto()
        
        return result
    
    def _draw_index(self, index: int, live_tensors, to_file: str = "test.png"):
        from fastfusion.visualization.reservationtree import mappings2reservationtree
        import pydot
        looptree = mappings2reservationtree(
            self.data.iloc[index][MAPPING],
            self.data.iloc[index].get(STATS, None),
            still_live_tensors=live_tensors,
        )
        graph = pydot.Dot(graph_type="digraph", ranksep="0.2", nodesep="0.2")
        looptree.to_pydot(graph)
        row = self.data.iloc[index]
        all_data = sorted(
            f"{k}: {v}" for k, v in row.items() if k not in DICT_COLUMNS and k != LOGSTRING
        )
        data_str = "\n".join(all_data)
        graph.add_node(pydot.Node("data", label=data_str, shape="plaintext"))
        with open(to_file, "wb") as f:
            f.write(graph.create_png())
    
    def _adjust_reservations_one_resource(
        self,
        resource: str,
        alloc: Iterable[TensorStorage],
        free: Iterable[TensorStorage],
    ):
        # Iterate through each reservation and level
        targets = defaultdict(int)
        for tensors, negate in [
            (alloc, False),
            (free, True),
        ]:
            if not tensors:
                continue
                
            cur_tensors = [t for t in tensors if t.resource_name == resource]
            if not cur_tensors:
                continue
                
            # We also allocate at any levels below the above_loop_index level
            for t in cur_tensors:
                targets[t.above_loop_index, False] += -t.size if negate else t.size
                for level in self.right_reservations[resource]:
                    if level > t.above_loop_index:
                        targets[level, False] += -t.size if negate else t.size
                for level in self.left_reservations.get(resource, set()):
                    if level > t.above_loop_index:
                        targets[level, True] += -t.size if negate else t.size
                        
        # Now apply the allocations. Sort so we go from top to bottom in case
        # there are maxes that propagate down.
        for (level, left), size in sorted(targets.items(), key=lambda x: x[0], reverse=True):
            target = nameloop2col(resource, level, left=left)
            if target in self.data:
                self.data.loc[:, target] += size
                continue

            # We're creating a new column, so copy allocations from any parents
            source = self.get_reservation_or_parent(resource, level-1)
            # source is None -> We're at the top level, no one to inherit from
            if source is not None:
                size = self.data[source] + size
            self.data[target] = size
        
    def adjust_reservations(
            self,
            alloc: Iterable[TensorStorage],
            free: Iterable[TensorStorage],
        ):
        # Get all unique resources
        all_resources = set()
        for tensors in [alloc, free]:
            all_resources.update(t.resource_name for t in tensors)
            
        # Handle each resource separately
        for resource in all_resources:
            cur_alloc = [t for t in alloc if t.resource_name == resource]
            cur_free = [t for t in free if t.resource_name == resource]
            self._adjust_reservations_one_resource(resource, cur_alloc, cur_free)
                
    def fail(self, index, live_tensors):
        from fastfusion.visualization.reservationtree import mappings2reservationtree
        from fastfusion.joining.sim import TensorStorage
        r = self.data.iloc[index]
        assert not self.data.isnull().values.any(), f"NaN in {self.data}"
        self = self.copy()
        self._draw_index(index, live_tensors, self._get_target_path(suffix="fail"))
        all_tensors = set(t for tn in r[MAPPING].values() for t in tn.storage)
        all_tensors = TensorStorage.get_backing_stores(all_tensors)
        for t in sorted(all_tensors):
            print(f"{t.__repr__()},")

        # Pipeline:
        # - Need to share temporal loops up to the spatial loop index
        #   Resources:
        #   - Energy
        #   - PE utilization
        #   - Buf utilization
        #   - Buf accesses (for BW calculation later)

        # - Options:
        #   - Non-pipelined: Sum resources above shared loops, max below.
        #   - Pipelined: Sum resources above shared loops, max below. Sum
        #     PE utilization. Latency is pipeline latency summed.
        #
        #  *  Can't bake into compatiblity unless we have a notion of left vs.
        #     right pipelined.

        # PIPELINE CHANGES REQUIRED:
        # - Latency above above loop index (first tile), below (all subsequent tiles)
        # - Mapping includes information for how may be fused:
        #   - Pipelined: Max below latencies,
        #   - Non-pipelined:
        # Shared resources:
        # -
        # SEQUENTIAL:
        # - In parallel: Fetch all above-shared-loop resources for all operations
        # - Sequentially: Fetch any below-shared-loop resources for all operations
        # PIPELINE:
        # - In parallel: Fetch all above-shared-loop resources for all operations
        # - Sequentially: Fetch any below-shared-loop resources for the first iteration of all operations
        # - In parallel: Fetch all below-shared-loop resources for all operations in all subsequent iterations

    def add_tensor(self, tensor):
        if len(self.data) == 0:
            return
        if TENSORS in self.data:
            last_einsum = list(self.data.iloc[0][TENSORS].keys())[-1]
            if tensor in self.data[TENSORS].iloc[0][last_einsum]:
                return
            for t in self.data[TENSORS]:
                t[last_einsum].append(tensor)

    def einsum_ids(self):
        return fzs(self.data[LOGSTRING].iloc[0].keys())

    @staticmethod
    def concat(paretos: list["Pareto"], skip_pareto: bool = False) -> "Pareto":
        if len(paretos) == 1:
            return paretos[0]
        
        required_cols = set.union(*[set(p.data.columns) for p in paretos])
        shared_cols = set.intersection(*[set(p.data.columns) for p in paretos])
        fill_cols = required_cols - shared_cols
        
        p = Pareto(
            pd.concat([p.data for p in paretos]).fillna(0),
            skip_pareto=len(paretos) == 1 or skip_pareto,
            fill_reservation_cols=fill_cols,
        )
        p.parents = paretos[0].parents
        return p

    def copy(self) -> "Pareto":
        p = Pareto(self.data.copy(), skip_pareto=True, check_above_subset_below=False)
        p.parents = copy.deepcopy(self.parents)
        return p

    def limit_capacity(self, resource2capacity: dict[str, Optional[int]]) -> bool:
        # TODO: Incorporate next shared loop index
        resource2capacity = resource2capacity or {}
        for resource, capacity in resource2capacity.items():
            if capacity is None:
                continue

            # Right reservations: Only check the greatest-index level
            right_loops = self.right_reservations.get(resource, set())
            if right_loops:
                n = max(right_loops)
                col = nameloop2col(resource, n)
                self.data = self.data[self.data[col] <= capacity]

            # Left reservations: Check all levels. If a loop is 0,
            # then we can drop the column.
            left_loops = self.left_reservations.get(resource, set())
            for l in sorted(left_loops):
                col = nameloop2col(resource, l, left=True)
                if col in self.data:
                    self.data = self.data[self.data[col] <= capacity]
                    if l == 0:
                        del self.data[col]
                        self.left_reservations[resource].discard(l)

    def squish_left_right(self, shared_loop_index: int = None) -> bool:
        needs_pareto = False
        for resource, reservations in self.left_reservations.items():
            if shared_loop_index + 1 not in reservations:
                continue
            self.right_reservations.setdefault(resource, set())
            source = nameloop2col(resource, shared_loop_index + 1, left=True)
            target = nameloop2col(resource, shared_loop_index + 1)
            self.right_reservations[resource].add(shared_loop_index + 1)
            self.left_reservations[resource].remove(shared_loop_index + 1)
            
            if shared_loop_index + 1 in self.right_reservations[resource]:
                max_to_col(self.data, target, source)
                needs_pareto = True
            else:
                self.data.rename(columns={source: target}, inplace=True)
            
        return needs_pareto

    def make_pareto(self):
        if len(self._data) <= 1:
            return
        columns = [
            c for c in self.data.columns if c not in RESERVED_COLUMNS and not is_merge_col(c)
        ]
        self._data = self.data[paretoset(self.data[columns])].reset_index(drop=True)

    def has_reservations(self):
        return any(col2nameloop(c) is not None for c in self.data.columns)


    # ============================================================================
    # Checking functions
    # ============================================================================
    def check_above_subset_below(self, live_tensors: set[str]=fzs()):
        assert not self.data.isnull().values.any(), f"NaN in {self.data}"
        targets = []
        for left, reservations_dict in [
            (True, self.left_reservations),
            (False, self.right_reservations),
        ]:
            for resource, reservations in reservations_dict.items():
                for r in reservations:
                    above = self.get_reservation_or_parent(resource, r - 1)
                    if above is not None:
                        below = nameloop2col(resource, r, left=left)
                        targets.append((above, below))
                
        for above, below in targets:
            if (self.data[below] < self.data[above]).any():
                first_failing_index = (self.data[below] < self.data[above]).idxmax()
                fail_row = self.data.iloc[first_failing_index]
                error = f"""
                {below} column is less than {above} column. A reservation at
                a level should include all reservations above it. There were {len(fail_row)} rows
                with this error. One example: {fail_row}
                """
                self.fail(first_failing_index, live_tensors)
                raise ValueError(error)
            

    def check_reservations(self, live_tensors: set[int]):
        from fastfusion.visualization.reservationtree import mappings2reservationtree
        from fastfusion.joining.sim import TensorStorage
        assert not self.data.isnull().values.any(), f"NaN in {self.data}"

        self = self.copy()

        self.free_to_loop_index(-1, check_correctness=False)
        self.squish_left_right(-1)

        for i, r in self.data.iterrows():
            looptree = mappings2reservationtree(
                r[MAPPING],
                r.get(STATS, None),
                # skip_backing_tensors_in_right_branch=live_tensors,
                still_live_tensors=live_tensors,
            )
            reservations = dict(looptree.get_reservations())
            
            # If r doesn't have any columns, continue. It's a copy Einsum so it has no
            # stats.
            if r.empty:
                continue

            for k, v in reservations.items():
                col = self.get_reservation_or_parent(k, 0)
                if str(k) == "0":
                    continue
                if col not in self.data.columns:
                    got = r[[c for c in self.data.columns if col2nameloop(c) is not None]]
                    self.fail(i, live_tensors)
                    raise ValueError(f"Missing {k}: Expected {reservations}. Got: {got}")
                if r[col] != v:
                    got = r[[c for c in self.data.columns if col2nameloop(c) is not None]]
                    self.fail(i, live_tensors)
                    looptree = mappings2reservationtree(
                        r[MAPPING],
                        r.get(STATS, None),
                        # skip_backing_tensors_in_right_branch=live_tensors,
                        still_live_tensors=live_tensors,
                    )
                    raise ValueError(
                        f"Mismatched {k}: {v} != {r[col]}. Expected {reservations}. Got: {got}"
                    )