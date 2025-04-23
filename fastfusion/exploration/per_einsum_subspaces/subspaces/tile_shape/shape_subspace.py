from operator import le, lt, ge, gt, eq
import re

from combinatorics.integer import integer_factorizations_to_n_parts


CONSTRAINT_RE_PARSER = re.compile('^(>=|>|<|<=|==)\s*(\d+)')


STRING_TO_OPERATOR = {
    '==': eq,
    '>=': ge,
    '>':  gt,
    '<=': le,
    '<':  lt
}


def get_propagated_constraint(comparison: str, limit):
    if comparison == '>=':
        return lambda x: x >= limit
    elif comparison == '>':
        return lambda x: x > limit
    elif comparison == '<=':
        return None
    elif comparison == '<':
        return None
    elif comparison == '==':
        return lambda x: x >= limit


def parse_tile_constraint(constraint_str: str):
    match = CONSTRAINT_RE_PARSER.match(constraint_str)
    if match is None:
        raise RuntimeError(f'Cannot parse constraint {constraint_str}')

    comparison, limit = match.groups()

    limit = int(limit)
    if limit == 128:
        comparison = "=="

    if comparison in STRING_TO_OPERATOR:
        operator = STRING_TO_OPERATOR[comparison]
        main_constraint = lambda x: operator(x, limit)
    else:
        raise RuntimeError(f'Unknown comparison operator {comparison}')

    propagated_constraint = get_propagated_constraint(comparison, limit)

    return main_constraint, propagated_constraint


def parse_factor_constraint(constraint_str: str):
    match = CONSTRAINT_RE_PARSER.match(constraint_str)
    if match is None:
        raise RuntimeError(f'Cannot parse constraint {constraint_str}')

    comparison, limit = match.groups()

    limit = int(limit)
    if limit == 128:
        comparison = "=="

    if comparison in STRING_TO_OPERATOR:
        operator = STRING_TO_OPERATOR[comparison]
        main_constraint = lambda x: operator(x, limit)
    else:
        raise RuntimeError(f'Unknown comparison operator {comparison}')

    return main_constraint, None


class ShapeSubspace:
    def __init__(self,
                 rank_shapes: dict[int, int],
                 ranks: list[int],
                 tile_constraints: list[list[str]]=None,
                 factor_constraints: list[list[str]]=None,
                 n_fusion_relevant_loops: int=None):
        self.rank_shapes = rank_shapes
        self.ranks = ranks
        
        self.position_to_last = {}
        self.n_fusion_relevant_loops = n_fusion_relevant_loops
        self.fill_position_to_last()
        # print(f'Made shape subspace with tile constraints {self.tile_constraints} and factor constraints {self.factor_constraints}')

        self.parse_tile_constraints(self, tile_constraints)
        self.parse_factor_constraints(self, factor_constraints)

    def fill_position_to_last(self):
        self.rank_to_last = {}
        for i, r in enumerate(self.ranks):
            if r in self.rank_to_last:
                self.position_to_last[i] = self.rank_to_last[r]
            else:
                self.position_to_last[i] = None
            self.rank_to_last[r] = i

    def parse_tile_constraints(self, tile_constraints):
        self.tile_constraints = [[] for _ in self.ranks]
        if tile_constraints is not None:
            for i, constraints in enumerate(tile_constraints):
                for constraint in constraints:
                    constraint, upward_prop_constraint = parse_tile_constraint(constraint)
                    self.add_tile_constraint(i, constraint, upward_prop_constraint)

    def parse_factor_constraints(self, factor_constraints):
        self.factor_constraints = [[] for _ in self.ranks]
        if factor_constraints is not None:
            for i, constraints in enumerate(factor_constraints):
                for constraint in constraints:
                    constraint, upward_prop_constraint = parse_factor_constraint(constraint)
                    self.add_factor_constraint(i, constraint, upward_prop_constraint)

    def add_tile_constraint(self, rank_idx, constraint, propagated_constraint):
        self.tile_constraints[rank_idx].append(constraint)
        if propagated_constraint is not None:
            last = self.position_to_last[rank_idx]
            while last is not None:
                self.tile_constraints[last].append(propagated_constraint)
                last = self.position_to_last[last]

    def add_factor_constraint(self, rank_idx, constraint, propagated_constraint):
        self.factor_constraints[rank_idx].append(constraint)
        if propagated_constraint is not None:
            last = self.position_to_last[rank_idx]
            while last is not None:
                self.factor_constraints[last].append(propagated_constraint)
                last = self.position_to_last[last]

def check_add_to_pareto(a, pareto):
    for b in pareto:
        if all(b[k] <= a[k] for k in a):
            return False
    return True

class SkippableDfsIterator:
    def __init__(self, shape_subspace: ShapeSubspace):
        self.shapes = shape_subspace.rank_shapes
        self.ranks = shape_subspace.ranks
        self.choice_generators = self.make_choice_generators(shape_subspace)
        self.pos_to_last = shape_subspace.position_to_last
        self.tile_constraints = shape_subspace.tile_constraints
        self.factor_constraints = shape_subspace.factor_constraints
        self.n_fusion_relevant_loops = shape_subspace.n_fusion_relevant_loops
        if self.n_fusion_relevant_loops is None:
            self.n_fusion_relevant_loops = len(self.choice_generators)

        self.is_started = False
        self.is_done = False
        self.just_skipped = False

        self.choice_iterators = None
        self.choice = None
        self.is_first_choice = None
        
        self.paretos = None
        self.prev_paretos = None
        self.would_have_skipped = None

    def _add_pareto_point(self, idx, point):
        if check_add_to_pareto(point, self.paretos[idx]):
            if self.would_have_skipped[idx]:
                print(f'Would have skipped rank {idx}. Prev pareto size: {len(self.prev_paretos[idx])}, new pareto size: {len(self.paretos[idx])}')
                to_print = []
                to_print.append(f'Skipping rank {idx}. Prev pareto size: {len(self.prev_paretos[idx])}, new pareto size: {len(self.paretos[idx])}')
                for r in self.paretos[idx]:
                    to_print.append(f'\tNEW: {r}')
                for r in self.prev_paretos[idx]:
                    to_print.append(f'\tOLD: {r}')
                print('\n'.join(to_print))
                self.would_have_skipped[idx] = True
            self.paretos[idx].append(point)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.is_started:
            self.is_started = True
            self.initialize_choice_iterators()
        elif not self.just_skipped:
            self.is_done = True
            for idx in range(len(self.choice_iterators) - 1, -1, -1):
                try:
                    if idx > 0:
                        for r in self.paretos[idx]:
                            self._add_pareto_point(idx - 1, r)
                    self.move_iterator(idx)
                    self.prev_idx = idx
                    break
                except StopIteration as e:
                    pass
            if len(self.choice_iterators) == 0:
                idx = 0
            for j in range(idx+1, len(self.choice_iterators)):
                self.restart_iterator(j)
        else:
            self.just_skipped = False

        if self.is_done:
            raise StopIteration()

        return self.choice.copy()

    def skip_current_rank_iteration(self, chain_skip_if_first=True):
        if self.is_done:
            return

        skip_limit = 1
        if chain_skip_if_first:
            for i in range(len(self.choice_iterators)):
                idx = len(self.choice_iterators)-i-1
                if not self.is_first_choice[idx]:
                    break
            if len(self.choice_iterators) == 0:
                skip_limit = 0
            else:
                skip_limit = i+1

        if skip_limit == len(self.choice_iterators):
            self.is_done = True
            self.just_skipped = True
            return

        self.is_done = True
        for i in range(skip_limit, len(self.choice_iterators)):
            idx = len(self.choice_iterators)-i-1
            try:
                self.move_iterator(idx)
                self.prev_idx = idx
                break
            except StopIteration as e:
                self.restart_iterator(idx)
        for j in range(idx+1, len(self.choice_iterators)):
            self.restart_iterator(j)

        self.just_skipped = True

    def register_result(self, is_pareto, result):
        if len(self.choice_iterators) > 0:
            self._add_pareto_point(-1, result)

    def make_choice_generators(self, shape_subspace: ShapeSubspace):
        choice_generators = []

        def gen(shape, tile_constraints, factor_constraints):
            if shape == 1:
                res = [(1, 1)]
            else:
                res = list(integer_factorizations_to_n_parts(shape, 2))[:-1]
            res = [i for i in res if all(c(i[0]) for c in tile_constraints)]
            res = [i for i in res if all(c(i[1]) for c in factor_constraints)]
            res = [i[0] for i in res]
            return res

        for _ in shape_subspace.ranks:
            choice_generators.append(gen)
        return choice_generators

    def initialize_choice_iterators(self):
        self.choice_iterators = [None]*len(self.choice_generators)
        self.choice = [None]*len(self.choice_generators)
        self.is_first_choice = [None]*len(self.choice_generators)
        self.paretos = [None]*len(self.choice_generators)
        self.prev_paretos = [None]*len(self.choice_generators)
        self.would_have_skipped = [False]*len(self.choice_generators)
        for i in range(len(self.choice_generators)):
            try:
                self.restart_iterator(i)
            except StopIteration as e:
                pass
        self.prev_idx = len(self.choice_iterators) - 1

    def restart_iterator(self, idx):
        last = self.pos_to_last[idx]
        if last is None:
            shape = self.shapes[self.ranks[idx]]
        else:
            shape = self.choice[last]
        self.choice_iterators[idx] = \
            iter(self.choice_generators[idx](shape,
                                             self.tile_constraints[idx],
                                             self.factor_constraints[idx]))
        self.is_first_choice[idx] = True
        self.prev_paretos[idx] = None
        self.would_have_skipped[idx] = False
        self.paretos[idx] = []
        try:
            self.choice[idx] = next(self.choice_iterators[idx])
        except StopIteration as e:
            self.choice[idx] = 1
            self.choice_iterators[idx] = iter([])
            raise StopIteration()

    def move_iterator(self, idx):
        val = next(self.choice_iterators[idx])
        # If none of the new pareto points are better than the previous pareto points, then we can stop
        # if self.paretos[idx] and self.prev_paretos[idx] and idx > self.n_fusion_relevant_loops: # Remove self.paretos[idx] and it'll stop if the first iteration of all inner loops fails.
        #     if not any(check_add_to_pareto(r, self.prev_paretos[idx]) for r in self.paretos[idx]):
        #         # print(f'Skipping rank {idx}. Prev pareto size: {len(self.prev_paretos[idx])}, new pareto size: {len(self.paretos[idx])}')
        #         # to_print = []
        #         # to_print.append(f'Skipping rank {idx}. Prev pareto size: {len(self.prev_paretos[idx])}, new pareto size: {len(self.paretos[idx])}')
        #         # for r in self.paretos[idx]:
        #         #     to_print.append(f'\tNEW: {r}')
        #         # for r in self.prev_paretos[idx]:
        #         #     to_print.append(f'\tOLD: {r}')
        #         # print('\n'.join(to_print))
        #         # self.would_have_skipped[idx] = True
        #         raise StopIteration()
        self.choice[idx] = val
        self.is_first_choice[idx] = False
        self.is_done = False
        self.prev_paretos[idx] = self.paretos[idx]
        self.paretos[idx] = []
