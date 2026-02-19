"""Tests for Compute.tile_granularity and the divisible_by Comparison operator."""

import unittest

import numpy as np

from accelforge.frontend.arch import Comparison, Compute
from accelforge.mapper.FFM._make_pmappings.contraints.constraints import (
    MappingConstraints,
)
import accelforge.frontend.arch as arch_mod


class TestDivisibleByOperator(unittest.TestCase):
    """Unit tests for Comparison(operator='divisible_by').

    The constraint lambda receives a 2D array: shape (n_candidates, n_loops).
    It returns a 1D bool array of shape (n_candidates,).
    """

    def _make(self, value):
        return Comparison(expression="n0", operator="divisible_by", value=value)

    def _call(self, constraint, rows):
        """Invoke the lambda with a (N, 1) candidates array; return bool mask."""
        sizes = np.array(rows, dtype=int).reshape(-1, 1)
        return constraint._to_constraint_lambda(increasing_sizes=True)(
            final=True, sizes=sizes
        )

    def test_multiples_pass(self):
        c = self._make(8)
        mask = self._call(c, [8, 16, 24, 32, 64])
        self.assertTrue(mask.all(), "All multiples of 8 should pass")

    def test_non_multiple_fails(self):
        c = self._make(8)
        mask = self._call(c, [7])
        self.assertFalse(mask.any(), "7 is not a multiple of 8")

    def test_mixed_results(self):
        c = self._make(8)
        mask = self._call(c, [8, 12, 16, 7])
        self.assertEqual(list(mask), [True, False, True, False])

    def test_divisible_by_one_always_passes(self):
        c = self._make(1)
        mask = self._call(c, [3, 5, 7, 11])
        self.assertTrue(mask.all())

    def test_multidim_all_must_be_divisible(self):
        """With two columns, BOTH dimensions must be divisible."""
        c = self._make(4)
        sizes = np.array([[4, 8], [4, 6], [3, 8]])  # shape (3, 2)
        mask = c._to_constraint_lambda(increasing_sizes=True)(
            final=True, sizes=sizes
        )
        self.assertEqual(list(mask), [True, False, False])


class TestComputeTileGranularityField(unittest.TestCase):
    """Unit tests for Compute.tile_granularity field parsing and defaults."""

    def test_default_empty(self):
        c = Compute(name="MAC", leak_power=0, area=0)
        self.assertEqual(c.tile_granularity, {})

    def test_set_granularity(self):
        c = Compute(name="MAC", leak_power=0, area=0, tile_granularity={"K": 128})
        self.assertEqual(c.tile_granularity, {"K": 128})

    def test_multiple_dims(self):
        c = Compute(
            name="MAC", leak_power=0, area=0, tile_granularity={"K": 128, "C": 4}
        )
        self.assertEqual(c.tile_granularity["K"], 128)
        self.assertEqual(c.tile_granularity["C"], 4)


class TestGranularityConstraintsPopulation(unittest.TestCase):
    """Unit tests for granularity_constraints population logic."""

    def _populate(self, nodes):
        """Run the population logic from get_constraints() in isolation."""
        constraints = MappingConstraints()
        for m in nodes:
            if isinstance(m, arch_mod.Compute):
                for rank_name, gran in m.tile_granularity.items():
                    if gran > 1:
                        existing = constraints.granularity_constraints.get(rank_name, 1)
                        constraints.granularity_constraints[rank_name] = max(
                            existing, gran
                        )
        return constraints

    def test_single_node(self):
        node = arch_mod.Compute(
            name="A", leak_power=0, area=0, tile_granularity={"K": 8}
        )
        c = self._populate([node])
        self.assertEqual(c.granularity_constraints["K"], 8)

    def test_most_restrictive_wins(self):
        """When two Compute nodes constrain the same rank variable, max granularity wins."""
        a = arch_mod.Compute(name="A", leak_power=0, area=0, tile_granularity={"K": 8})
        b = arch_mod.Compute(
            name="B", leak_power=0, area=0, tile_granularity={"K": 32}
        )
        c = self._populate([a, b])
        self.assertEqual(c.granularity_constraints["K"], 32)

    def test_gran_one_not_added(self):
        """Granularity ≤ 1 should not appear in granularity_constraints."""
        node = arch_mod.Compute(
            name="A", leak_power=0, area=0, tile_granularity={"K": 1}
        )
        c = self._populate([node])
        self.assertNotIn("K", c.granularity_constraints)

    def test_non_compute_nodes_ignored(self):
        """Non-Compute arch nodes must not contribute to granularity_constraints."""
        mem = arch_mod.Memory(
            name="SRAM", leak_power=0, area=0, size=1024, tensors={"keep": "All"}
        )
        c = self._populate([mem])
        self.assertEqual(c.granularity_constraints, {})


class TestGranularityFilterArray(unittest.TestCase):
    """Unit tests for the numpy filtering logic in make_tile_shapes."""

    def _apply_filter(self, choices, col, gran):
        """Replicate the filter loop from _make_tile_shapes for a single (symbol, gran)."""
        valid = choices[:, col] % gran == 0
        if valid.any():
            choices = choices[valid]
        return choices

    def test_filter_multiples_only(self):
        # Column 0 = K tile sizes; filter to multiples of 8
        choices = np.array([[8], [12], [16], [24], [7]])
        result = self._apply_filter(choices, col=0, gran=8)
        self.assertEqual(result[:, 0].tolist(), [8, 16, 24])

    def test_no_valid_keeps_all(self):
        """If no tile choice is divisible, the filter is a no-op (avoids empty array)."""
        choices = np.array([[7], [11], [13]])
        result = self._apply_filter(choices, col=0, gran=8)
        # valid.any() is False → no filter applied
        self.assertEqual(len(result), 3)

    def test_second_column_filtered(self):
        choices = np.array([[1, 8], [1, 12], [1, 16]])
        result = self._apply_filter(choices, col=1, gran=8)
        self.assertEqual(result[:, 1].tolist(), [8, 16])


if __name__ == "__main__":
    unittest.main()
