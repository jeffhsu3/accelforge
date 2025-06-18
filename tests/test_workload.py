from pathlib import Path
import unittest

from fastfusion.frontend import Workload
from fastfusion.frontend.workload import *


class TestWorkload(unittest.TestCase):
    def test_mha(self):
        workload = Workload.from_yaml(Path(__file__).parent / 'mha.yaml')

        TENSOR_NAMES_REF = {'I', 'WV', 'V', 'WK', 'K', 'WQ', 'Q', 'QK', 'AV',
                            'WZ', 'Z', 'WFFA', 'FFA', 'WFFB', 'FFB' }
        self.assertEqual(TENSOR_NAMES_REF, {t for t in workload.tensor_names})

        rank_variable_bounds = get_rank_variable_bounds(workload, 'Q')
        REF_RANK_VARIABLE_BOUNDS = {
            'm': 2, 'b': 2, 'h': 2, 'e': 2, 'd': 4
        }
        self.assertEqual(REF_RANK_VARIABLE_BOUNDS, rank_variable_bounds)

        REF_TENSOR_Q_SIZE = 16  # B*H*M*E
        self.assertEqual(REF_TENSOR_Q_SIZE,
                         get_tensor_size(workload, TensorName('Q')))

    def test_conv(self):
        workload = Workload.from_yaml(Path(__file__).parent / 'mobilenet.workload.yaml')
        self.assertEqual(
            workload.get_pairwise_equivalent_rank_variables(),
            {
                'pb0': {'pa0', 'pb0'},
                'qb0': {'qa0', 'qb0'},
                't0': {'t0'},
                'pa0': {'r0', 'pb0'},
                'qa0': {'qb0', 's0'},
                'r0': {'pa0'},
                's0': {'qa0'}
            }
        )

    def test_conv_initial_deltas(self):
        workload = Workload.from_yaml(Path(__file__).parent / 'mobilenet_long.workload.yaml')

        initial_deltas = get_stride_and_halo(workload)
        print(initial_deltas)