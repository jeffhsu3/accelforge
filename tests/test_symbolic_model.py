import unittest
from pathlib import Path
from pprint import pp

from fastfusion.frontend.mapping import Mapping
from fastfusion.frontend.workload import Workload
from fastfusion.model.looptree.accesses import buffer_accesses_from_buffet_actions
from fastfusion.model.looptree.energy import gather_actions, compute_energy_from_actions
from fastfusion.model.looptree.reuse.summarized.symbolic_new import analyze_reuse


class TestSymbolicModel(unittest.TestCase):
    def test_q_mapping(self):
        mapping = Mapping.from_yaml(Path(__file__).parent / 'Q_mapping.yaml')
        workload = Workload.from_yaml(Path(__file__).parent / 'mha.yaml')

        result = analyze_reuse(mapping, workload)

        self.assertEqual(result.per_einsum_ops[('MAC', 'Q')], 64)

    def test_conv_mapping(self):
        mapping = Mapping.from_yaml(Path(__file__).parent / 'conv.mapping.yaml')
        workload = Workload.from_yaml(Path(__file__).parent / 'conv.workload.yaml')

        result = analyze_reuse(mapping, workload)

        self.assertEqual(result.per_einsum_ops[('MAC', 'conv')], 120)


class TestSymbolicAccesses(unittest.TestCase):
    def test_q_mapping(self):
        mapping = Mapping.from_yaml(Path(__file__).parent / 'Q_mapping.yaml')
        workload = Workload.from_yaml(Path(__file__).parent / 'mha.yaml')

        result = analyze_reuse(mapping, workload)

        accesses = buffer_accesses_from_buffet_actions(result, mapping, workload, is_path=True)

        main_memory_I_accesses= accesses.get_accesses('MainMemory', 'I', 'Q')
        self.assertEqual(main_memory_I_accesses.total_reads, 64)
        self.assertEqual(main_memory_I_accesses.total_writes, 0)
        self.assertEqual(main_memory_I_accesses.max_per_unit_reads, 64)
        self.assertEqual(main_memory_I_accesses.max_per_unit_writes, 0)

        local_buffer_I_accesses= accesses.get_accesses('LocalBuffer', 'I', 'Q')
        self.assertEqual(local_buffer_I_accesses.total_reads, 64)
        self.assertEqual(local_buffer_I_accesses.total_writes, 64)
        self.assertEqual(local_buffer_I_accesses.max_per_unit_reads, 64)
        self.assertEqual(local_buffer_I_accesses.max_per_unit_writes, 16)

        local_buffer_Q_accesses= accesses.get_accesses('LocalBuffer', 'Q', 'Q')
        self.assertEqual(local_buffer_Q_accesses.total_reads, 64)
        self.assertEqual(local_buffer_Q_accesses.total_writes, 128)
        self.assertEqual(local_buffer_Q_accesses.max_per_unit_reads, 16)
        self.assertEqual(local_buffer_Q_accesses.max_per_unit_writes, 32)

class TestSymbolicActions(unittest.TestCase):
    def test_q_mapping(self):
        mapping = Mapping.from_yaml(Path(__file__).parent / 'Q_mapping.yaml')
        workload = Workload.from_yaml(Path(__file__).parent / 'mha.yaml')

        result = analyze_reuse(mapping, workload)
        actions = gather_actions(result, mapping, workload, None, is_path=True, use_name=True)

        self.assertEqual(actions[('LocalBuffer', 'read')], 128)
        self.assertEqual(actions[('LocalBuffer', 'write')], 192)

        self.assertEqual(actions[('MAC', 'compute')], 64)