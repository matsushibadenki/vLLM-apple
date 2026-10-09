import unittest
from unittest.mock import patch

from scripts.qualify_gemma2_batch_mask import _normal_memory_point


class NormalMemoryPointTests(unittest.TestCase):
    def test_stage_and_workload_match_without_tensor_operations(self):
        snapshot = dict(registry={'active': 0, 'queued': 0}, http={'active': 1},
                        process_activity={'available': True, 'pid': 123},
                        prompt_cache={'accounted_bytes': 10}, allocator={'active_bytes': 20})
        benchmark = dict(workload_sha256='workload', requests=12, quality_passed=12, slo_quality_passed=11)
        with patch('scripts.qualify_gemma2_batch_mask._idle_resources', return_value=snapshot) as read:
            result = _normal_memory_point(19167, benchmark, 123, 'after_long_before_faults')
            self.assertTrue(result['same_worker'])
            self.assertEqual(result['workload_sha256'], 'workload')
            self.assertEqual(result['slo_quality_passed'], 11)
            self.assertEqual(result['prompt_cache']['accounted_bytes'], 10)
            read.assert_called_once_with(19167)
            result = _normal_memory_point(19167, benchmark, 124, 'after_long_before_faults')
            self.assertFalse(result['same_worker'])
            result = _normal_memory_point(19167, benchmark, 123.0, 'after_long_before_faults')
            self.assertFalse(result['same_worker'])
            snapshot['registry']['active'] = 1
            with self.assertRaises(RuntimeError):
                _normal_memory_point(19167, benchmark, 123, 'after_long_before_faults')
