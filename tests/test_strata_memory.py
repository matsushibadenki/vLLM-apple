import unittest

from vllm_apple.strata_memory import ProcessMemoryBudget


class ProcessBudgetTests(unittest.TestCase):
    def test_overlap_and_cache_and_reserve(self):
        policy = ProcessMemoryBudget(1000, 100)
        values = dict(rss_bytes=400, mlx_active_bytes=300, mlx_cache_bytes=200,
                      resident_bytes=100, available_bytes=1000, requested_bytes=900)
        self.assertEqual(policy.resident_limit(**values), 500)
        values['available_bytes'] = 250
        self.assertEqual(policy.resident_limit(**values), 250)
        values['mlx_cache_bytes'] = 700
        with self.assertRaises(MemoryError):
            policy.resident_limit(**values)

    def test_invalid_observation(self):
        with self.assertRaises(ValueError):
            ProcessMemoryBudget(100, 100)
        with self.assertRaises(ValueError):
            ProcessMemoryBudget(True, 0)
