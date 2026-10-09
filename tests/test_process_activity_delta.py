import unittest

from vllm_apple.process_activity_delta import activity_delta, memory_activity_delta


class ActivityDeltaTests(unittest.TestCase):
    def sample(self):
        return dict(available=True, source='Darwin PROC_PIDTASKINFO', pid=123,
                    faults=10, pageins=2, cow_faults=3, context_switches=40, resident_size=100)

    def test_cumulative_delta_allows_rss_reduction(self):
        before = self.sample()
        after = dict(before, faults=14, resident_size=80)
        result = activity_delta(before, after)
        self.assertTrue(result['available'])
        self.assertEqual(result['counters']['faults'], 4)
        self.assertEqual(result['counters']['pageins'], 0)
        self.assertEqual(result['rss_change_bytes'], -20)

    def test_missing_reset_wrong_pid_or_source_is_not_zero_activity(self):
        before = self.sample()
        for after in (None, {}, dict(before, available=False), dict(before, pid=124),
                      dict(before, pid=True), dict(before, source='other'), dict(before, pageins=1),
                      dict(before, faults=True), dict(before, resident_size=-1)):
            with self.subTest(after=after):
                result = activity_delta(before, after)
                self.assertFalse(result['available'])
                self.assertNotIn('counters', result)


    def test_allocator_and_rss_are_independent_changes(self):
        before = {'process_activity': self.sample(), 'allocator': {'active_bytes': 10, 'cache_bytes': 20}}
        after = {'process_activity': dict(self.sample(), resident_size=80),
                 'allocator': {'active_bytes': 15, 'cache_bytes': 18}}
        result = memory_activity_delta(before, after)
        self.assertEqual(result['rss_change_bytes'], -20)
        self.assertEqual(result['allocator']['change_bytes'], {'active_bytes': 5, 'cache_bytes': -2})
        after.pop('allocator')
        result = memory_activity_delta(before, after)
        self.assertTrue(result['available'])
        self.assertFalse(result['allocator']['available'])
        self.assertNotIn('change_bytes', result['allocator'])
