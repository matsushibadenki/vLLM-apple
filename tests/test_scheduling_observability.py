import unittest

from vllm_apple.execution import ExecutionBackend
from vllm_apple.scheduling_observability import SchedulingObservability
from vllm_apple.service import RuntimeService


class SchedulingObservabilityTests(unittest.TestCase):
    def test_fixed_cardinality_and_saturating_counters(self):
        metrics = SchedulingObservability()
        metrics.assignment(ExecutionBackend.CPU)
        for wait in (0, 1_000_000, 10_000_000, 100_000_000):
            metrics.queue_wait(wait)
        metrics.fallback(2, exhausted=True)
        metrics.contention_rejection()
        metrics.steal()
        metrics.adaptive_transition("deferred")
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["assignments"]["cpu"], 1)
        self.assertEqual(set(snapshot["queue_wait_buckets"].values()), {1})
        self.assertEqual(snapshot["fallback_attempts"], 2)
        self.assertEqual(snapshot["fallback_exhausted"], 1)
        self.assertEqual(snapshot["contention_rejections"], 1)
        self.assertEqual(snapshot["steals"], 1)
        self.assertEqual(snapshot["adaptive_transitions"]["deferred"], 1)
        self.assertEqual(SchedulingObservability._increment(2_147_483_647), 2_147_483_647)
        with self.assertRaises(ValueError):
            metrics.adaptive_transition("secret-arbitrary-status")
        with self.assertRaises(ValueError):
            metrics.queue_wait(-1)

    def test_runtime_snapshot_exposes_only_bounded_aggregate_keys(self):
        snapshot = RuntimeService().snapshot().to_dict()["scheduling_observability"]
        self.assertEqual(set(snapshot), {
            "assignments", "queue_wait_buckets", "fallback_attempts",
            "fallback_exhausted", "contention_rejections", "steals",
            "adaptive_transitions", "adaptive_policy",
        })
        self.assertEqual(snapshot["adaptive_policy"]["level"], 0)
        self.assertNotIn("request_id", repr(snapshot))
        self.assertNotIn("operator", repr(snapshot))


if __name__ == "__main__":
    unittest.main()
