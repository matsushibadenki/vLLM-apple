import unittest

from tests.test_expert_residency import Backend
from vllm_apple.expert_execution import ResidentExpertExecutor
from vllm_apple.expert_residency import ExpertResidencyManager


class ExpertExecutionTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()
        self.manager = ExpertResidencyManager(
            self.backend, maximum_entries=2, maximum_bytes=200,
            eviction_policy="cost_frequency",
        )
        self.executor = ResidentExpertExecutor(self.manager, maximum_samples=1)
        self.addCleanup(self.manager.close)

    def run_experts(self, consumer, phase="decode", experts=(0, 1)):
        return self.executor.execute(
            phase=phase, layer=0, selected_experts=experts,
            routing_weights=(0.75, 0.25), consume=consumer,
        )

    def test_router_order_weights_and_phase_hits_preserved(self):
        def consume(resources, weights):
            self.assertEqual(self.manager.snapshot()["active_leases"], 2)
            self.assertFalse(self.manager.resize(maximum_entries=1, maximum_bytes=100))
            return tuple(r.handle for r in resources), weights
        result = self.run_experts(consume, "prefill")
        self.assertEqual(result, (((0, 0), (0, 1)), (0.75, 0.25)))
        self.assertEqual(self.manager.snapshot()["active_leases"], 0)
        self.assertEqual(self.manager.snapshot()["resident_bytes"], 100)
        self.assertEqual(self.executor.telemetry["decode"].snapshot()["samples"], 0)

    def test_failure_releases_all_leases(self):
        def fail(*args):
            raise RuntimeError("execution failed")
        with self.assertRaisesRegex(RuntimeError, "execution failed"):
            self.run_experts(fail)
        self.assertEqual(self.manager.snapshot()["active_leases"], 0)
        self.assertEqual(self.executor.telemetry["decode"].snapshot()["samples"], 0)

    def test_partial_acquisition_failure_releases_leases(self):
        self.manager.resize(maximum_entries=1, maximum_bytes=100)
        with self.assertRaisesRegex(ValueError, "pinned"):
            self.run_experts(lambda *args: self.fail("must not execute"))
        self.assertEqual(self.manager.snapshot()["active_leases"], 0)

    def test_invalid_router_does_not_load(self):
        with self.assertRaises(ValueError):
            self.run_experts(lambda *args: None, experts=(0, 0))
        self.assertEqual(self.backend.loaded, [])

    def test_hits_and_bounded_samples(self):
        for _ in range(3):
            self.run_experts(lambda *args: None)
        snapshot = self.executor.telemetry["decode"].snapshot()
        self.assertEqual(snapshot["cache_hits"], 2)
        self.assertEqual(snapshot["samples"], 1)
        self.assertEqual(snapshot["dropped"], 2)
