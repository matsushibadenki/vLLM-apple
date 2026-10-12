import unittest
from unittest.mock import patch

from vllm_apple.expert_residency import (
    ExpertKey,
    ExpertResidencyManager,
    ExpertResource,
)


class Backend:
    def __init__(self):
        self.loaded = []
        self.released = []

    def load_expert(self, key):
        self.loaded.append(key)
        return ExpertResource((key.layer, key.expert), 100)

    def release_expert(self, resource):
        self.released.append(resource.handle)


class ExpertResidencyTests(unittest.TestCase):
    def test_reservation_evicts_before_loading_and_rejects_pinned_without_load(self):
        class ReservedBackend(Backend):
            def resident_bytes_upper_bound(self, key):
                return 100

            def load_expert(self, key):
                self.assert_capacity()
                return super().load_expert(key)

        backend = ReservedBackend()
        manager = ExpertResidencyManager(backend, maximum_entries=1, maximum_bytes=100)
        backend.assert_capacity = lambda: self.assertEqual(manager.snapshot()["resident_bytes"], 0)
        first = manager.acquire(ExpertKey(0, 0))
        with self.assertRaisesRegex(ValueError, "pinned"):
            manager.acquire(ExpertKey(0, 1))
        self.assertEqual(len(backend.loaded), 1)
        first.release()
        manager.acquire(ExpertKey(0, 1)).release()
        self.assertEqual(backend.released, [(0, 0)])
        self.assertEqual(manager.snapshot()["peak_reserved_bytes"], 100)
        manager.close()
        too_small = ExpertResidencyManager(backend, maximum_entries=1, maximum_bytes=99)
        before = len(backend.loaded)
        with self.assertRaisesRegex(ValueError, "insufficient"):
            too_small.acquire(ExpertKey(0, 3))
        self.assertEqual(len(backend.loaded), before)
        too_small.close()

    def test_bound_violation_releases_resource_and_failed_load_remains_recoverable(self):
        class ReservedBackend(Backend):
            def resident_bytes_upper_bound(self, key):
                return 99 if key.expert == 2 else 100

            def load_expert(self, key):
                if key.expert == 1:
                    raise MemoryError("simulated loader failure")
                return super().load_expert(key)

        backend = ReservedBackend()
        manager = ExpertResidencyManager(backend, maximum_entries=1, maximum_bytes=100)
        manager.acquire(ExpertKey(0, 0)).release()
        with self.assertRaises(MemoryError):
            manager.acquire(ExpertKey(0, 1))
        self.assertEqual(manager.snapshot()["resident_bytes"], 0)
        self.assertEqual(manager.snapshot()["active_leases"], 0)
        with self.assertRaisesRegex(ValueError, "backend bound"):
            manager.acquire(ExpertKey(0, 2))
        self.assertIn((0, 2), backend.released)
        manager.acquire(ExpertKey(0, 0)).release()
        manager.close()

    def test_cost_policy_retains_expensive_expert_instead_of_lru(self):
        backend = Backend()
        manager = ExpertResidencyManager(
            backend, maximum_entries=2, maximum_bytes=200,
            eviction_policy="cost_frequency",
        )
        with patch("vllm_apple.expert_residency.time.perf_counter_ns",
                   side_effect=[0, 1000, 1000, 1010, 1010, 1020]):
            for key in (ExpertKey(0, 0), ExpertKey(0, 1), ExpertKey(0, 2)):
                manager.acquire(key).release()
        self.assertEqual(backend.released, [(0, 1)])
        manager.close()

    def test_cost_policy_preserves_pinned_expert_and_deferred_budget(self):
        manager = ExpertResidencyManager(
            Backend(), maximum_entries=2, maximum_bytes=200,
            eviction_policy="cost_frequency",
        )
        first = manager.acquire(ExpertKey(0, 0))
        second = manager.acquire(ExpertKey(0, 1))
        self.assertFalse(manager.resize(maximum_entries=1, maximum_bytes=100))
        with self.assertRaisesRegex(ValueError, "pinned"):
            manager.acquire(ExpertKey(0, 2))
        second.release()
        self.assertEqual(manager.snapshot()["resident_bytes"], 100)
        first.release()
        manager.close()

    def test_unknown_policy_rejected_before_backend_load(self):
        backend = Backend()
        with self.assertRaises(ValueError):
            ExpertResidencyManager(
                backend, maximum_entries=2, maximum_bytes=200,
                eviction_policy="automatic",
            )
        self.assertEqual(backend.loaded, [])

    def test_layer_expert_lru_hits_and_eviction(self):
        backend = Backend()
        manager = ExpertResidencyManager(
            backend, maximum_entries=2, maximum_bytes=200
        )
        first = manager.acquire(ExpertKey(0, 0))
        first.release()
        second = manager.acquire(ExpertKey(0, 1))
        second.release()
        hit = manager.acquire(ExpertKey(0, 0))
        hit.release()
        third = manager.acquire(ExpertKey(1, 0))
        third.release()
        self.assertEqual(backend.released, [(0, 1)])
        self.assertEqual(manager.snapshot()["hits"], 1)
        self.assertEqual(manager.snapshot()["evictions"], 1)
        self.assertEqual(manager.snapshot()["peak_resident_bytes"], 200)
        manager.close()
        self.assertEqual(manager.snapshot()["resident_bytes"], 0)
        self.assertEqual(manager.snapshot()["peak_resident_bytes"], 200)

    def test_active_lease_pins_resource_and_defers_resize(self):
        backend = Backend()
        manager = ExpertResidencyManager(
            backend, maximum_entries=2, maximum_bytes=200
        )
        first = manager.acquire(ExpertKey(0, 0))
        second = manager.acquire(ExpertKey(0, 1))
        with self.assertRaisesRegex(ValueError, "pinned"):
            manager.acquire(ExpertKey(0, 2))
        self.assertFalse(manager.resize(maximum_entries=1, maximum_bytes=100))
        second.release()
        self.assertFalse(manager.snapshot()["resize_pending"])
        self.assertEqual(manager.snapshot()["entries"], 1)
        first.release()
        manager.close()

    def test_invalid_key_and_active_close_fail_closed(self):
        with self.assertRaises(ValueError):
            ExpertKey(-1, 0)
        manager = ExpertResidencyManager(
            Backend(), maximum_entries=1, maximum_bytes=100
        )
        lease = manager.acquire(ExpertKey(0, 0))
        with self.assertRaisesRegex(RuntimeError, "active"):
            manager.close()
        lease.release()
        manager.close()


if __name__ == "__main__":
    unittest.main()
