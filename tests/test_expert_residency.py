import unittest

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
        manager.close()

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
