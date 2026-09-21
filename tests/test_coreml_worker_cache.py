import unittest
from pathlib import Path
from unittest.mock import Mock

from vllm_apple.ane_probe import CoreMLANEModelProbeConfig
from vllm_apple.coreml_worker_cache import CoreMLWorkerCache, CoreMLWorkerCacheKey


def config(digest: str) -> CoreMLANEModelProbeConfig:
    return CoreMLANEModelProbeConfig(
        Path("/model.mlmodelc"), Path("/manifest.json"), digest,
        "input", "output", (1.0, 2.0), (2.0, 4.0), 100,
    )


class CoreMLWorkerCacheTests(unittest.TestCase):
    def key(self, value):
        return CoreMLWorkerCacheKey.from_config(
            value, hardware_fingerprint="m4-test", os_version="macos-test")

    def test_reuses_exact_identity_and_releases_lease(self):
        created = []

        def factory(_config):
            worker = Mock()
            created.append(worker)
            return worker

        value = config("a" * 64)
        cache = CoreMLWorkerCache(2, worker_factory=factory)
        first = cache.acquire(self.key(value), value)
        second = cache.acquire(self.key(value), value)
        self.assertIs(first.worker, second.worker)
        self.assertEqual(len(created), 1)
        self.assertEqual(cache.snapshot()["active_leases"], 2)
        first.release()
        second.release()
        cache.close()
        created[0].close.assert_called_once()

    def test_lru_evicts_only_idle_worker(self):
        workers = []

        def factory(_config):
            workers.append(Mock())
            return workers[-1]

        first, second = config("a" * 64), config("b" * 64)
        cache = CoreMLWorkerCache(1, worker_factory=factory)
        lease = cache.acquire(self.key(first), first)
        with self.assertRaisesRegex(RuntimeError, "fully leased"):
            cache.acquire(self.key(second), second)
        lease.release()
        replacement = cache.acquire(self.key(second), second)
        workers[0].close.assert_called_once()
        replacement.release()
        cache.close()

    def test_identity_mismatch_and_active_close_fail_closed(self):
        value = config("a" * 64)
        cache = CoreMLWorkerCache(worker_factory=lambda _config: Mock())
        wrong = CoreMLWorkerCacheKey.from_config(
            config("b" * 64), hardware_fingerprint="m4-test",
            os_version="macos-test")
        with self.assertRaisesRegex(ValueError, "does not match"):
            cache.acquire(wrong, value)
        lease = cache.acquire(self.key(value), value)
        with self.assertRaisesRegex(RuntimeError, "active leases"):
            cache.close()
        lease.release()
        cache.close()


if __name__ == "__main__":
    unittest.main()
