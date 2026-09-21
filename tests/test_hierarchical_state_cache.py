import hashlib
import tempfile
import unittest
from pathlib import Path

from vllm_apple.hierarchical_state_cache import (
    HierarchicalStateCache,
    HierarchicalStateKey,
    HierarchicalStateKind,
)


def key(kind, value):
    return HierarchicalStateKey(kind, hashlib.sha256(value.encode()).hexdigest())


class HierarchicalStateCacheTests(unittest.TestCase):
    def test_hot_lru_spills_and_cold_hit_promotes(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = HierarchicalStateCache(
                Path(directory), hot_entries=1, hot_bytes=8,
                cold_entries=2, cold_bytes=32,
            )
            first = key(HierarchicalStateKind.KV, "first")
            second = key(HierarchicalStateKind.PREFIX, "second")
            cache.put(first, b"1234")
            cache.put(second, b"5678")
            self.assertEqual(cache.snapshot()["spills"], 1)
            restored = cache.get(first)
            self.assertEqual((restored.payload, restored.tier), (b"1234", "hot"))
            self.assertEqual(cache.snapshot()["promotions"], 1)
            cache.clear()
            self.assertEqual(cache.snapshot()["cold_entries"], 0)

    def test_large_embedding_uses_cold_tier_and_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = HierarchicalStateCache(
                Path(directory), hot_entries=1, hot_bytes=4,
                cold_entries=2, cold_bytes=32,
            )
            vision = key(HierarchicalStateKind.VISION_EMBEDDING, "image")
            self.assertEqual(cache.put(vision, b"embedding"), "cold")
            cold_file = next(Path(directory).glob("*.bin"))
            cold_file.write_bytes(b"tampered!")
            with self.assertRaisesRegex(ValueError, "digest"):
                cache.get(vision)

    def test_all_state_kinds_share_bounds_and_keys_store_no_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = HierarchicalStateCache(
                Path(directory), hot_entries=2, hot_bytes=16,
                cold_entries=2, cold_bytes=16,
            )
            for index, kind in enumerate(HierarchicalStateKind):
                cache.put(key(kind, str(index)), bytes((index + 1,)) * 4)
            snapshot = cache.snapshot()
            self.assertLessEqual(snapshot["hot_entries"], 2)
            self.assertLessEqual(snapshot["cold_entries"], 2)
            self.assertTrue(all("prompt" not in path.name for path in Path(directory).iterdir()))


if __name__ == "__main__":
    unittest.main()
