import os
import tempfile
import unittest
from pathlib import Path

from vllm_apple.expert_residency import ExpertKey
from vllm_apple.expert_ssd_tier import ExpertSSDStore


class ExpertSSDStoreTests(unittest.TestCase):
    def test_private_store_roundtrip_and_lru_eviction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experts"
            store = ExpertSSDStore(root, maximum_entries=2, maximum_bytes=8)
            first, second, third = ExpertKey(0, 0), ExpertKey(0, 1), ExpertKey(0, 2)
            store.put(first, b"aa")
            store.put(second, b"bb")
            self.assertEqual(store.get(first).data, b"aa")
            store.put(third, b"cc")
            self.assertIsNone(store.get(second))
            self.assertEqual(store.snapshot()["evictions"], 1)
            self.assertEqual(os.stat(root).st_mode & 0o777, 0o700)
            store.clear()
            self.assertEqual(tuple(root.iterdir()), ())

    def test_tamper_is_evicted_and_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experts"
            store = ExpertSSDStore(root, maximum_entries=2, maximum_bytes=8)
            key = ExpertKey(2, 3)
            store.put(key, b"abcd")
            path = next(root.iterdir())
            path.write_bytes(b"bad!")
            self.assertIsNone(store.get(key))
            self.assertEqual(store.snapshot()["integrity_failures"], 1)

    def test_public_root_and_oversize_value_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experts"
            root.mkdir(mode=0o755)
            with self.assertRaises(ValueError):
                ExpertSSDStore(root, maximum_entries=1, maximum_bytes=1)
            os.chmod(root, 0o700)
            store = ExpertSSDStore(root, maximum_entries=1, maximum_bytes=1)
            with self.assertRaises(ValueError):
                store.put(ExpertKey(0, 0), b"xx")


if __name__ == "__main__":
    unittest.main()
