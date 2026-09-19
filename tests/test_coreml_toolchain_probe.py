import tempfile
import unittest
from pathlib import Path

from vllm_apple.coreml_toolchain_probe import _tree_digest, run_coreml_toolchain_probe


class CoreMLToolchainProbeTests(unittest.TestCase):
    def test_tree_digest_is_stable_and_content_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").write_bytes(b"one")
            (root / "nested").mkdir()
            (root / "nested" / "b").write_bytes(b"two")
            first = _tree_digest(root)
            second = _tree_digest(root)
            (root / "nested" / "b").write_bytes(b"three")
            changed = _tree_digest(root)
        self.assertEqual(first, second)
        self.assertEqual(first[1], 2)
        self.assertNotEqual(first[0], changed[0])

    def test_existing_destination_is_rejected_before_optional_import(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "must be new"):
                run_coreml_toolchain_probe(Path(directory))


if __name__ == "__main__":
    unittest.main()
