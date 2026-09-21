from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.qualify_mlx_speculative_gemma import _tree_digest


class MLXSpeculativeGemmaQualificationTests(unittest.TestCase):
    def test_tree_digest_ignores_git_and_cache_but_binds_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            (root / ".cache").mkdir()
            (root / ".git" / "state").write_bytes(b"ignored")
            (root / ".cache" / "state").write_bytes(b"ignored")
            (root / "model.safetensors").write_bytes(b"weight")
            first = _tree_digest(root)
            self.assertEqual(first[1:], (1, 6))
            (root / ".git" / "state").write_bytes(b"changed")
            self.assertEqual(_tree_digest(root), first)
            (root / "model.safetensors").write_bytes(b"changed")
            self.assertNotEqual(_tree_digest(root)[0], first[0])


if __name__ == "__main__":
    unittest.main()
