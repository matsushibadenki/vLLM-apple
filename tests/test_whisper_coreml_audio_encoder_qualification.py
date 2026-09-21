from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.qualify_whisper_coreml_audio_encoder import _artifact_digest


class WhisperCoreMLAudioEncoderQualificationTests(unittest.TestCase):
    def test_artifact_digest_is_content_and_path_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights").mkdir()
            (root / "model.mlmodel").write_bytes(b"model")
            (root / "weights" / "weight.bin").write_bytes(b"weight")
            first = _artifact_digest(root)
            self.assertEqual(first[1:], (2, 11))
            (root / "weights" / "weight.bin").write_bytes(b"changed")
            self.assertNotEqual(_artifact_digest(root)[0], first[0])

    def test_artifact_digest_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_bytes(b"x")
            (root / "link").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlinks"):
                _artifact_digest(root)


if __name__ == "__main__":
    unittest.main()
