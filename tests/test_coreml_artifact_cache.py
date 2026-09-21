import os
import stat
import tempfile
import unittest
from pathlib import Path

from vllm_apple.coreml_artifact_cache import (
    CoreMLArtifactCache,
    CoreMLArtifactCacheIdentity,
)


def identity(toolchain="xcode-26"):
    return CoreMLArtifactCacheIdentity(
        "a" * 64, "b" * 64, toolchain, "coreml-9", "25A1",
        "cpu_and_neural_engine",
    )


class CoreMLArtifactCacheTests(unittest.TestCase):
    def test_publish_load_bind_environment_and_revoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "fixture.mlmodelc"
            model.mkdir()
            (model / "model.bin").write_bytes(b"compiled")
            cache_root = root / "cache"
            cache_root.mkdir(mode=0o700)
            cache = CoreMLArtifactCache(cache_root, signing_key=b"k" * 32)
            published = cache.publish(identity(), model)
            loaded = cache.load(identity())
            self.assertEqual(loaded.tree_sha256, published.tree_sha256)
            self.assertEqual(stat.S_IMODE(loaded.artifact_path.stat().st_mode), 0o700)
            with self.assertRaisesRegex(ValueError, "quarantined"):
                cache.load(identity("different-toolchain"))
            revoked = cache.revoke(identity(), "toolchain_revoked")
            self.assertEqual(revoked.parent.name, "quarantine")
            self.assertTrue((revoked / "revocation.json").is_file())

    def test_tamper_is_quarantined_and_symlink_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "fixture.mlmodelc"
            model.mkdir()
            (model / "model.bin").write_bytes(b"compiled")
            cache_root = root / "cache"
            cache_root.mkdir(mode=0o700)
            cache = CoreMLArtifactCache(cache_root, signing_key=b"k" * 32)
            entry = cache.publish(identity(), model)
            (entry.artifact_path / "model.bin").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "quarantined"):
                cache.load(identity())
            self.assertFalse((cache_root / "entries" / identity().cache_id).exists())

            unsafe = root / "unsafe.mlmodelc"
            unsafe.mkdir()
            os.symlink(model / "model.bin", unsafe / "linked")
            with self.assertRaisesRegex(ValueError, "unsafe"):
                cache.publish(identity("new"), unsafe)

    def test_cache_root_and_signing_key_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            root.mkdir(mode=0o755)
            with self.assertRaisesRegex(ValueError, "private"):
                CoreMLArtifactCache(root, signing_key=b"k" * 32)
            root.chmod(0o700)
            with self.assertRaisesRegex(ValueError, "32 bytes"):
                CoreMLArtifactCache(root, signing_key=b"short")


if __name__ == "__main__":
    unittest.main()
