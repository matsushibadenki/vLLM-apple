import hashlib
import tempfile
import unittest
from pathlib import Path

from vllm_apple.qwen3_vl_persistent_encoder import (
    publish_qwen3_vl_persistent_transport_manifest,
)


class Qwen3VLPersistentEncoderTests(unittest.TestCase):
    def test_manifest_binds_all_worker_digests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "transport"
            root.mkdir(mode=0o700)
            payload = b"\0" * (64 * 2048 * 2)
            digest = hashlib.sha256(payload).hexdigest()
            for name in ("final", "deepstack_0", "deepstack_1", "deepstack_2"):
                path = root / f"{name}.fp16"
                path.write_bytes(payload)
                path.chmod(0o600)
            manifest = publish_qwen3_vl_persistent_transport_manifest(
                root, {"output_digests": [digest] * 4}, graph_id="a" * 64
            )
            self.assertEqual(manifest["graph_id"], "a" * 64)
            self.assertEqual(len(manifest["records"]), 4)
            self.assertEqual((root / "manifest.json").stat().st_mode & 0o777, 0o600)

    def test_digest_mismatch_fails_before_manifest_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "transport"
            root.mkdir(mode=0o700)
            payload = b"\0" * (64 * 2048 * 2)
            for name in ("final", "deepstack_0", "deepstack_1", "deepstack_2"):
                path = root / f"{name}.fp16"
                path.write_bytes(payload)
                path.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "digest changed"):
                publish_qwen3_vl_persistent_transport_manifest(
                    root, {"output_digests": ["b" * 64] * 4}, graph_id="a" * 64
                )
            self.assertFalse((root / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
