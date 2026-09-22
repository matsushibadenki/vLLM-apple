import hashlib
import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vllm_apple.mflux_qwen_prompt_handoff import (
    MANIFEST_NAME,
    PAYLOAD_NAME,
    consume_mflux_qwen_prompt_handoff,
    save_mflux_qwen_prompt_handoff,
)


@unittest.skipUnless(
    importlib.util.find_spec("numpy") and importlib.util.find_spec("safetensors"),
    "NumPy and safetensors are required",
)
class MFluxQwenPromptHandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        import numpy as np

        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.plan_sha256 = hashlib.sha256(b"plan").hexdigest()
        self.prompt_sha256 = hashlib.sha256(b"prompt").hexdigest()
        self.embeddings = np.ones((1, 2, 3584), dtype=np.float32)
        self.mask = np.ones((1, 2), dtype=np.int32)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def publish(self) -> Path:
        return save_mflux_qwen_prompt_handoff(
            self.root, self.embeddings, self.mask,
            plan_sha256=self.plan_sha256,
            prompt_sha256=self.prompt_sha256,
            sample_index=0,
        )

    def test_round_trip_is_single_use_and_private(self) -> None:
        import numpy as np

        manifest = self.publish()
        self.assertEqual(manifest.name, MANIFEST_NAME)
        self.assertEqual(manifest.stat().st_mode & 0o777, 0o600)
        self.assertEqual((self.root / PAYLOAD_NAME).stat().st_mode & 0o777, 0o600)
        embeddings, mask = consume_mflux_qwen_prompt_handoff(
            manifest,
            plan_sha256=self.plan_sha256,
            prompt_sha256=self.prompt_sha256,
            sample_index=0,
        )
        np.testing.assert_array_equal(embeddings, self.embeddings)
        np.testing.assert_array_equal(mask, self.mask)
        self.assertEqual(list(self.root.iterdir()), [])
        with self.assertRaises(ValueError):
            consume_mflux_qwen_prompt_handoff(
                manifest,
                plan_sha256=self.plan_sha256,
                prompt_sha256=self.prompt_sha256,
                sample_index=0,
            )

    def test_wrong_identity_and_payload_tampering_remove_files(self) -> None:
        manifest = self.publish()
        with self.assertRaisesRegex(ValueError, "identity does not match"):
            consume_mflux_qwen_prompt_handoff(
                manifest,
                plan_sha256=hashlib.sha256(b"other").hexdigest(),
                prompt_sha256=self.prompt_sha256,
                sample_index=0,
            )
        self.assertEqual(list(self.root.iterdir()), [])

        manifest = self.publish()
        payload = self.root / PAYLOAD_NAME
        with payload.open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "payload does not match"):
            consume_mflux_qwen_prompt_handoff(
                manifest,
                plan_sha256=self.plan_sha256,
                prompt_sha256=self.prompt_sha256,
                sample_index=0,
            )
        self.assertEqual(list(self.root.iterdir()), [])

    def test_invalid_shape_and_nonprivate_directory_rejected_before_publish(self) -> None:
        import numpy as np

        with self.assertRaisesRegex(ValueError, "tensor contract"):
            save_mflux_qwen_prompt_handoff(
                self.root,
                np.ones((1, 2, 1024), dtype=np.float32),
                self.mask,
                plan_sha256=self.plan_sha256,
                prompt_sha256=self.prompt_sha256,
                sample_index=0,
            )
        self.assertEqual(list(self.root.iterdir()), [])
        self.root.chmod(0o755)
        with self.assertRaisesRegex(ValueError, "directory must be private"):
            self.publish()

    def test_nonprivate_consume_does_not_delete_unowned_files(self) -> None:
        manifest = self.publish()
        payload = self.root / PAYLOAD_NAME
        self.root.chmod(0o755)
        with self.assertRaisesRegex(ValueError, "directory must be private"):
            consume_mflux_qwen_prompt_handoff(
                manifest,
                plan_sha256=self.plan_sha256,
                prompt_sha256=self.prompt_sha256,
                sample_index=0,
            )
        self.assertTrue(manifest.exists())
        self.assertTrue(payload.exists())
