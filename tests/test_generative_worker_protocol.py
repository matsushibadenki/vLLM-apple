import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vllm_apple.generative_qualification import (
    GenerativeArtifactComponent,
    build_generative_qualification_plan,
)
from vllm_apple.generative_worker_protocol import (
    build_generative_worker_request,
    consume_private_generative_request,
    save_private_generative_request,
)
from vllm_apple.types import GIB, HardwareInfo, MemoryInfo


class GenerativeWorkerProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.model = self.workspace / "model"
        self.output = self.workspace / "output"
        self.model.mkdir()
        self.output.mkdir()
        hardware = HardwareInfo(
            "Darwin",
            "arm64",
            "Apple M4",
            10,
            10,
            10,
            MemoryInfo(32 * GIB, 28 * GIB),
            True,
            "test",
        )
        self.plan = build_generative_qualification_plan(
            candidate_id="flux2-klein-9b-base",
            artifact_bytes=8 * GIB,
            estimated_resident_bytes=18 * GIB,
            hardware=hardware,
            target=self.workspace,
            quantization="int4",
            components=(
                GenerativeArtifactComponent("transformer", "denoiser", 6 * GIB, 14 * GIB),
                GenerativeArtifactComponent("encoder", "text_encoder", GIB, 2 * GIB),
                GenerativeArtifactComponent("vae", "vae", GIB, 2 * GIB),
            ),
        )

    def request(self):
        return build_generative_worker_request(
            self.plan,
            workspace_root=self.workspace,
            model_root=self.model,
            output_root=self.output,
            mode="text-to-image",
            prompt="A small red apple on a table",
            seed=7,
            sample_index=0,
        )

    def test_private_request_is_consumed_once_and_unlinked(self) -> None:
        request_path = self.workspace / "private" / "request.json"
        save_private_generative_request(self.request(), request_path)
        self.assertEqual(request_path.stat().st_mode & 0o777, 0o600)
        parsed = consume_private_generative_request(
            request_path, workspace_root=self.workspace
        )
        self.assertEqual(parsed["prompt"], "A small red apple on a table")
        self.assertFalse(request_path.exists())

    def test_prompt_tampering_is_rejected_and_request_is_still_removed(self) -> None:
        payload = self.request()
        payload["prompt"] = "changed"
        request_path = self.workspace / "request.json"
        save_private_generative_request(payload, request_path)
        with self.assertRaisesRegex(ValueError, "identity"):
            consume_private_generative_request(request_path, workspace_root=self.workspace)
        self.assertFalse(request_path.exists())

    def test_non_private_request_and_output_inside_model_are_rejected(self) -> None:
        request_path = self.workspace / "request.json"
        save_private_generative_request(self.request(), request_path)
        request_path.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "private"):
            consume_private_generative_request(request_path, workspace_root=self.workspace)
        self.assertFalse(request_path.exists())
        nested = self.model / "output"
        nested.mkdir()
        with self.assertRaisesRegex(ValueError, "outside the model"):
            build_generative_worker_request(
                self.plan,
                workspace_root=self.workspace,
                model_root=self.model,
                output_root=nested,
                mode="text-to-image",
                prompt="test",
                seed=0,
                sample_index=0,
            )

    def test_symlink_request_is_never_followed(self) -> None:
        target = self.workspace / "target.json"
        save_private_generative_request(self.request(), target)
        link = self.workspace / "link.json"
        link.symlink_to(target)
        with self.assertRaises(OSError):
            consume_private_generative_request(link, workspace_root=self.workspace)
        self.assertTrue(target.exists())
        os.unlink(link)

    def test_image_edit_binds_private_input_image_digest(self) -> None:
        image = self.workspace / "input.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"private-image")
        image.chmod(0o600)
        payload = build_generative_worker_request(
            self.plan,
            workspace_root=self.workspace,
            model_root=self.model,
            output_root=self.output,
            mode="image-edit",
            prompt="Make the apple green",
            seed=7,
            sample_index=0,
            input_image_path=image,
        )
        self.assertEqual(payload["abi_version"], 3)
        self.assertFalse(payload["disk_offload"])
        self.assertEqual(payload["input_image_bytes"], image.stat().st_size)
        self.assertEqual(len(payload["input_image_sha256"]), 64)
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"tampered")
        with self.assertRaisesRegex(ValueError, "digest"):
            save_private_generative_request(payload, self.workspace / "edit.json")
            consume_private_generative_request(
                self.workspace / "edit.json", workspace_root=self.workspace
            )

    def test_disk_offload_is_explicitly_bound(self) -> None:
        payload = build_generative_worker_request(
            self.plan,
            workspace_root=self.workspace,
            model_root=self.model,
            output_root=self.output,
            mode="text-to-image",
            prompt="test",
            seed=0,
            sample_index=0,
            disk_offload=True,
        )
        self.assertIs(payload["disk_offload"], True)
        payload["disk_offload"] = "true"
        with self.assertRaisesRegex(ValueError, "disk offload"):
            save_private_generative_request(payload, self.workspace / "disk.json")
            consume_private_generative_request(
                self.workspace / "disk.json", workspace_root=self.workspace
            )

    def test_image_edit_requires_private_png_or_jpeg(self) -> None:
        image = self.workspace / "input.bin"
        image.write_bytes(b"not an image")
        image.chmod(0o600)
        with self.assertRaisesRegex(ValueError, "PNG or JPEG"):
            build_generative_worker_request(
                self.plan,
                workspace_root=self.workspace,
                model_root=self.model,
                output_root=self.output,
                mode="image-edit",
                prompt="edit",
                seed=0,
                sample_index=0,
                input_image_path=image,
            )


if __name__ == "__main__":
    unittest.main()
