import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_apple.cli import main
from vllm_apple.numeric_artifact import NumericArtifactReader


class NumericCLITests(unittest.TestCase):
    def test_create_emits_transport_identity_for_private_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            packed = root / "packed.bin"
            scales = root / "scales.bin"
            output = root / "weight.json"
            packed.write_bytes(bytes([0xA2, 0x02]))
            scales.write_bytes(bytes([56, 64, 72]))
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                result = main(
                    [
                        "numeric-artifact-create",
                        "--packed", str(packed),
                        "--scales", str(scales),
                        "--output", str(output),
                        "--shape", "3,1",
                        "--scale-axis", "1",
                    ]
                )
            report = json.loads(stream.getvalue())
            self.assertEqual(result, 0)
            self.assertTrue(report["created"])
            self.assertEqual(report["artifact_name"], "weight.json")
            self.assertEqual(report["target_dtype"], "F32")
            self.assertTrue(report["stores_tensor_values"])
            loaded = NumericArtifactReader(root).read(
                report["artifact_name"], report["artifact_digest"])
            self.assertEqual(loaded.execution_contract.contract_id, report["contract_id"])

    def test_create_rejects_geometry_scale_mismatch_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            packed = root / "packed.bin"
            scales = root / "scales.bin"
            output = root / "weight.json"
            packed.write_bytes(bytes([0]))
            scales.write_bytes(bytes([56]))
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                result = main(
                    [
                        "numeric-artifact-create",
                        "--packed", str(packed),
                        "--scales", str(scales),
                        "--output", str(output),
                        "--shape", "1",
                        "--scale-axis", "2",
                    ]
                )
            self.assertEqual(result, 2)
            self.assertEqual(
                json.loads(stream.getvalue())["error_code"],
                "numeric_artifact_creation_failed",
            )
            self.assertFalse(output.exists())

    @patch("vllm_apple.cli.Qwen4RuntimeClient")
    def test_load_and_unload_forward_bounded_runtime_arguments(self, client_type):
        client = client_type.return_value
        client.load_numeric.return_value = {
            "passed": True,
            "handle": "b" * 32,
            "artifact_state": "consumed",
        }
        client.unload.return_value = {"passed": True}
        client.load_numeric_streaming.return_value = {
            "passed": True,
            "handle": "c" * 32,
            "artifact_state": "consumed",
        }
        common = [
            "--socket", "/tmp/runtime.sock",
            "--session-file", "/tmp/session.json",
            "--sequence", "4",
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            loaded = main(
                [
                    "numeric-runtime-load",
                    *common,
                    "--artifact-name", "weight.json",
                    "--artifact-digest", "a" * 64,
                    "--target-dtype", "BF16",
                    "--scratch-bytes", "32",
                ]
            )
            unloaded = main(
                [
                    "numeric-runtime-unload",
                    *common,
                    "--handle", "b" * 32,
                ]
            )
            streamed = main(
                [
                    "numeric-runtime-load",
                    *common,
                    "--artifact-name", "stream.json",
                    "--artifact-digest", "d" * 64,
                    "--target-dtype", "F32",
                    "--tile-bytes", "4096",
                    "--buffer-count", "2",
                ]
            )
        self.assertEqual((loaded, unloaded, streamed), (0, 0, 0))
        client.load_numeric.assert_called_once_with(
            sequence=4,
            artifact_name="weight.json",
            artifact_digest="a" * 64,
            target_dtype="BF16",
            scratch_bytes=32,
        )
        client.unload.assert_called_once_with(sequence=4, handle="b" * 32)
        client.load_numeric_streaming.assert_called_once_with(
            sequence=4,
            artifact_name="stream.json",
            artifact_digest="d" * 64,
            target_dtype="F32",
            scratch_bytes=0,
            tile_bytes=4096,
            buffer_count=2,
        )


if __name__ == "__main__":
    unittest.main()
