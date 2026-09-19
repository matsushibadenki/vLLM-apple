import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_qwen3_vl_conversion_plan import Qwen3VLConversionPlanTests
from vllm_apple.qwen3_vl_conversion_plan import build_qwen3_vl_coreml_conversion_plan
from vllm_apple.qwen3_vl_conversion_worker import (
    _convert_tensor,
    stage_qwen3_vl_coreml_weights,
)

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "Qwen3-VL conversion worker requires the coreml extra")
class Qwen3VLConversionWorkerTests(unittest.TestCase):
    def test_stages_complete_private_atomic_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Qwen3VLConversionPlanTests().fixture(root)
            plan = build_qwen3_vl_coreml_conversion_plan(
                root, source, fixed_grid_profiles=((1, 2, 2),)
            )
            output = root / "staged"

            result = stage_qwen3_vl_coreml_weights(
                root,
                output,
                source,
                plan,
                maximum_output_bytes=plan.target_tensor_bytes,
            )

            self.assertEqual(result, output.resolve())
            self.assertEqual(output.stat().st_mode & 0o777, 0o700)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["plan_id"], plan.plan_id)
            self.assertEqual(manifest["tensor_count"], 27)
            self.assertEqual(manifest["total_bytes"], plan.target_tensor_bytes)
            self.assertEqual((output / "manifest.json").stat().st_mode & 0o777, 0o600)
            for record in manifest["records"]:
                payload = (output / record["file"]).read_bytes()
                self.assertEqual(len(payload), record["bytes"])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), record["sha256"])
            patch_record = next(
                record
                for record in manifest["records"]
                if record["name"] == "vision_tower.patch_embed.proj.weight"
            )
            self.assertEqual(patch_record["shape"], [2, 3, 1, 1, 1])

    def test_conv3d_payload_is_converted_and_transposed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            destination = root / "target.bin"
            values = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            bf16 = (values.view(np.uint32) >> 16).astype("<u2").tobytes()
            source.write_bytes(bf16)

            digest = _convert_tensor(
                source,
                {"file_offset": 0, "bytes": 8, "shape": [1, 1, 1, 2, 2]},
                destination,
                transpose_conv3d=True,
                numpy=np,
            )

            converted = np.frombuffer(destination.read_bytes(), dtype="<f2").tolist()
            self.assertEqual(converted, [1.0, 3.0, 2.0, 4.0])
            self.assertEqual(digest, hashlib.sha256(destination.read_bytes()).hexdigest())

    def test_failure_does_not_publish_partial_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Qwen3VLConversionPlanTests().fixture(root)
            plan = build_qwen3_vl_coreml_conversion_plan(
                root, source, fixed_grid_profiles=((1, 2, 2),)
            )
            output = root / "staged"
            with patch(
                "vllm_apple.qwen3_vl_conversion_worker._convert_tensor",
                side_effect=OSError("injected write failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected write failure"):
                    stage_qwen3_vl_coreml_weights(
                        root,
                        output,
                        source,
                        plan,
                        maximum_output_bytes=plan.target_tensor_bytes,
                    )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".staged.*")), [])

    def test_rejects_output_limit_before_creating_temporary_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Qwen3VLConversionPlanTests().fixture(root)
            plan = build_qwen3_vl_coreml_conversion_plan(
                root, source, fixed_grid_profiles=((1, 2, 2),)
            )
            output = root / "staged"
            with self.assertRaisesRegex(ValueError, "does not match its plan"):
                stage_qwen3_vl_coreml_weights(
                    root,
                    output,
                    source,
                    plan,
                    maximum_output_bytes=plan.target_tensor_bytes - 1,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
