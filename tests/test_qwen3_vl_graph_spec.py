import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tests.test_qwen3_vl_conversion_plan import Qwen3VLConversionPlanTests
from vllm_apple.qwen3_vl_conversion_plan import build_qwen3_vl_coreml_conversion_plan
from vllm_apple.qwen3_vl_conversion_worker import _descriptors
from vllm_apple.qwen3_vl_graph_spec import build_qwen3_vl_coreml_graph_spec


class Qwen3VLGraphSpecTests(unittest.TestCase):
    def staged_fixture(self, root: Path):
        source = Qwen3VLConversionPlanTests().fixture(root)
        plan = build_qwen3_vl_coreml_conversion_plan(
            root, source, fixed_grid_profiles=((1, 2, 2), (1, 4, 4))
        )
        index = json.loads((root / "model.safetensors.index.json").read_text())
        names = sorted(name for name in index["weight_map"] if name.startswith("vision_tower."))
        descriptors = _descriptors(root, index["weight_map"], names)
        staged = root / "staged"
        staged.mkdir(mode=0o700)
        records = []
        for name in names:
            descriptor = descriptors[name]
            shape = list(descriptor["shape"])
            transform = "bf16_to_fp16"
            if name == "vision_tower.patch_embed.proj.weight":
                shape = [shape[0], shape[4], shape[1], shape[2], shape[3]]
                transform = "conv3d_mlx_otwci_to_coreml_oictw"
            payload = b"\0" * descriptor["bytes"]
            filename = hashlib.sha256(name.encode()).hexdigest() + ".fp16"
            (staged / filename).write_bytes(payload)
            records.append({
                "name": name,
                "file": filename,
                "dtype": "F16",
                "shape": shape,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "transform": transform,
            })
        (staged / "manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "plan_id": plan.plan_id,
            "source_artifact_fingerprint": source.artifact_fingerprint,
            "model_revision": source.model_revision,
            "tensor_count": len(records),
            "total_bytes": sum(record["bytes"] for record in records),
            "records": records,
        }))
        return source, plan, staged

    def test_builds_fixed_profile_shapes_and_graph_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            source, plan, staged = self.staged_fixture(Path(directory))
            spec = build_qwen3_vl_coreml_graph_spec(staged, source, plan)

        self.assertEqual(spec.compute_precision, "fp16")
        self.assertEqual(spec.head_dimension, 2)
        self.assertEqual(spec.profiles[0].pixel_values_shape, (4, 3))
        self.assertEqual(spec.profiles[0].hidden_states_shape, (4, 2))
        self.assertEqual(spec.profiles[0].deepstack_output_shapes, ((4, 2),))
        self.assertEqual(spec.profiles[1].hidden_states_shape, (16, 2))
        self.assertEqual(len(spec.staged_weights_sha256), 64)
        self.assertEqual(len(spec.graph_id), 64)
        self.assertIn("vision_rope_theta_10000", spec.graph_semantics)

    def test_weight_tamper_and_unexpected_file_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            source, plan, staged = self.staged_fixture(Path(directory))
            manifest = json.loads((staged / "manifest.json").read_text())
            (staged / manifest["records"][0]["file"]).write_bytes(b"x" * 2)
            with self.assertRaisesRegex(ValueError, "weight file|digest"):
                build_qwen3_vl_coreml_graph_spec(staged, source, plan)
        with tempfile.TemporaryDirectory() as directory:
            source, plan, staged = self.staged_fixture(Path(directory))
            (staged / "extra.bin").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "unexpected files"):
                build_qwen3_vl_coreml_graph_spec(staged, source, plan)

    def test_plan_replay_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, plan, staged = self.staged_fixture(Path(directory))
            manifest_path = staged / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["plan_id"] = "f" * 64
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "do not match"):
                build_qwen3_vl_coreml_graph_spec(staged, source, plan)


if __name__ == "__main__":
    unittest.main()
