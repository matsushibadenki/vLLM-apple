import json
import struct
import tempfile
import unittest
from pathlib import Path

from vllm_apple.qwen3_vl_ane import inspect_qwen3_vl_vision_for_ane


REVISION = "9c4f5209e57b31f4b9dfba735de3fb983739c9cc"


class Qwen3VLANEAdapterTests(unittest.TestCase):
    def fixture(self, root: Path, *, omit: str | None = None) -> None:
        config = {
            "model_type": "qwen3_vl",
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "quantization_config": {"bits": 4, "mode": "affine"},
            "vision_config": {
                "depth": 2,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 4,
                "patch_size": 16,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "out_hidden_size": 24,
                "deepstack_visual_indexes": [0],
            },
        }
        processor = {"processor_class": "Qwen3VLProcessor"}
        keys = {
            "vision_tower.patch_embed.proj.weight",
            "vision_tower.pos_embed.weight",
            "vision_tower.merger.linear_fc1.weight",
            "vision_tower.merger.linear_fc2.weight",
            "vision_tower.deepstack_merger_list.0.linear_fc1.weight",
            "vision_tower.deepstack_merger_list.0.linear_fc2.weight",
        }
        for layer in range(2):
            keys.update({
                f"vision_tower.blocks.{layer}.attn.qkv.weight",
                f"vision_tower.blocks.{layer}.attn.proj.weight",
                f"vision_tower.blocks.{layer}.mlp.linear_fc1.weight",
                f"vision_tower.blocks.{layer}.mlp.linear_fc2.weight",
                f"vision_tower.blocks.{layer}.norm1.weight",
                f"vision_tower.blocks.{layer}.norm2.weight",
            })
        if omit is not None:
            keys.remove(omit)
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (root / "preprocessor_config.json").write_text(
            json.dumps(processor), encoding="utf-8"
        )
        header = {}
        data = bytearray()
        for key in sorted(keys):
            start = len(data)
            data.extend(b"\0\0")
            header[key] = {
                "dtype": "BF16", "shape": [1], "data_offsets": [start, len(data)]
            }
        encoded = json.dumps(header, separators=(",", ":")).encode()
        (root / "model.safetensors").write_bytes(
            struct.pack("<Q", len(encoded)) + encoded + data
        )
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": 7},
            "weight_map": {key: "model.safetensors" for key in sorted(keys)},
        }), encoding="utf-8")

    def test_inspection_binds_shape_revision_inventory_and_quantization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            spec = inspect_qwen3_vl_vision_for_ane(root, model_revision=REVISION)
        self.assertEqual(spec.depth, 2)
        self.assertEqual(spec.hidden_size, 16)
        self.assertEqual(spec.deepstack_visual_indexes, (0,))
        self.assertEqual(spec.source_precision, "bf16")
        self.assertGreater(spec.artifact_bytes, 7)
        self.assertTrue(spec.operator.startswith("vision_encoder@"))
        self.assertFalse(spec.coreml_artifact_ready)

    def test_missing_layer_tensor_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, omit="vision_tower.blocks.1.attn.qkv.weight")
            with self.assertRaisesRegex(ValueError, "inventory is incomplete"):
                inspect_qwen3_vl_vision_for_ane(root, model_revision=REVISION)

    def test_wrong_architecture_and_quantization_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            config_path = root / "config.json"
            config = json.loads(config_path.read_text())
            config["architectures"] = ["AnotherModel"]
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "not a supported"):
                inspect_qwen3_vl_vision_for_ane(root, model_revision=REVISION)

    def test_revision_must_be_an_exact_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            with self.assertRaisesRegex(ValueError, "commit SHA"):
                inspect_qwen3_vl_vision_for_ane(root, model_revision="main")

    def test_real_qwen3_vl_artifact_contract_when_present(self):
        root = Path("models/Qwen3-VL-2B-Instruct-4bit")
        if not root.is_dir():
            self.skipTest("local Qwen3-VL artifact is not installed")
        spec = inspect_qwen3_vl_vision_for_ane(root, model_revision=REVISION)
        self.assertEqual(spec.depth, 24)
        self.assertEqual(spec.hidden_size, 1024)
        self.assertEqual(spec.intermediate_size, 4096)
        self.assertEqual(spec.output_hidden_size, 2048)
        self.assertEqual(spec.deepstack_visual_indexes, (5, 11, 17))
        self.assertEqual(spec.vision_tensor_count, 315)
        self.assertGreater(spec.artifact_bytes, 1_000_000_000)
        self.assertFalse(spec.coreml_artifact_ready)


if __name__ == "__main__":
    unittest.main()
