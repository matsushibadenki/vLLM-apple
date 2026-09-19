import json
import struct
import tempfile
import unittest
from pathlib import Path

from vllm_apple.qwen3_vl_ane import Qwen3VLVisionANEAdapterSpec
from vllm_apple.qwen3_vl_conversion_plan import build_qwen3_vl_coreml_conversion_plan


class Qwen3VLConversionPlanTests(unittest.TestCase):
    def fixture(self, root: Path, *, bad_qkv=False):
        fingerprint = "a" * 64
        tensors = {
            "vision_tower.patch_embed.proj.weight": [2, 1, 1, 1, 3],
            "vision_tower.patch_embed.proj.bias": [2],
            "vision_tower.pos_embed.weight": [4, 2],
        }
        for prefix, norm_size in (("vision_tower.merger", 2),
                                  ("vision_tower.deepstack_merger_list.0", 2)):
            tensors.update({
                f"{prefix}.norm.weight": [norm_size],
                f"{prefix}.norm.bias": [norm_size],
                f"{prefix}.linear_fc1.weight": [2, 2],
                f"{prefix}.linear_fc1.bias": [2],
                f"{prefix}.linear_fc2.weight": [2, 2],
                f"{prefix}.linear_fc2.bias": [2],
            })
        block = "vision_tower.blocks.0"
        tensors.update({
            f"{block}.attn.qkv.weight": [5 if bad_qkv else 6, 2],
            f"{block}.attn.qkv.bias": [6],
            f"{block}.attn.proj.weight": [2, 2],
            f"{block}.attn.proj.bias": [2],
            f"{block}.mlp.linear_fc1.weight": [4, 2],
            f"{block}.mlp.linear_fc1.bias": [4],
            f"{block}.mlp.linear_fc2.weight": [2, 4],
            f"{block}.mlp.linear_fc2.bias": [2],
            f"{block}.norm1.weight": [2],
            f"{block}.norm1.bias": [2],
            f"{block}.norm2.weight": [2],
            f"{block}.norm2.bias": [2],
        })
        header = {}
        data = bytearray()
        for name, shape in sorted(tensors.items()):
            elements = 1
            for dimension in shape:
                elements *= dimension
            start = len(data)
            data.extend(b"\0" * (elements * 2))
            header[name] = {
                "dtype": "BF16", "shape": shape, "data_offsets": [start, len(data)]
            }
        encoded = json.dumps(header, separators=(",", ":")).encode()
        shard = root / "model.safetensors"
        shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": len(data)},
            "weight_map": {name: shard.name for name in tensors},
        }))
        source = Qwen3VLVisionANEAdapterSpec(
            "b" * 40, fingerprint, f"vision_encoder@{fingerprint[:16]}",
            1, 2, 4, 1, 1, 1, 1, 2, (0,), len(tensors), shard.stat().st_size,
            "bf16", False,
        )
        return source

    def test_builds_complete_bf16_to_fp16_plan_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.fixture(root)
            plan = build_qwen3_vl_coreml_conversion_plan(
                root, source, fixed_grid_profiles=((1, 2, 2), (1, 4, 4))
            )
        self.assertEqual(plan.tensor_count, 27)
        self.assertEqual(plan.source_precision, "bf16")
        self.assertEqual(plan.target_precision, "fp16")
        self.assertEqual(plan.source_tensor_bytes, plan.target_tensor_bytes)
        self.assertIn(("conv3d_mlx_otwci_to_coreml_oictw", 1), plan.transform_counts)
        self.assertEqual(len(plan.plan_id), 64)

    def test_shape_drift_and_invalid_grid_fail_before_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.fixture(root, bad_qkv=True)
            with self.assertRaisesRegex(ValueError, "block tensor shape"):
                build_qwen3_vl_coreml_conversion_plan(
                    root, source, fixed_grid_profiles=((1, 2, 2),)
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "grid profile"):
                build_qwen3_vl_coreml_conversion_plan(
                    root, source, fixed_grid_profiles=((1, 2305, 1),)
                )

    def test_real_artifact_plan_when_present(self):
        root = Path("models/Qwen3-VL-2B-Instruct-4bit")
        if not root.is_dir():
            self.skipTest("local Qwen3-VL artifact is not installed")
        from vllm_apple.qwen3_vl_ane import inspect_qwen3_vl_vision_for_ane

        source = inspect_qwen3_vl_vision_for_ane(
            root, model_revision="9c4f5209e57b31f4b9dfba735de3fb983739c9cc"
        )
        plan = build_qwen3_vl_coreml_conversion_plan(
            root, source, fixed_grid_profiles=((1, 16, 16), (1, 32, 32), (1, 48, 48))
        )
        self.assertEqual(plan.tensor_count, 315)
        self.assertEqual(plan.source_precision, "bf16")
        self.assertGreater(plan.source_tensor_bytes, 500_000_000)
        self.assertEqual(dict(plan.transform_counts)["conv3d_mlx_otwci_to_coreml_oictw"], 1)


if __name__ == "__main__":
    unittest.main()
