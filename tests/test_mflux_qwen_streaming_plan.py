import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.inspect_mflux_qwen_streaming import _deployable_tree_identity
from vllm_apple.mflux_qwen_layer_loader import (
    _unique_object,
    load_mflux_qwen_text_layer,
)
from vllm_apple.mflux_qwen_streaming_encoder import build_streaming_qwen_text_encoder
from vllm_apple.mflux_qwen_streaming_plan import inspect_mflux_qwen_text_encoder_staging


class MFluxQwenStreamingPlanTests(unittest.TestCase):
    def test_layer_loader_rejects_invalid_index_and_duplicate_header_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the fixed profile"):
            load_mflux_qwen_text_layer(Path("unused"), -1)
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            _unique_object([("weight", 1), ("weight", 2)])

    def test_streaming_encoder_rejects_unbounded_layer_count_before_loading(self) -> None:
        with self.assertRaisesRegex(ValueError, "layer limit is invalid"):
            build_streaming_qwen_text_encoder(Path("unused"), layer_limit=29)

    def test_deployable_identity_excludes_git_lfs_store(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weight.safetensors").write_bytes(b"deployable")
            (root / ".git" / "lfs").mkdir(parents=True)
            (root / ".git" / "lfs" / "object").write_bytes(b"duplicate")
            digest, count, size = _deployable_tree_identity(root)
            self.assertEqual((count, size), (1, len(b"deployable")))
            (root / ".git" / "lfs" / "object").write_bytes(b"changed duplicate")
            self.assertEqual(_deployable_tree_identity(root)[0], digest)

    @unittest.skipUnless(
        importlib.util.find_spec("safetensors") and importlib.util.find_spec("numpy"),
        "optional safetensors and numpy are required",
    )
    def test_header_only_layer_inventory_and_index_binding(self) -> None:
        import numpy as np
        from safetensors.numpy import save_file

        with TemporaryDirectory() as directory:
            root = Path(directory)
            component = root / "text_encoder"
            component.mkdir()
            arrays = {
                "encoder.embed_tokens.weight": np.ones((2, 4), dtype=np.float16),
                "encoder.layers.0.a.weight": np.ones((4, 4), dtype=np.float16),
                "encoder.layers.1.a.weight": np.ones((4, 4), dtype=np.float16),
                "encoder.norm.weight": np.ones((4,), dtype=np.float16),
            }
            save_file(arrays, component / "0.safetensors")
            index = component / "model.safetensors.index.json"
            index.write_text(json.dumps({
                "metadata": {"quantization_level": "4"},
                "weight_map": {name: "0.safetensors" for name in arrays},
            }))
            plan = inspect_mflux_qwen_text_encoder_staging(root, expected_layers=2)
            self.assertEqual(plan.layer_payload_bytes, (32, 32))
            self.assertEqual(plan.tensor_dtype_counts, {"F16": 4})
            self.assertEqual(plan.quantized_weight_tensor_count, 0)
            self.assertEqual(plan.static_payload_bytes, 24)
            self.assertEqual(plan.static_plus_maximum_layer_bytes, 56)
            self.assertEqual(plan.total_payload_bytes, 88)

            payload = json.loads(index.read_text())
            payload["weight_map"]["encoder.layers.2.a.weight"] = "0.safetensors"
            index.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
                inspect_mflux_qwen_text_encoder_staging(root, expected_layers=2)


if __name__ == "__main__":
    unittest.main()
