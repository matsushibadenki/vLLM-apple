import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vllm_apple.mflux_qwen_transformer_loader import load_mflux_qwen_transformer_block
from vllm_apple.mflux_qwen_transformer_plan import inspect_mflux_qwen_transformer_staging


class MFluxQwenTransformerPlanTests(unittest.TestCase):
    def test_loader_rejects_invalid_block_before_weight_access(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the fixed profile"):
            load_mflux_qwen_transformer_block(Path("unused"), 60)

    def test_invalid_expected_depth_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "block count is invalid"):
            inspect_mflux_qwen_transformer_staging(Path("unused"), expected_blocks=0)

    @unittest.skipUnless(
        importlib.util.find_spec("numpy") and importlib.util.find_spec("safetensors"),
        "NumPy and safetensors are required",
    )
    def test_header_only_inventory_and_index_binding(self) -> None:
        import numpy as np
        from safetensors.numpy import save_file

        with TemporaryDirectory() as directory:
            component = Path(directory) / "transformer"
            component.mkdir()
            arrays = {
                "img_in.bias": np.ones((2,), dtype=np.float16),
                "transformer_blocks.0.attn.q.weight": np.ones((2, 8), dtype=np.uint32),
                "transformer_blocks.0.attn.q.scales": np.ones((2, 1), dtype=np.float16),
                "transformer_blocks.0.attn.q.biases": np.ones((2, 1), dtype=np.float16),
                "transformer_blocks.1.attn.q.weight": np.ones((2, 8), dtype=np.uint32),
                "transformer_blocks.1.attn.q.scales": np.ones((2, 1), dtype=np.float16),
                "transformer_blocks.1.attn.q.biases": np.ones((2, 1), dtype=np.float16),
            }
            save_file(arrays, component / "0.safetensors")
            index = component / "model.safetensors.index.json"
            index.write_text(json.dumps({
                "metadata": {"quantization_level": "4"},
                "weight_map": {name: "0.safetensors" for name in arrays},
            }))
            plan = inspect_mflux_qwen_transformer_staging(
                Path(directory), expected_blocks=2
            )
            self.assertEqual(plan.shard_count, 1)
            self.assertEqual(plan.tensor_count, 7)
            self.assertEqual(plan.block_payload_bytes, (72, 72))
            self.assertEqual(plan.static_payload_bytes, 4)
            self.assertEqual(plan.static_plus_maximum_block_bytes, 76)
            self.assertEqual(plan.quantized_weight_tensor_count, 2)
            self.assertEqual(plan.quantization_aux_tensor_count, 4)
            self.assertEqual(plan.quantized_block_weight_counts, (1, 1))

            arrays["transformer_blocks.1.attn.q.scales"] = np.ones((2, 2), dtype=np.float16)
            save_file(arrays, component / "0.safetensors")
            with self.assertRaisesRegex(ValueError, "packed 4-bit layout"):
                inspect_mflux_qwen_transformer_staging(Path(directory), expected_blocks=2)
            arrays["transformer_blocks.1.attn.q.scales"] = np.ones((2, 1), dtype=np.float16)
            save_file(arrays, component / "0.safetensors")

            payload = json.loads(index.read_text())
            payload["weight_map"]["transformer_blocks.2.attn.q.weight"] = "0.safetensors"
            index.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
                inspect_mflux_qwen_transformer_staging(Path(directory), expected_blocks=2)
