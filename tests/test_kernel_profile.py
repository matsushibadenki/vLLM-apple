import json
import tempfile
import unittest
from pathlib import Path

from tests.schema_validator import validate_instance
from vllm_apple.cli import main
from vllm_apple.kernel_profile import build_model_kernel_shape_profile
from vllm_apple.model import ModelInspectionError, inspect_model


class ModelKernelShapeProfileTests(unittest.TestCase):
    def model(self, directory: str, *, maximum_context: int = 32768):
        path = Path(directory)
        config = {
            "architectures": ["QwenForCausalLM"],
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "max_position_embeddings": maximum_context,
            "torch_dtype": "bfloat16",
        }
        (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"weights")
        return inspect_model(path)

    def test_profile_uses_model_attention_shape_and_is_schema_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = build_model_kernel_shape_profile(self.model(directory))
        self.assertEqual(profile.architecture, "QwenForCausalLM")
        self.assertEqual([shape.context_tokens for shape in profile.shapes], [128, 1024, 4096, 16384])
        shape = profile.shapes[1]
        self.assertEqual((shape.query_heads, shape.kv_heads, shape.head_dimension), (32, 8, 128))
        self.assertEqual(shape.blocks_per_sequence, 64)
        self.assertEqual(shape.kv_working_set_bytes, 1024 * 2 * 8 * 128 * 2)
        schema = json.loads(
            Path("schemas/runtime/kernel-shape-profile-v1.schema.json").read_text()
        )
        validate_instance(profile.to_dict(), schema)

    def test_contexts_are_deduplicated_clamped_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = self.model(directory, maximum_context=512)
            first = build_model_kernel_shape_profile(
                model, context_tiers=(1024, 128, 4096, 128)
            )
            second = build_model_kernel_shape_profile(
                model, context_tiers=(128, 512)
            )
        self.assertEqual([shape.context_tokens for shape in first.shapes], [128, 512])
        self.assertEqual(first.profile_id, second.profile_id)

    def test_invalid_gqa_shape_is_rejected_without_allocating_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = self.model(directory)
            model.config["num_key_value_heads"] = 7
            with self.assertRaises(ModelInspectionError):
                build_model_kernel_shape_profile(model)

    def test_cli_emits_model_backed_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.model(directory, maximum_context=2048)
            from contextlib import redirect_stdout
            from io import StringIO

            output = StringIO()
            with redirect_stdout(output):
                status = main(
                    [
                        "kernel-shape-profile",
                        directory,
                        "--contexts",
                        "256,4096",
                        "--block-tokens",
                        "32",
                    ]
                )
        payload = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(
            [shape["context_tokens"] for shape in payload["shapes"]], [256, 2048]
        )
        self.assertEqual(payload["shapes"][0]["block_tokens"], 32)


if __name__ == "__main__":
    unittest.main()
