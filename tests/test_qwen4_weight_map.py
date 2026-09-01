import json
import tempfile
import unittest
from pathlib import Path

from vllm_apple.qwen4_weight_map import inspect_qwen4_weight_map


class Qwen4WeightMapTests(unittest.TestCase):
    def fixture(self, directory: str):
        root = Path(directory)
        config = {
            "model_type": "qwen4_exp",
            "language_model_only": False,
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "text_config": {
                "model_type": "qwen4_exp_text",
                "num_hidden_layers": 2,
                "layer_types": ["linear_attention", "full_attention"],
                "max_position_embeddings": 262144,
                "mtp_num_hidden_layers": 1,
                "mtp": {"num_hidden_layers": 1},
                "ple_layer_ids": [],
                "split_ngram_parts": 2,
            },
            "vision_config": {"depth": 1},
        }
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return root, config

    def write_index(self, root: Path, names: set[str]):
        path = root / "model.safetensors.index.json"
        path.write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 1024},
                    "weight_map": {name: "model-00001-of-00001.safetensors" for name in names},
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_missing_map_is_bounded_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.fixture(directory)
            result = inspect_qwen4_weight_map(root, self.write_index(root, {"lm_head.weight"}))
        self.assertFalse(result["compatible"])
        self.assertGreater(result["missing_count"], 32)
        self.assertEqual(len(result["missing"]), 32)
        self.assertTrue(result["missing_truncated"])
        self.assertFalse(result["loads_weights"])

    def test_rejects_path_traversal_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.fixture(directory)
            path = root / "model.safetensors.index.json"
            path.write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 1024},
                        "weight_map": {"lm_head.weight": "../secret.safetensors"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "shard name"):
                inspect_qwen4_weight_map(root, path)


if __name__ == "__main__":
    unittest.main()
