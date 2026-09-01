import json
import tempfile
import unittest
from pathlib import Path

from vllm_apple.qwen4_conversion_plan import build_qwen4_conversion_plan
from vllm_apple.qwen4_weight_map import _text_keys


class Qwen4ConversionPlanTests(unittest.TestCase):
    def test_builds_single_shard_streaming_plan_without_tensor_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "model_type": "qwen4_exp",
                "language_model_only": True,
                "text_config": {
                    "model_type": "qwen4_exp_text",
                    "num_hidden_layers": 2,
                    "layer_types": ["linear_attention", "full_attention"],
                    "max_position_embeddings": 262144,
                    "mtp_num_hidden_layers": 0,
                    "ple_layer_ids": [],
                    "split_ngram_parts": 2,
                },
            }
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            names = sorted(_text_keys(config))
            weight_map = {
                name: f"model-{index % 2 + 1:05d}-of-00002.safetensors"
                for index, name in enumerate(names)
            }
            index = root / "model.safetensors.index.json"
            index.write_text(
                json.dumps({"metadata": {"total_size": 4096}, "weight_map": weight_map}),
                encoding="utf-8",
            )
            result = build_qwen4_conversion_plan(root, index)
        self.assertEqual(result["source_entries"], len(names))
        self.assertEqual(result["source_shards"], 2)
        self.assertEqual(result["peak_open_source_shards"], 1)
        self.assertFalse(result["requires_full_artifact_in_memory"])
        self.assertFalse(result["loads_tensor_data"])
        self.assertIn("per_layer_embedding", result["enabled_components"])
        self.assertEqual(
            result["disabled_optional_components"],
            ["multi_token_prediction", "vision_encoder"],
        )
        self.assertEqual(len(result["plan_id"]), 64)


if __name__ == "__main__":
    unittest.main()
