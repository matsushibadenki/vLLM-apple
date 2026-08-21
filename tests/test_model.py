import json
import tempfile
import unittest
from pathlib import Path

from vllm_apple.model import ModelInspectionError, inspect_model, resolve_model_path


class ModelInspectionTests(unittest.TestCase):
    def test_standard_gqa_transformer_memory_is_derived_from_local_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = {
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
                "max_position_embeddings": 32768,
                "torch_dtype": "bfloat16",
            }
            (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (path / "model-00001-of-00002.safetensors").write_bytes(b"a" * 100)
            (path / "model-00002-of-00002.safetensors").write_bytes(b"b" * 200)

            inspected = inspect_model(str(path))
            self.assertEqual(inspected.memory_spec.weights_bytes, 300)
            self.assertEqual(inspected.memory_spec.kv_bytes_per_token, 131072)
            self.assertEqual(inspected.memory_spec.model_max_context, 32768)
            self.assertEqual(inspected.kv_dtype_bytes, 2)

    def test_hugging_face_cache_resolution_uses_latest_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshots = Path(directory) / "models--org--model" / "snapshots"
            older = snapshots / "old"
            newer = snapshots / "new"
            older.mkdir(parents=True)
            newer.mkdir()
            (older / "config.json").write_text("{}")
            (newer / "config.json").write_text("{}")
            newer.touch()
            self.assertEqual(resolve_model_path("org/model", Path(directory)), newer)

    def test_missing_metadata_fails_instead_of_guessing_kv_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text("{}")
            (path / "model.safetensors").write_bytes(b"weights")
            with self.assertRaises(ModelInspectionError):
                inspect_model(str(path))


if __name__ == "__main__":
    unittest.main()
