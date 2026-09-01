import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_apple.qwen4_shard_stager import (
    CHECKPOINT_NAME,
    COPY_CHUNK_BYTES,
    stage_qwen4_shards,
    verify_qwen4_stage,
)
from vllm_apple.qwen4_weight_map import _text_keys


class Qwen4ShardStagerTests(unittest.TestCase):
    def source(self, root: Path) -> Path:
        source = root / "source"
        source.mkdir()
        config = {
            "model_type": "qwen4_exp",
            "language_model_only": True,
            "text_config": {
                "model_type": "qwen4_exp_text",
                "num_hidden_layers": 1,
                "layer_types": ["full_attention"],
                "max_position_embeddings": 262144,
                "mtp_num_hidden_layers": 0,
                "ple_layer_ids": [],
                "split_ngram_parts": 1,
            },
        }
        (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
        names = _text_keys(config)
        weight_map = {
            name: "model-00001-of-00002.safetensors" if index % 2 == 0 else "model-00002-of-00002.safetensors"
            for index, name in enumerate(sorted(names))
        }
        (source / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 32}, "weight_map": weight_map}),
            encoding="utf-8",
        )
        (source / "model-00001-of-00002.safetensors").write_bytes(b"first-shard")
        (source / "model-00002-of-00002.safetensors").write_bytes(b"second-shard")
        return source

    def test_atomically_stages_private_identity_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            output = root / "output"
            result = stage_qwen4_shards(source, output, maximum_output_bytes=4096)
            self.assertTrue(result["completed"])
            self.assertEqual(result["copied_shards"], 2)
            self.assertEqual(result["copy_chunk_bytes"], COPY_CHUNK_BYTES)
            self.assertEqual((output / "model-00001-of-00002.safetensors").read_bytes(), b"first-shard")
            self.assertFalse((output / CHECKPOINT_NAME).exists())
            self.assertEqual(output.stat().st_mode & 0o777, 0o700)
            verified = verify_qwen4_stage(output, maximum_artifact_bytes=4096)
            self.assertTrue(verified["verified"])
            self.assertEqual(verified["shard_count"], 2)

    def test_resume_reuses_only_digest_bound_completed_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            output = root / "output"
            original = __import__("vllm_apple.qwen4_shard_stager", fromlist=["_copy_atomic"])._copy_atomic
            calls = 0

            def interrupted(source_path, destination, *, maximum_bytes):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("interrupt")
                return original(source_path, destination, maximum_bytes=maximum_bytes)

            with patch("vllm_apple.qwen4_shard_stager._copy_atomic", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "interrupt"):
                    stage_qwen4_shards(source, output, maximum_output_bytes=4096)
            result = stage_qwen4_shards(
                source, output, maximum_output_bytes=4096, resume=True
            )
            self.assertEqual(result["reused_shards"], 1)
            self.assertEqual(result["copied_shards"], 1)

    def test_resume_rejects_tampered_completed_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            output = root / "output"
            original = __import__("vllm_apple.qwen4_shard_stager", fromlist=["_copy_atomic"])._copy_atomic
            calls = 0

            def interrupted(source_path, destination, *, maximum_bytes):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("interrupt")
                return original(source_path, destination, maximum_bytes=maximum_bytes)

            with patch("vllm_apple.qwen4_shard_stager._copy_atomic", side_effect=interrupted):
                with self.assertRaises(RuntimeError):
                    stage_qwen4_shards(source, output, maximum_output_bytes=4096)
            (output / "model-00001-of-00002.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                stage_qwen4_shards(source, output, maximum_output_bytes=4096, resume=True)

    def test_stops_before_copying_a_shard_beyond_the_artifact_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            output = root / "output"
            first_size = (source / "model-00001-of-00002.safetensors").stat().st_size
            with self.assertRaisesRegex(ValueError, "exceeded"):
                stage_qwen4_shards(source, output, maximum_output_bytes=first_size)
            self.assertTrue((output / "model-00001-of-00002.safetensors").exists())
            self.assertFalse((output / "model-00002-of-00002.safetensors").exists())

    def test_verifier_rejects_tampering_and_unexpected_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            output = root / "output"
            stage_qwen4_shards(source, output, maximum_output_bytes=4096)
            shard = output / "model-00002-of-00002.safetensors"
            shard.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "digest"):
                verify_qwen4_stage(output, maximum_artifact_bytes=4096)

            stage_root = root / "second-output"
            stage_qwen4_shards(source, stage_root, maximum_output_bytes=4096)
            (stage_root / "unexpected.txt").write_text("unexpected", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                verify_qwen4_stage(stage_root, maximum_artifact_bytes=4096)


if __name__ == "__main__":
    unittest.main()
