import json
import tempfile
import unittest
from pathlib import Path

from vllm_apple.model import (
    MAX_MODEL_CONFIG_BYTES,
    ModelCapabilityError,
    ModelInspectionError,
    assess_model_memory_fit,
    ensure_model_backend_compatible,
    inspect_model_architecture,
    inspect_model,
    resolve_model_path,
)
from vllm_apple.types import GIB, HardwareInfo, MemoryInfo


class ModelInspectionTests(unittest.TestCase):
    def test_qwen_modes_and_yarn_context_are_independent_capabilities(self) -> None:
        capability = inspect_model_architecture(
            {
                "model_type": "qwen4_exp",
                "language_model_only": True,
                "text_config": {
                    "model_type": "qwen4_exp_text",
                    "mtp_num_hidden_layers": 0,
                    "max_position_embeddings": 262144,
                    "rope_parameters": {"rope_type": "yarn", "factor": 4.0},
                },
            }
        )
        self.assertEqual(capability.modes, ("text", "yarn"))
        self.assertEqual(capability.native_context_tokens, 262144)
        self.assertEqual(capability.extended_context_tokens, 1048576)
        self.assertNotIn("vision_encoder", capability.required_features)
        self.assertNotIn("multi_token_prediction", capability.required_features)
        self.assertIn("yarn_extended_context", capability.required_features)

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
            path_inspected = inspect_model(path)
            self.assertEqual(path_inspected.model_id, str(path))

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

    def test_qwen_flash_next_metadata_is_bounded_and_rejected_before_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = {
                "model_type": "qwen4_exp",
                "architectures": ["Qwen3.8-Flash-NextForConditionalGeneration"],
                "language_model_only": False,
                "vision_config": {},
                "text_config": {
                    "model_type": "qwen4_exp_text",
                    "num_hidden_layers": 4,
                    "layer_types": [
                        "linear_attention",
                        "linear_attention",
                        "linear_attention",
                        "full_attention",
                    ],
                    "num_attention_heads": 24,
                    "num_key_value_heads": 2,
                    "head_dim": 256,
                    "hidden_size": 2560,
                    "max_position_embeddings": 262144,
                    "dtype": "bfloat16",
                    "linear_num_key_heads": 16,
                    "linear_num_value_heads": 48,
                    "linear_key_head_dim": 128,
                    "linear_value_head_dim": 128,
                    "linear_conv_kernel_dim": 4,
                    "mamba_ssm_dtype": "float32",
                    "moe_intermediate_size": 640,
                    "shared_expert_intermediate_size": 640,
                    "num_experts": 512,
                    "num_experts_per_tok": 10,
                    "ngram_vocab_size_base": 20_000_000,
                    "ngram_size": 3,
                    "heads_per_ngram": 8,
                    "split_ngram_parts": 128,
                    "hc_count": 4,
                    "hc_lowrank": 320,
                    "vocab_size": 248320,
                    "mtp_num_hidden_layers": 1,
                    "mtp_use_dedicated_embeddings": False,
                    "mtp": {
                        "hybrid": True,
                        "num_hidden_layers": 1,
                        "layer_types": ["full_attention"],
                    },
                    "indexer_kv_heads": 1,
                    "indexer_n_heads": 4,
                    "indexer_head_dim": 128,
                    "indexer_compress_ratio": 4,
                    "indexer_budget": 2048,
                },
            }
            (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (path / "model.safetensors").write_bytes(b"weights")
            inspected = inspect_model(path)
            capability = inspected.architecture_capability
            self.assertEqual(capability.architecture, "qwen4_exp")
            self.assertIn("gated_deltanet", capability.required_features)
            self.assertIn("qwen_sparse_attention", capability.required_features)
            self.assertEqual(capability.modes, ("text", "mtp", "vision"))
            self.assertEqual(capability.native_context_tokens, 262144)
            self.assertIsNone(capability.extended_context_tokens)
            self.assertIsNotNone(inspected.state_memory_spec)
            assert inspected.state_memory_spec is not None
            self.assertEqual(
                inspected.state_memory_spec.recurrent_state_bytes,
                3 * (48 * 128 * 128 + (2 * 16 * 128 + 48 * 128) * 3) * 4,
            )
            self.assertEqual(inspected.memory_spec.kv_bytes_per_token, 2048)
            self.assertEqual(
                inspected.state_memory_spec.sparse_index_bytes_per_token, 64
            )
            self.assertEqual(
                inspected.state_memory_spec.sparse_retrieval_bytes_per_token, 1024
            )
            self.assertEqual(inspected.state_memory_spec.sparse_retrieval_tokens, 2048)
            expert_parameters = 3 * 2560 * 640
            self.assertEqual(
                inspected.state_memory_spec.expert_storage_bytes,
                4 * (512 + 1) * expert_parameters * 2,
            )
            self.assertEqual(
                inspected.state_memory_spec.expert_working_set_bytes,
                4 * (10 + 1) * expert_parameters * 2,
            )
            self.assertEqual(inspected.state_memory_spec.weights_bytes, 0)
            self.assertEqual(
                inspected.state_memory_spec.ngram_storage_bytes,
                20_000_000 * 2560 * 2,
            )
            self.assertEqual(
                inspected.state_memory_spec.ngram_working_set_bytes,
                20_000_000 * 2560 * 2 // 128,
            )
            self.assertEqual(
                inspected.state_memory_spec.scratch_workspace_bytes,
                4 * (2560 + 320) * 2,
            )
            attention_parameters = 2 * 2560 * 24 * 256 + 2 * 2560 * 2 * 256
            self.assertEqual(
                inspected.state_memory_spec.mtp_storage_bytes,
                (attention_parameters + (512 + 1) * expert_parameters) * 2,
            )
            self.assertEqual(
                inspected.state_memory_spec.mtp_working_set_bytes,
                (attention_parameters + (10 + 1) * expert_parameters) * 2,
            )
            small_mac = HardwareInfo(
                platform="Darwin",
                architecture="arm64",
                soc="Apple M",
                physical_cpu_count=8,
                logical_cpu_count=8,
                gpu_core_count=10,
                memory=MemoryInfo(2 * GIB, 2 * GIB),
                is_apple_silicon=True,
                os_version="test",
            )
            fit = assess_model_memory_fit(
                inspected, small_mac, context_tokens=262144
            )
            self.assertEqual(fit.artifact_bytes, len(b"weights"))
            self.assertEqual(fit.hard_ceiling_bytes, GIB)
            self.assertFalse(fit.fits)
            growth = (
                inspected.state_memory_spec.state_bytes(4096)
                - inspected.state_memory_spec.state_bytes(2048)
            )
            self.assertEqual(growth, 2048 * (2048 + 64))
            with self.assertRaisesRegex(
                ModelCapabilityError, "backend_missing_model_capabilities"
            ):
                ensure_model_backend_compatible(inspected)
            ensure_model_backend_compatible(
                inspected,
                available_features=frozenset(capability.required_features),
            )
            (path / "model.safetensors").unlink()
            with self.assertRaisesRegex(
                ModelCapabilityError, "backend_missing_model_capabilities"
            ):
                inspect_model(path, backend="vllm_metal")

    def test_oversized_or_deep_config_is_rejected_before_weight_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config_path = path / "config.json"
            config_path.write_bytes(b" " * (MAX_MODEL_CONFIG_BYTES + 1))
            with self.assertRaisesRegex(ModelInspectionError, "bounded limit"):
                inspect_model(path)

            nested = value = {}
            for _ in range(20):
                child = {}
                value["child"] = child
                value = child
            config_path.write_text(json.dumps(nested), encoding="utf-8")
            with self.assertRaisesRegex(ModelInspectionError, "bounded limits"):
                inspect_model(path)


if __name__ == "__main__":
    unittest.main()
