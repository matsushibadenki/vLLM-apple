import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from tests.schema_validator import validate_instance
from vllm_apple.architecture_registry import describe_architecture
from vllm_apple.cli import main
from vllm_apple.model import inspect_model
from vllm_apple.model_recommendation import build_model_recommendation
from vllm_apple.types import GIB, HardwareInfo, MemoryInfo


def hardware() -> HardwareInfo:
    return HardwareInfo(
        platform="Darwin",
        architecture="arm64",
        soc="Apple M4",
        physical_cpu_count=10,
        logical_cpu_count=10,
        gpu_core_count=10,
        memory=MemoryInfo(total_bytes=32 * GIB, available_bytes=24 * GIB),
        is_apple_silicon=True,
        os_version="test",
    )


def model(directory: str) -> Path:
    root = Path(directory) / "model"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "hidden_size": 32,
                "max_position_embeddings": 4096,
                "torch_dtype": "float16",
            }
        )
    )
    (root / "weights.safetensors").write_bytes(b"x" * 1024)
    return root


class ModelRecommendationTests(unittest.TestCase):
    def test_report_is_schema_valid_and_does_not_expose_config_or_resolved_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = model(directory)
            report = build_model_recommendation(
                inspect_model(root), hardware(), backend="mlx_lm"
            ).to_dict()
        schema = json.loads(
            Path("schemas/runtime/model-recommendation-v2.schema.json").read_text()
        )
        validate_instance(report, schema)
        self.assertFalse(report["runnable"])
        self.assertFalse(report["declared_features_match"])
        self.assertEqual(report["recognition"], "recognized")
        self.assertEqual(report["recommended_tier"], "balanced")
        self.assertEqual(report["recommended_context_tokens"], 4096)
        self.assertNotIn("config", report)
        self.assertNotIn("path", report)

    def test_cli_returns_machine_readable_recommendation_without_loading_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = model(directory)
            output = StringIO()
            with patch("vllm_apple.cli.detect_hardware", return_value=hardware()), redirect_stdout(
                output
            ):
                exit_code = main(["inspect-model", str(root), "--backend", "mlx_lm"])
        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertFalse(payload["backend_compatible"])
        self.assertEqual(payload["qualification"], "unverified")
        self.assertTrue(payload["fits_memory"])


    def test_all_declared_features_only_enable_validation_not_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inspected = inspect_model(model(directory))
            features = frozenset(describe_architecture(inspected.config)["required_features"])
            report = build_model_recommendation(
                inspected, hardware(), backend="mlx_lm", available_features=features
            ).to_dict()
        self.assertTrue(report["declared_features_match"])
        self.assertTrue(report["eligible_for_validation"])
        self.assertFalse(report["runnable"])
        self.assertFalse(report["backend_compatible"])
        self.assertEqual(report["compatibility_issue"], "backend_execution_unverified")

    def test_unknown_model_and_unsupported_structure_cannot_be_overridden_by_features(self):
        for changes, expected in (
            ({"model_type": "custom_qwen3"}, "architecture_unknown"),
            ({"layer_types": ["linear_attention"] * 2}, "architecture_invalid_field:layer_types"),
        ):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                root = model(directory)
                config_path = root / "config.json"
                config = json.loads(config_path.read_text())
                features = frozenset(describe_architecture(config)["required_features"])
                config.update(changes)
                config_path.write_text(json.dumps(config))
                report = build_model_recommendation(
                    inspect_model(root), hardware(), backend="mlx_lm", available_features=features
                ).to_dict()
                self.assertFalse(report["eligible_for_validation"])
                self.assertFalse(report["runnable"])
                self.assertEqual(report["compatibility_issue"], expected)

    def test_matching_features_do_not_override_requested_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            inspected = inspect_model(model(directory))
            features = frozenset(describe_architecture(inspected.config)["required_features"])
            report = build_model_recommendation(
                inspected, hardware(), backend="mlx_lm", available_features=features,
                requested_modes=frozenset({"vision"}),
            )
        self.assertFalse(report.eligible_for_validation)
        self.assertEqual(report.compatibility_issue, "model_missing_requested_modes:vision")

    def test_memory_failure_prevents_validation_even_when_features_match(self):
        with tempfile.TemporaryDirectory() as directory:
            inspected = inspect_model(model(directory))
            features = frozenset(describe_architecture(inspected.config)["required_features"])
            constrained = replace(
                hardware(), memory=MemoryInfo(total_bytes=32 * GIB, available_bytes=GIB)
            )
            report = build_model_recommendation(
                inspected, constrained, backend="mlx_lm", available_features=features
            )
        self.assertTrue(report.declared_features_match)
        self.assertFalse(report.fits_memory)
        self.assertFalse(report.eligible_for_validation)

    def test_invalid_declarations_are_rejected_before_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            inspected = inspect_model(model(directory))
            with self.assertRaisesRegex(ValueError, "backend feature declarations"):
                build_model_recommendation(
                    inspected, hardware(), backend="mlx_lm", available_features=frozenset({True})
                )

    def test_cli_feature_flags_cannot_certify_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = model(directory)
            features = describe_architecture(json.loads((root / "config.json").read_text()))[
                "required_features"
            ]
            arguments = ["inspect-model", str(root)]
            for feature in features:
                arguments.extend(["--feature", feature])
            output = StringIO()
            with patch("vllm_apple.cli.detect_hardware", return_value=hardware()), redirect_stdout(output):
                self.assertEqual(main(arguments), 1)
            payload = json.loads(output.getvalue())
            self.assertTrue(payload["eligible_for_validation"])
            self.assertFalse(payload["runnable"])


if __name__ == "__main__":
    unittest.main()
