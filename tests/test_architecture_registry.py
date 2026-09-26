import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from tests.schema_validator import validate_instance
from vllm_apple.architecture_registry import ARCHITECTURE_RECIPES, inspect_architecture
from vllm_apple.cli import main
from vllm_apple.model import MAX_MODEL_CONFIG_BYTES, ModelInspectionError

FIXTURES = Path(__file__).parent / "fixtures" / "architectures"


class ArchitectureRegistryTests(unittest.TestCase):
    def inspect(self, config):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps(config))
            return inspect_architecture(root)

    def config(self, name="qwen3"):
        return json.loads((FIXTURES / f"{name}.json").read_text())

    def test_fixed_structural_fixtures_are_described_but_never_certified(self):
        schema = json.loads(Path("schemas/runtime/architecture-inspection-v1.schema.json").read_text())
        for name in ARCHITECTURE_RECIPES:
            with self.subTest(name=name):
                report = inspect_architecture(FIXTURES / f"{name}.json")
                validate_instance(report, schema)
                self.assertEqual(report["structure_status"], "described")
                self.assertEqual(report["backend"]["compatibility"], "unverified")
                self.assertEqual(set(report["qualification"].values()), {"unverified"})
                self.assertEqual(report["artifact_status"], "unverified")
                self.assertTrue(all(s["allocation_bytes"] is None for s in report["state_layout"]))

    def test_unknown_model_cannot_borrow_class_name_or_nested_backbone(self):
        for config in ({}, {"model_type": "custom_llama", "architectures": ["LlamaForCausalLM"]},
                       {"model_type": "custom_vlm", "text_config": self.config()}):
            with self.subTest(config=config):
                report = self.inspect(config)
                self.assertEqual(report["recognition"], "unknown")
                self.assertEqual(report["structure_status"], "unverified")
                self.assertEqual(report["layers"], [])
                self.assertEqual(report["issues"], ["architecture_unknown"])

    def test_explicit_head_dim_is_not_replaced_with_hidden_divided_by_heads(self):
        config = self.config()
        config.update(hidden_size=64, num_attention_heads=4, head_dim=32)
        report = self.inspect(config)
        self.assertEqual(report["layers"][0]["head_dim"], 32)

    def test_attention_head_relationships(self):
        for kv, expected in ((4, "mha"), (1, "mqa"), (2, "gqa")):
            config = self.config()
            config.update(num_attention_heads=4, num_key_value_heads=kv)
            self.assertEqual(self.inspect(config)["layers"][0]["attention"], expected)

    def test_gemma_window_keeps_global_layer_state_separate(self):
        report = self.inspect(self.config("gemma2"))
        self.assertEqual([s["kind"] for s in report["state_layout"]],
                         ["window_kv", "append_kv", "window_kv", "append_kv"])
        self.assertIsNone(report["state_layout"][1]["window_tokens"])
        self.assertFalse(report["state_layout"][0]["retention_verified"])

    def test_explicit_patterns_override_alternating_recipe(self):
        config = self.config("gemma2")
        config["layer_types"] = ["full_attention"] * 4
        report = self.inspect(config)
        self.assertEqual({s["kind"] for s in report["state_layout"]}, {"append_kv"})

    def test_qwen_window_boundary_and_disabled_window(self):
        config = self.config("qwen2")
        config.update(sliding_window=128, use_sliding_window=True, max_window_layers=1)
        report = self.inspect(config)
        self.assertEqual([s["kind"] for s in report["state_layout"]],
                         ["window_kv", "append_kv"])
        config["use_sliding_window"] = False
        self.assertEqual({s["kind"] for s in self.inspect(config)["state_layout"]}, {"append_kv"})

    def test_moe_reports_total_and_active_experts_separately(self):
        report = self.inspect(self.config("mixtral"))
        self.assertEqual(report["layers"][0]["experts"], 8)
        self.assertEqual(report["layers"][0]["active_experts"], 2)
        config = self.config("mixtral")
        config["num_experts_per_tok"] = 9
        self.assertEqual(self.inspect(config)["structure_status"], "unverified")

    def test_incomplete_or_unsupported_config_has_no_partial_layer_description(self):
        changes = [
            {"num_hidden_layers": True}, {"num_hidden_layers": 513},
            {"num_key_value_heads": 3}, {"head_dim": 0},
            {"layer_types": ["full_attention"]},
            {"layer_types": ["linear_attention"] * 2},
            {"layer_types": [{}, {}]},
            {"layer_types": ["sliding_attention"] * 2},
            {"sliding_window": 128}, {"use_sliding_window": "false"},
            {"use_sliding_window": True, "max_window_layers": 3, "sliding_window": 128},
            {"rope_scaling": {"rope_type": "custom"}},
            {"rope_scaling": {"rope_type": []}},
            {"rope_scaling": []}, {"text_config": {}},
        ]
        for change in changes:
            with self.subTest(change=change):
                config = self.config()
                config.update(change)
                report = self.inspect(config)
                self.assertEqual(report["structure_status"], "unverified")
                self.assertEqual(report["layers"], [])
                self.assertEqual(report["state_layout"], [])
                self.assertTrue(report["issues"])

    def test_unknown_rope_does_not_get_silently_treated_as_default(self):
        config = self.config()
        config["rope_scaling"] = {"rope_type": "yarn", "factor": 4}
        self.assertIn("rope_yarn", self.inspect(config)["required_features"])
        config["rope_scaling"]["rope_type"] = "new_rope"
        self.assertEqual(self.inspect(config)["structure_status"], "unverified")

    def test_config_hash_is_order_independent_and_changes_with_configuration(self):
        config = self.config()
        before = self.inspect(config)["config_sha256"]
        self.assertEqual(before, self.inspect(dict(reversed(list(config.items()))))["config_sha256"])
        config["head_dim"] += 1
        self.assertNotEqual(before, self.inspect(config)["config_sha256"])

    def test_bounded_read_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            for content in ('[]', '{', ' ' * (MAX_MODEL_CONFIG_BYTES + 1)):
                path.write_text(content)
                with self.assertRaises(ModelInspectionError):
                    inspect_architecture(path)
        with self.assertRaises(ModelInspectionError):
            self.inspect({"model_type": "unknown", "value": float("nan")})

    def test_cli_does_not_require_weights_or_detect_hardware(self):
        output = StringIO()
        with patch("vllm_apple.cli.detect_hardware", side_effect=AssertionError("hardware probe")):
            with redirect_stdout(output):
                code = main(["inspect-architecture", str(FIXTURES / "qwen3.json")])
        self.assertEqual(code, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["qualification"]["loadable"], "unverified")
        self.assertNotIn("path", report)

    def test_cli_unknown_and_invalid_file_exit_codes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text('{"model_type":"new_model"}')
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["inspect-architecture", str(path)]), 1)
            path.write_text('{')
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["inspect-architecture", str(path)]), 2)
            self.assertEqual(json.loads(output.getvalue())["error_code"], "architecture_inspection_failed")
