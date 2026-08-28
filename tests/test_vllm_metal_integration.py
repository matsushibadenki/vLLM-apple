import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from tests.schema_validator import validate_instance
from vllm_apple.cli import main
from vllm_apple.vllm_metal_integration import inspect_vllm_metal_integration


class VLLMMetalIntegrationInspectionTests(unittest.TestCase):
    def source_tree(self, root: Path, *, hooked: bool) -> Path:
        package = root / "vllm_metal"
        python_path = package / "attention" / "impls" / "sdpa.py"
        cpp_path = package / "metal" / "paged_ops.cpp"
        python_path.parent.mkdir(parents=True)
        cpp_path.parent.mkdir(parents=True)
        python_source = "ops.paged_attention_primitive(query)\n"
        cpp_source = (
            "dispatch_paged_attention_v2_online();\n"
            "paged_attention_primitive();\n"
            "dispatch_paged_attention_tiled();\n"
            "constexpr int NUM_THREADS = 256;"
        )
        if hooked:
            python_source += (
                "from vllm_apple.vllm_middleware import invoke_paged_attention_kernel\n"
            )
            cpp_source += (
                "\n#define VLLM_APPLE_THREAD_CONFIG_ABI_V1 1\n"
                "int score_width; int softmax_width; int output_width;\n"
            )
        python_path.write_text(python_source)
        cpp_path.write_text(cpp_source)
        return root

    def test_current_native_v2_is_detected_but_not_falsely_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inspection = inspect_vllm_metal_integration(
                self.source_tree(Path(directory), hooked=False)
            )
        self.assertTrue(inspection.native_v2_detected)
        self.assertFalse(inspection.compatible)
        self.assertEqual(
            inspection.issues,
            (
                "python_callsite_hook_missing",
                "dynamic_thread_configuration_abi_missing",
            ),
        )
        schema = json.loads(
            Path("schemas/runtime/vllm-metal-integration-v1.schema.json").read_text()
        )
        validate_instance(inspection.to_dict(), schema)

    def test_complete_python_and_cpp_abi_is_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inspection = inspect_vllm_metal_integration(
                self.source_tree(Path(directory), hooked=True)
            )
        self.assertTrue(inspection.compatible)
        self.assertEqual(inspection.issues, ())

    def test_cli_returns_one_for_safe_native_v2_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.source_tree(Path(directory), hooked=False)
            output = StringIO()
            with redirect_stdout(output):
                status = main(["vllm-metal-integration-inspect", str(root)])
        self.assertEqual(status, 1)
        self.assertFalse(json.loads(output.getvalue())["compatible"])


if __name__ == "__main__":
    unittest.main()
