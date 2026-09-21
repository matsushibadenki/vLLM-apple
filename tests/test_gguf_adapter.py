from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from tests.schema_validator import validate_instance
from vllm_apple.model import InspectedModel, ModelMemorySpec
from vllm_apple.optimizer import (
    ADAPTER_API_VERSION,
    AdapterUnavailableError,
    GGUFOptimizationAdapter,
)


def _model(path: Path) -> InspectedModel:
    (path / "config.json").write_text('{"torch_dtype":"float16"}', encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")
    return InspectedModel(
        "test-model",
        path,
        {"torch_dtype": "float16"},
        ModelMemorySpec("test-model", 1000, 1),
        2,
    )


class GGUFOptimizationAdapterTests(unittest.TestCase):
    def test_capability_is_fail_closed_without_explicit_converter(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            model = _model(Path(directory))
            capability = GGUFOptimizationAdapter().detect(model, "safetensors", "float16")
            self.assertFalse(capability.executable)
            self.assertIn("dependency_missing:llama.cpp-convert-hf-to-gguf", capability.issues)

    def test_builds_versioned_fixed_argument_q8_invocation(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            model = _model(source)
            converter = root / "convert_hf_to_gguf.py"
            converter.write_text("# test fixture\n", encoding="utf-8")
            converter.chmod(0o700)
            adapter = GGUFOptimizationAdapter(
                converter,
                "b6123",
                python_executable=sys.executable,
                supported_converter_versions=("b6123",),
            )
            invocation = adapter.build_export_invocation(
                model,
                root / "artifact",
                output_type="q8_0",
                maximum_output_bytes=1000,
            )
            self.assertEqual(invocation.adapter_api_version, ADAPTER_API_VERSION)
            self.assertEqual(invocation.implementation_version, "1.0.0")
            self.assertEqual(invocation.converter_version, "b6123")
            self.assertEqual(invocation.estimated_output_bytes, 600)
            self.assertEqual(
                invocation.command,
                (
                    str(Path(sys.executable).absolute()),
                    str(converter.resolve()),
                    str(source.resolve()),
                    "--outfile",
                    "model.gguf",
                    "--outtype",
                    "q8_0",
                ),
            )
            self.assertEqual(invocation.to_dict()["schema_version"], 1)
            schema = Path("schemas/optimizer/gguf-export-invocation-v1.schema.json")
            validate_instance(invocation.to_dict(), json.loads(schema.read_text()))

    def test_rejects_unknown_version_output_and_small_budget(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            model = _model(source)
            converter = root / "convert.py"
            converter.write_text("# fixture\n", encoding="utf-8")
            converter.chmod(0o700)
            unsupported = GGUFOptimizationAdapter(converter, "main", python_executable=sys.executable)
            with self.assertRaises(AdapterUnavailableError):
                unsupported.build_export_invocation(
                    model, root / "out-a", output_type="f16", maximum_output_bytes=2000
                )
            unknown_build = GGUFOptimizationAdapter(
                converter,
                "b6124",
                python_executable=sys.executable,
                supported_converter_versions=("b6123",),
            )
            self.assertIn(
                "dependency_version_unsupported:llama.cpp:b6124",
                unknown_build.detect(model, "safetensors", "float16").issues,
            )
            adapter = GGUFOptimizationAdapter(
                converter,
                "b6123",
                python_executable=sys.executable,
                supported_converter_versions=("b6123",),
            )
            with self.assertRaises(ValueError):
                adapter.build_export_invocation(
                    model, root / "out-b", output_type="q4_k_m", maximum_output_bytes=2000
                )
            with self.assertRaises(ValueError):
                adapter.build_export_invocation(
                    model, root / "out-c", output_type="f16", maximum_output_bytes=1000
                )

    def test_rejects_group_or_world_writable_converter(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            model = _model(source)
            converter = root / "convert.py"
            converter.write_text("# fixture\n", encoding="utf-8")
            converter.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IWGRP)
            capability = GGUFOptimizationAdapter(
                converter,
                "b6123",
                python_executable=sys.executable,
                supported_converter_versions=("b6123",),
            ).detect(model, "safetensors", "float16")
            self.assertFalse(capability.executable)
            self.assertIn("converter_path_unsafe", capability.issues)

    def test_converter_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            real = root / "real.py"
            real.write_text("# fixture\n", encoding="utf-8")
            real.chmod(0o700)
            link = root / "convert.py"
            os.symlink(real, link)
            source = root / "source"
            source.mkdir()
            capability = GGUFOptimizationAdapter(
                link,
                "b6123",
                python_executable=sys.executable,
                supported_converter_versions=("b6123",),
            ).detect(_model(source), "safetensors", "float16")
            self.assertFalse(capability.executable)
            self.assertIn("converter_path_unsafe", capability.issues)


if __name__ == "__main__":
    unittest.main()
