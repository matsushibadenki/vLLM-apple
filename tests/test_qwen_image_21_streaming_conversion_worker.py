import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vllm_apple.qwen_image_21_streaming_conversion_worker import (
    convert_qwen_image_21_int8_streaming_atomic,
)


def config() -> dict[str, object]:
    return {
        "quantization_config": {
            "quant_method": "torchao",
            "quant_type": {
                "default": {"_type": "Int8WeightOnlyConfig", "_version": 2, "_data": {}}
            },
        }
    }


class QwenImage21StreamingConversionWorkerTests(unittest.TestCase):
    def test_components_run_in_order_and_promote_atomically(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            source.joinpath("model_index.json").write_text(
                json.dumps({"_class_name": "QwenImage21Pipeline"})
            )
            for name in ("transformer", "text_encoder", "vae"):
                component = source / name
                component.mkdir()
                component.joinpath("config.json").write_text("{}")
                component.joinpath("model.safetensors").write_bytes(name.encode())
            output = root / "output"
            calls = []

            def runner(_source: Path, staging: Path, component: str) -> None:
                calls.append(component)
                destination = staging / component
                destination.mkdir()
                destination.joinpath("config.json").write_text(json.dumps(config()))
                destination.joinpath("model.safetensors").write_bytes(component.encode())

            report = convert_qwen_image_21_int8_streaming_atomic(
                source, output, component_runner=runner
            )
            self.assertEqual(calls, ["transformer", "text_encoder"])
            self.assertEqual(report["completed_components"], calls)
            self.assertTrue(output.joinpath("vae/model.safetensors").is_file())
            self.assertTrue(report["passed"])

    def test_second_phase_failure_removes_all_staging(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            source.joinpath("model_index.json").write_text(
                json.dumps({"_class_name": "QwenImage21Pipeline"})
            )
            for name in ("transformer", "text_encoder", "vae"):
                component = source / name
                component.mkdir()
                component.joinpath("config.json").write_text("{}")
                component.joinpath("model.safetensors").write_bytes(name.encode())
            output = root / "output"

            def runner(_source: Path, staging: Path, component: str) -> None:
                if component == "text_encoder":
                    raise RuntimeError("second phase failed")
                destination = staging / component
                destination.mkdir()
                destination.joinpath("config.json").write_text(json.dumps(config()))
                destination.joinpath("model.safetensors").write_bytes(b"x")

            with self.assertRaisesRegex(RuntimeError, "second phase"):
                convert_qwen_image_21_int8_streaming_atomic(
                    source, output, component_runner=runner
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".output.staging-*")), [])
