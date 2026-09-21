import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vllm_apple.qwen_image_21_conversion_worker import (
    convert_qwen_image_21_int8_atomic,
)


def write_artifact(root: Path, *, quantize_both: bool) -> None:
    root.joinpath("model_index.json").write_text(
        json.dumps({"_class_name": "QwenImage21Pipeline"})
    )
    for component in ("transformer", "text_encoder", "vae"):
        directory = root / component
        directory.mkdir()
        config = {}
        if component != "vae" and (quantize_both or component == "transformer"):
            config["quantization_config"] = {
                "quant_method": "torchao",
                "quant_type": {
                    "default": {"_type": "Int8WeightOnlyConfig", "_version": 2, "_data": {}}
                },
            }
        directory.joinpath("config.json").write_text(json.dumps(config))
        directory.joinpath("model.safetensors").write_bytes(component.encode())


class QwenImage21ConversionWorkerTests(unittest.TestCase):
    def test_valid_staging_artifact_is_atomically_promoted(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            output = root / "output"

            def converter(_source: Path, staging: Path) -> None:
                write_artifact(staging, quantize_both=True)

            report = convert_qwen_image_21_int8_atomic(
                source, output, converter=converter
            )
            self.assertTrue(output.is_dir())
            self.assertTrue(report["passed"])
            self.assertEqual(set(report["quantized_components"]), {"transformer", "text_encoder"})
            self.assertEqual(len(report["artifact_root_sha256"]), 64)

    def test_incomplete_quantization_removes_staging_and_output(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            output = root / "output"

            def converter(_source: Path, staging: Path) -> None:
                write_artifact(staging, quantize_both=False)

            with self.assertRaisesRegex(RuntimeError, "both required components"):
                convert_qwen_image_21_int8_atomic(source, output, converter=converter)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".output.staging-*")), [])

    def test_converter_failure_removes_staging(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            output = root / "output"

            def converter(_source: Path, _staging: Path) -> None:
                raise RuntimeError("conversion failed")

            with self.assertRaisesRegex(RuntimeError, "conversion failed"):
                convert_qwen_image_21_int8_atomic(source, output, converter=converter)
            self.assertEqual(list(root.glob(".output.staging-*")), [])
