import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from vllm_apple.qwen_image_21_conversion import (
    build_qwen_image_21_conversion_plan,
    build_qwen_image_21_streaming_conversion_plan,
)
from vllm_apple.types import GIB, HardwareInfo, MemoryInfo


def artifact() -> dict[str, object]:
    return {
        "pipeline_class": "QwenImage21Pipeline",
        "components": [
            {"name": "transformer", "role": "denoiser", "artifact_bytes": 14 * GIB},
            {"name": "text_encoder", "role": "text_encoder", "artifact_bytes": 17 * GIB},
            {"name": "vae", "role": "vae", "artifact_bytes": GIB},
        ],
    }


def hardware(available_gib: int) -> HardwareInfo:
    return HardwareInfo(
        "Darwin", "arm64", "Apple M4", 10, 10, 10,
        MemoryInfo(32 * GIB, available_gib * GIB), True, "test",
    )


class QwenImage21ConversionTests(unittest.TestCase):
    def test_plan_accounts_for_largest_source_shard_during_conversion(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            source.joinpath("a.safetensors").write_bytes(b"12345678")
            with patch("shutil.disk_usage") as disk_usage:
                disk_usage.return_value.free = 100 * GIB
                plan = build_qwen_image_21_conversion_plan(
                    artifact(), hardware(32), source=source,
                    output=root / "output", conversion_ready=True,
                )
        self.assertEqual(plan["largest_source_shard_bytes"], 8)
        self.assertEqual(
            plan["estimated_conversion_peak_bytes"],
            plan["projected_steady_resident_bytes"] + 8,
        )
        self.assertEqual(
            plan["minimum_available_memory_bytes"],
            plan["estimated_conversion_peak_bytes"] + int(32 * GIB * 0.08),
        )
        self.assertTrue(plan["eligible"])
        self.assertFalse(plan["weights_loaded"])

    def test_plan_rejects_insufficient_dynamic_memory(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            source.joinpath("a.safetensors").write_bytes(b"12345678")
            with patch("shutil.disk_usage") as disk_usage:
                disk_usage.return_value.free = 100 * GIB
                plan = build_qwen_image_21_conversion_plan(
                    artifact(), hardware(20), source=source,
                    output=root / "output", conversion_ready=True,
                )
        self.assertFalse(plan["eligible"])
        self.assertIn("conversion_peak_exceeds_dynamic_memory_ceiling", plan["issues"])

    def test_plan_rejects_output_inside_source(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory)
            source.joinpath("a.safetensors").write_bytes(b"12345678")
            with self.assertRaisesRegex(ValueError, "outside"):
                build_qwen_image_21_conversion_plan(
                    artifact(), hardware(32), source=source,
                    output=source / "output", conversion_ready=True,
                )

    def test_streaming_plan_releases_each_component_before_the_next(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            for name, size in (("transformer", 10), ("text_encoder", 5)):
                component = source / name
                component.mkdir(parents=True)
                component.joinpath("model.safetensors").write_bytes(b"x" * size)
            with patch("shutil.disk_usage") as disk_usage:
                disk_usage.return_value.free = 100 * GIB
                plan = build_qwen_image_21_streaming_conversion_plan(
                    artifact(), hardware(32), source=source,
                    output=root / "output", conversion_ready=True,
                )
        transformer = plan["phases"][0]
        text_encoder = plan["phases"][1]
        self.assertEqual(transformer["largest_source_shard_bytes"], 10)
        self.assertEqual(text_encoder["largest_source_shard_bytes"], 5)
        self.assertEqual(
            plan["estimated_conversion_peak_bytes"],
            max(item["estimated_peak_bytes"] for item in plan["phases"]),
        )
        self.assertTrue(plan["eligible"])
