import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from tests.schema_validator import validate_instance
from vllm_apple.artifact_admission import assess_artifact_admission
from vllm_apple.cli import main
from vllm_apple.types import GIB, HardwareInfo, MemoryInfo


def hardware(total_gib: int, available_gib: int) -> HardwareInfo:
    return HardwareInfo(
        platform="Darwin",
        architecture="arm64",
        soc="Apple M",
        physical_cpu_count=10,
        logical_cpu_count=10,
        gpu_core_count=16,
        memory=MemoryInfo(total_gib * GIB, available_gib * GIB),
        is_apple_silicon=True,
        os_version="test",
    )


class ArtifactAdmissionTests(unittest.TestCase):
    def test_large_qwen_candidate_is_rejected_before_download_on_32_gib(self) -> None:
        result = assess_artifact_admission(
            model="Qwen/Qwen3.8-Flash-Next",
            artifact_bytes=105 * GIB,
            estimated_resident_bytes=105 * GIB,
            hardware=hardware(32, 20),
            disk_free_bytes=500 * GIB,
        )
        self.assertFalse(result.fits_memory)
        self.assertTrue(result.fits_disk)
        self.assertFalse(result.eligible)
        validate_instance(
            result.to_dict(),
            json.loads(Path("schemas/runtime/artifact-admission-v1.schema.json").read_text()),
        )

    def test_disk_staging_headroom_is_required(self) -> None:
        result = assess_artifact_admission(
            model="Qwen/Qwen3.8-Flash-Next",
            artifact_bytes=100 * GIB,
            estimated_resident_bytes=100 * GIB,
            hardware=hardware(192, 180),
            disk_free_bytes=104 * GIB,
        )
        self.assertTrue(result.fits_memory)
        self.assertFalse(result.fits_disk)
        self.assertEqual(result.disk_required_bytes, 105 * GIB)

    def test_cli_rejects_ineligible_artifact_without_downloading(self) -> None:
        with TemporaryDirectory() as directory:
            output = StringIO()
            with (
                patch("vllm_apple.cli.detect_hardware", return_value=hardware(32, 20)),
                redirect_stdout(output),
            ):
                exit_code = main(
                    [
                        "artifact-admission",
                        "--model",
                        "Qwen/Qwen3.8-Flash-Next",
                        "--artifact-gib",
                        "105",
                        "--resident-gib",
                        "105",
                        "--target",
                        directory,
                    ]
                )
        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertFalse(payload["eligible"])
        self.assertFalse(payload["fits_memory"])
        self.assertEqual(payload["model"], "Qwen/Qwen3.8-Flash-Next")

    def test_model_identifier_rejects_control_characters(self) -> None:
        with self.assertRaisesRegex(ValueError, "model identifier"):
            assess_artifact_admission(
                model="Qwen/model\nforged",
                artifact_bytes=GIB,
                estimated_resident_bytes=GIB,
                hardware=hardware(32, 20),
                disk_free_bytes=100 * GIB,
            )

    def test_cli_accepts_exact_byte_sizes_for_evidence_binding(self) -> None:
        with TemporaryDirectory() as directory:
            output = StringIO()
            with (
                patch("vllm_apple.cli.detect_hardware", return_value=hardware(192, 180)),
                patch(
                    "vllm_apple.artifact_admission.shutil.disk_usage",
                    return_value=SimpleNamespace(free=500 * GIB),
                ),
                redirect_stdout(output),
            ):
                exit_code = main(
                    [
                        "artifact-admission",
                        "--model",
                        "Qwen/Qwen3.8-Flash-Next",
                        "--artifact-bytes",
                        str(105 * GIB + 17),
                        "--resident-bytes",
                        str(105 * GIB + 29),
                        "--target",
                        directory,
                    ]
                )
        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["fits_disk"])
        self.assertEqual(payload["artifact_bytes"], 105 * GIB + 17)
        self.assertEqual(payload["estimated_resident_bytes"], 105 * GIB + 29)


if __name__ == "__main__":
    unittest.main()
