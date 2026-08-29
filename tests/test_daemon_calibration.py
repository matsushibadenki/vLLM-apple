import tempfile
import unittest
from pathlib import Path

from vllm_apple.daemon import apply_startup_kv_calibration
from vllm_apple.kv_calibration import default_calibration_report_path
from vllm_apple.long_context import save_long_context_report
from vllm_apple.model import InspectedModel
from vllm_apple.types import ModelMemorySpec


def model() -> InspectedModel:
    return InspectedModel(
        model_id="model",
        path=Path("/model"),
        config={},
        memory_spec=ModelMemorySpec("model", 1000, 100_000, model_max_context=8192),
        kv_dtype_bytes=2,
    )


def report(backend: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "evaluation_id": "c" * 24,
        "model_id": "model",
        "hardware_fingerprint": "hardware",
        "backend": backend,
        "stages": [
            {
                "status": "passed",
                "target_tokens": tokens,
                "actual_prompt_tokens": tokens,
                "state_bytes": tokens * 1000,
                "retrieval_score": 1.0,
            }
            for tokens in (1024, 2048, 4096)
        ],
    }


class DaemonCalibrationTests(unittest.TestCase):
    def test_applies_matching_vllm_calibration_and_exposes_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = default_calibration_report_path(
                "model", "hardware", "vllm_metal", "c" * 24, application_support=root
            )
            save_long_context_report(report("vllm_metal"), path)
            calibrated, provenance = apply_startup_kv_calibration(
                model(), "hardware", root=root
            )
        self.assertEqual(calibrated.memory_spec.kv_bytes_per_token, 1250)
        self.assertEqual(calibrated.memory_spec.model_max_context, 4096)
        self.assertEqual(provenance["status"], "applied")
        self.assertEqual(provenance["evaluation_id"], "c" * 24)

    def test_does_not_apply_mlx_report_to_vllm_daemon(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = default_calibration_report_path(
                "model", "hardware", "mlx_lm", "c" * 24, application_support=root
            )
            save_long_context_report(report("mlx_lm"), path)
            unchanged, provenance = apply_startup_kv_calibration(
                model(), "hardware", root=root
            )
        self.assertEqual(unchanged.memory_spec.kv_bytes_per_token, 100_000)
        self.assertEqual(provenance["status"], "not_found")


if __name__ == "__main__":
    unittest.main()
