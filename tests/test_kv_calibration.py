import json
import tempfile
import unittest
from pathlib import Path

from vllm_apple.kv_calibration import (
    default_calibration_report_path,
    discover_latest_kv_calibration,
    load_kv_calibration,
)
from vllm_apple.long_context import save_long_context_report
from vllm_apple.types import ModelMemorySpec


def report(stages: list[dict[str, object]], model_id: str = "model") -> dict[str, object]:
    return {
        "schema_version": 1,
        "evaluation_id": "a" * 24,
        "model_id": model_id,
        "hardware_fingerprint": "hardware",
        "backend": "mlx_lm",
        "stages": stages,
    }


def stage(tokens: int, state_bytes: int) -> dict[str, object]:
    return {
        "status": "passed",
        "target_tokens": tokens,
        "actual_prompt_tokens": tokens,
        "state_bytes": state_bytes,
        "retrieval_score": 1.0,
    }


class KVCalibrationTests(unittest.TestCase):
    def _load(self, payload: dict[str, object]):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            path.chmod(0o600)
            return load_kv_calibration(
                path,
                expected_model_id="model",
                expected_hardware_fingerprint="hardware",
                expected_backend="mlx_lm",
            )

    def test_uses_worst_observation_with_margin_and_caps_context(self) -> None:
        calibration = self._load(
            report([stage(512, 512_000), stage(2048, 1_843_200), stage(4096, 3_276_800)])
        )
        self.assertEqual(calibration.observed_bytes_per_token, 1000)
        self.assertEqual(calibration.calibrated_bytes_per_token, 1250)
        calibrated = calibration.apply(ModelMemorySpec("model", 100, 100_000, model_max_context=8192))
        self.assertEqual(calibrated.kv_bytes_per_token, 1250)
        self.assertEqual(calibrated.model_max_context, 4096)

    def test_rejects_identity_sparse_and_non_monotonic_reports(self) -> None:
        with self.assertRaisesRegex(ValueError, "identity"):
            self._load(report([stage(1, 1)] * 3, model_id="other"))
        with self.assertRaisesRegex(ValueError, "three"):
            self._load(report([stage(1024, 100), stage(4096, 400)]))
        with self.assertRaisesRegex(ValueError, "monotonically"):
            self._load(report([stage(1024, 100), stage(2048, 90), stage(4096, 400)]))

    def test_rejects_different_hardware(self) -> None:
        payload = report([stage(1024, 100), stage(2048, 200), stage(4096, 400)])
        payload["hardware_fingerprint"] = "other"
        with self.assertRaisesRegex(ValueError, "hardware fingerprint"):
            self._load(payload)

    def test_discovers_newest_compatible_private_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older = default_calibration_report_path(
                "model", "hardware", "mlx_lm", "a" * 24, application_support=root, timestamp_ns=1
            )
            newer = default_calibration_report_path(
                "model", "hardware", "mlx_lm", "b" * 24, application_support=root, timestamp_ns=2
            )
            save_long_context_report(
                report([stage(1024, 100), stage(2048, 200), stage(4096, 400)]),
                older,
            )
            newer_payload = report(
                [stage(1024, 200), stage(2048, 400), stage(4096, 800)]
            )
            newer_payload["evaluation_id"] = "b" * 24
            save_long_context_report(newer_payload, newer)
            older.touch()
            newer.touch()
            calibration, selected = discover_latest_kv_calibration(
                expected_model_id="model",
                expected_hardware_fingerprint="hardware",
                expected_backend="mlx_lm",
                application_support=root,
            )
            self.assertEqual(selected, newer)
            self.assertEqual(calibration.evaluation_id, "b" * 24)

    def test_discovery_skips_invalid_newest_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = default_calibration_report_path(
                "model", "hardware", "mlx_lm", "a" * 24, application_support=root, timestamp_ns=1
            )
            invalid = default_calibration_report_path(
                "model", "hardware", "mlx_lm", "b" * 24, application_support=root, timestamp_ns=2
            )
            save_long_context_report(
                report([stage(1024, 100), stage(2048, 200), stage(4096, 400)]),
                valid,
            )
            save_long_context_report({"schema_version": 1}, invalid)
            valid.touch()
            invalid.touch()
            _, selected = discover_latest_kv_calibration(
                expected_model_id="model",
                expected_hardware_fingerprint="hardware",
                expected_backend="mlx_lm",
                application_support=root,
            )
            self.assertEqual(selected, valid)


if __name__ == "__main__":
    unittest.main()
