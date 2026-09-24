import json
import tempfile
import unittest
from pathlib import Path

from vllm_apple.mflux_qwen_promotion import load_mflux_qwen_promotion


class MFluxQwenPromotionTests(unittest.TestCase):
    def _report(self) -> dict:
        return {
            "schema_version": 1,
            "scope": "mflux_qwen_image_staged_real_prompt_streamed_denoising",
            "candidate_id": "qwen-image-2512",
            "artifact_root_sha256": "a" * 64,
            "passed": True,
            "uses_disjoint_encoder_processes": True,
            "uses_real_prompt": True,
            "uses_true_cfg": True,
            "guidance": 4.0,
            "denoise_steps": 20,
            "width": 160,
            "height": 160,
            "child_consumed_handoff_cleanup_verified": True,
            "private_handoff_cleanup_verified": True,
            "final_memory_pressure": "normal",
            "final_thermal_state": "fair",
            "encoder_peak_process_rss_bytes": 2_300_000_000,
            "transformer": {
                "passed": True,
                "width": 160,
                "height": 160,
                "steps": 20,
                "uses_real_prompt": True,
                "uses_true_cfg": True,
                "final_memory_pressure": "normal",
                "peak_mlx_bytes": 1_600_000_000,
                "peak_process_rss_bytes": 700_000_000,
                "step_reports": [
                    {
                        "completed_blocks": 120,
                        "latent_finite": True,
                        "memory_pressure": "normal",
                        "thermal_state": "nominal",
                    }
                    for _ in range(20)
                ],
            },
        }

    def _write(self, root: Path, report: dict) -> Path:
        path = root / "report.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        path.chmod(0o600)
        return path

    def test_accepts_all_normal_one_axis_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            promotion = load_mflux_qwen_promotion(
                self._write(Path(directory), self._report()),
                artifact_root_sha256="a" * 64,
                target_size=256,
            )
        self.assertEqual(promotion.baseline_size, 160)
        self.assertEqual(promotion.target_size, 256)
        self.assertEqual(promotion.minimum_available_bytes, 10_000_000_000)
        self.assertEqual(len(promotion.report_sha256), 64)

    def test_rejects_non_normal_step(self) -> None:
        report = self._report()
        report["transformer"]["step_reports"][7]["memory_pressure"] = "warning"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "strict contract"):
                load_mflux_qwen_promotion(
                    self._write(Path(directory), report),
                    artifact_root_sha256="a" * 64,
                    target_size=256,
                )

    def test_rejects_artifact_or_resolution_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(Path(directory), self._report())
            with self.assertRaisesRegex(ValueError, "strict contract"):
                load_mflux_qwen_promotion(path, artifact_root_sha256="b" * 64, target_size=256)
            with self.assertRaisesRegex(ValueError, "strict contract"):
                load_mflux_qwen_promotion(path, artifact_root_sha256="a" * 64, target_size=512)


if __name__ == "__main__":
    unittest.main()
