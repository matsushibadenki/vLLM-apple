import json
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.schema_validator import validate_instance
from vllm_apple.generative_evaluation import (
    GenerativeSampleEvidence,
    evaluate_generative_qualification,
    save_generative_evaluation_report,
)
from vllm_apple.generative_qualification import (
    GenerativeArtifactComponent,
    build_generative_qualification_plan,
)
from vllm_apple.types import GIB, HardwareInfo, MemoryInfo


def make_plan():
    hardware = HardwareInfo(
        "Darwin",
        "arm64",
        "Apple M4",
        10,
        10,
        10,
        MemoryInfo(32 * GIB, 28 * GIB),
        True,
        "test",
    )
    components = (
        GenerativeArtifactComponent("transformer", "denoiser", 6 * GIB, 14 * GIB),
        GenerativeArtifactComponent("encoder", "text_encoder", GIB, 2 * GIB),
        GenerativeArtifactComponent("vae", "vae", GIB, 2 * GIB),
    )
    directory = TemporaryDirectory()
    plan = build_generative_qualification_plan(
        candidate_id="wan2.2-ti2v-5b",
        artifact_bytes=8 * GIB,
        estimated_resident_bytes=18 * GIB,
        hardware=hardware,
        target=Path(directory.name),
        quantization="int4",
        components=components,
    )
    return directory, plan


def sample(index: int = 0, **overrides) -> GenerativeSampleEvidence:
    values = {
        "sample_index": index,
        "wall_time_ms": 10_000.0,
        "first_output_ms": 1_000.0,
        "peak_rss_bytes": 18 * GIB,
        "memory_pressure": "normal",
        "thermal_state": "nominal",
        "output_width": 640,
        "output_height": 360,
        "output_frames": 33,
        "output_sha256": "a" * 64,
    }
    values.update(overrides)
    return GenerativeSampleEvidence(**values)


class GenerativeEvaluationTests(unittest.TestCase):
    def test_safe_samples_build_schema_valid_privacy_preserving_report(self) -> None:
        directory, plan = make_plan()
        self.addCleanup(directory.cleanup)
        report = evaluate_generative_qualification(plan, (sample(), sample(1)))
        self.assertTrue(report.passed)
        self.assertFalse(report.stores_prompt)
        self.assertFalse(report.stores_output)
        self.assertEqual(report.minimum_frames_per_second, 3.3)
        schema = json.loads(
            Path("schemas/runtime/generative-evaluation-report-v1.schema.json").read_text()
        )
        validate_instance(report.to_dict(), schema)

    def test_unsafe_memory_thermal_and_retention_fail_closed(self) -> None:
        directory, plan = make_plan()
        self.addCleanup(directory.cleanup)
        report = evaluate_generative_qualification(
            plan,
            (
                sample(
                    peak_rss_bytes=30 * GIB,
                    memory_pressure="critical",
                    thermal_state="serious",
                    stores_output=True,
                ),
            ),
        )
        self.assertFalse(report.passed)
        self.assertTrue(report.stores_output)
        self.assertIn("sample:0:peak_rss_exceeds_hard_ceiling", report.issues)
        self.assertIn("sample:0:unsafe_memory_pressure", report.issues)
        self.assertIn("sample:0:unsafe_thermal_state", report.issues)
        self.assertIn("sample:0:private_content_retained", report.issues)

    def test_output_shape_and_sample_order_are_verified(self) -> None:
        directory, plan = make_plan()
        self.addCleanup(directory.cleanup)
        report = evaluate_generative_qualification(plan, (sample(output_frames=32),))
        self.assertFalse(report.passed)
        self.assertIn("sample:0:output_shape_mismatch", report.issues)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            evaluate_generative_qualification(plan, (sample(1),))

    def test_report_is_saved_atomically_with_private_permissions(self) -> None:
        directory, plan = make_plan()
        self.addCleanup(directory.cleanup)
        report = evaluate_generative_qualification(plan, (sample(),))
        destination = Path(directory.name) / "reports" / "generation.json"
        saved = save_generative_evaluation_report(report, destination)
        self.assertEqual(saved, destination.resolve())
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(destination.parent.stat().st_mode), 0o700)
        payload = json.loads(destination.read_text(encoding="utf-8"))
        self.assertTrue(payload["passed"])
        self.assertNotIn("prompt", payload)


if __name__ == "__main__":
    unittest.main()
