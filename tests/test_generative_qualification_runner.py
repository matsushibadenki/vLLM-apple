import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vllm_apple.generative_collector import GenerationTelemetryEvent
from vllm_apple.generative_qualification import (
    GenerativeArtifactComponent,
    build_generative_qualification_plan,
)
from vllm_apple.generative_qualification_runner import (
    run_generative_qualification,
    wait_for_memory_pressure_recovery,
)
from vllm_apple.generative_evaluation import GenerativeEvaluationProvenance
from vllm_apple.types import GIB, HardwareInfo, MemoryInfo


class FakeAdapter:
    calls = 0

    def __init__(self, command, **kwargs):
        self.command = command
        self.index = FakeAdapter.calls
        FakeAdapter.calls += 1

    def events(self):
        yield GenerationTelemetryEvent(
            "started", 0, 8 * GIB, "normal", "nominal"
        )
        yield GenerationTelemetryEvent(
            "completed",
            1000,
            8 * GIB,
            "normal",
            "nominal",
            512,
            512,
            1,
            f"{self.index + 1:064x}",
        )


class GenerativeQualificationRunnerTests(unittest.TestCase):
    def test_two_samples_are_collected_and_private_report_is_saved(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            private = root / "private"
            model.mkdir()
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
                GenerativeArtifactComponent("transformer", "denoiser", 6 * GIB, 10 * GIB),
                GenerativeArtifactComponent("text", "text_encoder", GIB, 4 * GIB),
                GenerativeArtifactComponent("vae", "vae", GIB, 2 * GIB),
            )
            plan = build_generative_qualification_plan(
                candidate_id="flux2-klein-9b-base",
                artifact_bytes=8 * GIB,
                estimated_resident_bytes=16 * GIB,
                hardware=hardware,
                target=root,
                quantization="int4",
                components=components,
                width=512,
                height=512,
                steps=2,
            )
            FakeAdapter.calls = 0
            report_path = root / "reports" / "report.json"
            report = run_generative_qualification(
                plan,
                workspace_root=root,
                model_root=model,
                private_root=private,
                report_path=report_path,
                prompt="private test prompt",
                sample_count=2,
                worker_command=("python", "-m", "worker"),
                provenance=GenerativeEvaluationProvenance(
                    "Darwin", "arm64", "Apple M4", 10, 32 * GIB,
                    "test", "1.0.0", "test", 8 * GIB, "int4", None, "test/model"
                ),
                adapter_factory=FakeAdapter,
                pressure_probe=lambda: "normal",
                recovery_poll_seconds=0.001,
            )
            self.assertTrue(report.passed)
            self.assertEqual(report.sample_count, 2)
            self.assertTrue(report_path.is_file())
            self.assertNotIn("private test prompt", report_path.read_text())
            self.assertFalse(list(private.glob("request-*.json")))

    def test_memory_recovery_requires_two_consecutive_normal_observations(self) -> None:
        pressures = iter(("warning", "normal", "warning", "normal", "normal"))
        now = [0.0]
        wait_for_memory_pressure_recovery(
            pressure_probe=lambda: next(pressures),
            timeout_seconds=10,
            poll_seconds=1,
            monotonic=lambda: now[0],
            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
        )
        self.assertEqual(now[0], 4.0)

    def test_memory_recovery_timeout_fails_closed(self) -> None:
        now = [0.0]
        with self.assertRaisesRegex(RuntimeError, "did not recover"):
            wait_for_memory_pressure_recovery(
                pressure_probe=lambda: "warning",
                timeout_seconds=2,
                poll_seconds=1,
                monotonic=lambda: now[0],
                sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
            )


if __name__ == "__main__":
    unittest.main()
