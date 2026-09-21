import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from vllm_apple.generative_collector import GenerationTelemetryEvent
from vllm_apple.generative_evaluation import GenerativeEvaluationProvenance
from vllm_apple.generative_qualification import (
    GenerativeArtifactComponent,
    build_generative_qualification_plan,
)
from vllm_apple.generative_qualification_runner import (
    resolve_generative_qualification_mode,
    run_generative_qualification,
    wait_for_memory_pressure_recovery,
)
from vllm_apple.types import GIB, HardwareInfo, MemoryInfo


class FakeAdapter:
    calls = 0
    requests = []

    def __init__(self, command, **kwargs):
        self.command = command
        self.index = FakeAdapter.calls
        FakeAdapter.calls += 1
        request_path = Path(command[command.index("--request") + 1])
        self.request = json.loads(request_path.read_text())
        FakeAdapter.requests.append(self.request)

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
            self.request["width"],
            self.request["height"],
            self.request["frames"],
            f"{self.index + 1:064x}",
        )


class GenerativeQualificationRunnerTests(unittest.TestCase):
    def test_two_phase_waits_for_encoder_before_generation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            hardware = HardwareInfo(
                "Darwin", "arm64", "Apple M4", 10, 10, 10,
                MemoryInfo(32 * GIB, 28 * GIB), True, "test",
            )
            plan = build_generative_qualification_plan(
                candidate_id="qwen-image-2.1", artifact_bytes=8 * GIB,
                estimated_resident_bytes=16 * GIB, hardware=hardware,
                target=root, quantization="int8",
                components=(
                    GenerativeArtifactComponent("transformer", "denoiser", 6 * GIB, 10 * GIB),
                    GenerativeArtifactComponent("text", "text_encoder", GIB, 4 * GIB),
                    GenerativeArtifactComponent("vae", "vae", GIB, 2 * GIB),
                ), width=512, height=512, steps=1,
            )
            order = []

            class Process:
                pid = 12345

                def __init__(self, command, **kwargs):
                    self.command = command
                    self.stdout = kwargs["stdout"]
                    order.append("encoder-start")

                def wait(self, timeout):
                    handoff = Path(self.command[self.command.index("--handoff-root") + 1])
                    (handoff / "prompt-embeddings.json").write_text("stub")
                    self.stdout.write(json.dumps({
                        "schema_version": 1,
                        "phase": "text_encoder",
                        "elapsed_ms": 5.0,
                        "peak_rss_bytes": 9 * GIB,
                        "memory_pressure": "normal",
                        "thermal_state": "nominal",
                    }).encode())
                    self.stdout.flush()
                    order.append("encoder-exit")
                    return 0

            class Adapter(FakeAdapter):
                def __init__(self, command, **kwargs):
                    order.append("generation-start")
                    self.assert_handoff(command)
                    super().__init__(command, **kwargs)

                @staticmethod
                def assert_handoff(command):
                    assert "--phase-handoff" in command
                    assert Path(command[command.index("--phase-handoff") + 1]).is_file()

            FakeAdapter.calls = 0
            FakeAdapter.requests = []
            with patch(
                "vllm_apple.generative_qualification_runner.subprocess.Popen",
                side_effect=Process,
            ):
                report = run_generative_qualification(
                    plan, workspace_root=root, model_root=model,
                    private_root=root / "private", report_path=root / "report.json",
                    prompt="private test prompt", sample_count=2,
                    worker_command=("python", "-m", "worker"),
                    phase_encoder_command=("python", "-m", "encoder"),
                    provenance=GenerativeEvaluationProvenance(
                        "Darwin", "arm64", "Apple M4", 10, 32 * GIB,
                        "test", "1.0.0", "test", 8 * GIB, "int8", None, "test/model",
                    ), adapter_factory=Adapter, pressure_probe=lambda: "normal",
                    recovery_poll_seconds=0.001,
                )
            self.assertTrue(report.passed)
            self.assertEqual(report.maximum_peak_rss_bytes, 9 * GIB)
            self.assertEqual(order, ["encoder-start", "encoder-exit", "generation-start"] * 2)
            self.assertFalse(list((root / "private").glob("handoff-*")))

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
            FakeAdapter.requests = []
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
            self.assertEqual({item["mode"] for item in FakeAdapter.requests}, {"text-to-image"})

    def test_video_plan_defaults_to_text_to_video_with_bounded_profile(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            private = root / "private"
            model.mkdir()
            hardware = HardwareInfo(
                "Darwin", "arm64", "Apple M4", 10, 10, 10,
                MemoryInfo(32 * GIB, 28 * GIB), True, "test",
            )
            components = (
                GenerativeArtifactComponent("transformer", "denoiser", 6 * GIB, 12 * GIB),
                GenerativeArtifactComponent("text", "text_encoder", GIB, 3 * GIB),
                GenerativeArtifactComponent("vae", "vae", GIB, 3 * GIB),
            )
            plan = build_generative_qualification_plan(
                candidate_id="wan2.2-ti2v-5b",
                artifact_bytes=8 * GIB,
                estimated_resident_bytes=18 * GIB,
                hardware=hardware,
                target=root,
                quantization="int4",
                components=components,
            )
            FakeAdapter.calls = 0
            FakeAdapter.requests = []
            report = run_generative_qualification(
                plan,
                workspace_root=root,
                model_root=model,
                private_root=private,
                report_path=root / "report.json",
                prompt="A robot walking through a quiet workshop",
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
        self.assertEqual((report.samples[0].output_width, report.samples[0].output_height), (640, 384))
        self.assertEqual(report.samples[0].output_frames, 33)
        self.assertEqual({item["mode"] for item in FakeAdapter.requests}, {"text-to-video"})

    def test_explicit_unsupported_mode_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            plan = build_generative_qualification_plan(
                candidate_id="wan2.2-ti2v-5b",
                artifact_bytes=8 * GIB,
                estimated_resident_bytes=18 * GIB,
                hardware=HardwareInfo(
                    "Darwin", "arm64", "Apple M4", 10, 10, 10,
                    MemoryInfo(32 * GIB, 28 * GIB), True, "test",
                ),
                target=Path(directory),
                quantization="int4",
                components=(
                    GenerativeArtifactComponent("transformer", "denoiser", 6 * GIB, 12 * GIB),
                    GenerativeArtifactComponent("text", "text_encoder", GIB, 3 * GIB),
                    GenerativeArtifactComponent("vae", "vae", GIB, 3 * GIB),
                ),
            )
        with self.assertRaisesRegex(ValueError, "not supported"):
            resolve_generative_qualification_mode(plan, "text-to-image")

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
