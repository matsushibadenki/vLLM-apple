import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vllm_apple.generative_collector import (
    MAX_TELEMETRY_EVENTS,
    GenerationTelemetryEvent,
    collect_generative_sample,
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
    directory = TemporaryDirectory()
    plan = build_generative_qualification_plan(
        candidate_id="wan2.2-ti2v-5b",
        artifact_bytes=8 * GIB,
        estimated_resident_bytes=18 * GIB,
        hardware=hardware,
        target=Path(directory.name),
        quantization="int4",
        components=(
            GenerativeArtifactComponent("transformer", "denoiser", 6 * GIB, 14 * GIB),
            GenerativeArtifactComponent("encoder", "text_encoder", GIB, 2 * GIB),
            GenerativeArtifactComponent("vae", "vae", GIB, 2 * GIB),
        ),
    )
    return directory, plan


def event(kind: str, elapsed: float, rss_gib: int = 10, **overrides):
    values = {
        "kind": kind,
        "elapsed_ms": elapsed,
        "process_rss_bytes": rss_gib * GIB,
        "memory_pressure": "normal",
        "thermal_state": "nominal",
    }
    values.update(overrides)
    return GenerationTelemetryEvent(**values)


def completed(elapsed: float = 10_000, **overrides):
    values = {
        "output_width": 640,
        "output_height": 360,
        "output_frames": 33,
        "output_sha256": "a" * 64,
    }
    values.update(overrides)
    return event("completed", elapsed, **values)


class GenerativeCollectorTests(unittest.TestCase):
    def test_stream_is_reduced_to_constant_memory_sample_evidence(self) -> None:
        directory, plan = make_plan()
        self.addCleanup(directory.cleanup)
        sample = collect_generative_sample(
            plan,
            sample_index=0,
            events=(
                event("started", 0, 8),
                event("progress", 500, 12, memory_pressure="warning"),
                event("first_output", 1_000, 14, thermal_state="fair"),
                completed(10_000, rss_gib=13),
            ),
        )
        self.assertEqual(sample.first_output_ms, 1_000)
        self.assertEqual(sample.peak_rss_bytes, 14 * GIB)
        self.assertEqual(sample.memory_pressure, "warning")
        self.assertEqual(sample.thermal_state, "fair")
        self.assertFalse(sample.stores_output)

    def test_missing_completion_and_non_monotonic_time_are_rejected(self) -> None:
        directory, plan = make_plan()
        self.addCleanup(directory.cleanup)
        with self.assertRaisesRegex(ValueError, "missing completion"):
            collect_generative_sample(plan, sample_index=0, events=(event("started", 0),))
        with self.assertRaisesRegex(ValueError, "monotonic"):
            collect_generative_sample(
                plan,
                sample_index=0,
                events=(event("started", 0), event("progress", 20), completed(10)),
            )

    def test_event_stream_is_strictly_bounded(self) -> None:
        directory, plan = make_plan()
        self.addCleanup(directory.cleanup)

        def too_many_events():
            yield event("started", 0)
            for index in range(MAX_TELEMETRY_EVENTS):
                yield event("progress", index + 1)
            yield completed(MAX_TELEMETRY_EVENTS + 1)

        with self.assertRaisesRegex(ValueError, "event limit"):
            collect_generative_sample(plan, sample_index=0, events=too_many_events())

    def test_output_metadata_cannot_leak_through_progress_events(self) -> None:
        with self.assertRaisesRegex(ValueError, "only allowed"):
            event("progress", 1, output_sha256="a" * 64)


if __name__ == "__main__":
    unittest.main()
