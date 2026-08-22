import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from tests.schema_validator import validate_instance
from vllm_apple.model import InspectedModel
from vllm_apple.optimizer import (
    ArtifactManifest,
    CalibrationManifest,
    OptimizationObjective,
    OptimizationPerformanceProfile,
    OptimizerErrorCode,
    OptimizationPathError,
    OptimizerFailure,
    OptimizerEventBus,
    OptimizerState,
    QualityBudget,
    Recoverability,
    ResourceBudget,
    build_dry_run_plan,
    profile_optimizer_io,
    validate_immutable_output_path,
)
from vllm_apple.optimizer.cli import main as optimizer_main
from vllm_apple.types import GIB, HardwareInfo, MemoryInfo, ModelMemorySpec


def hardware() -> HardwareInfo:
    return HardwareInfo(
        platform="Darwin",
        architecture="arm64",
        soc="Test M4",
        physical_cpu_count=10,
        logical_cpu_count=10,
        gpu_core_count=10,
        memory=MemoryInfo(total_bytes=32 * GIB, available_bytes=20 * GIB),
        is_apple_silicon=True,
        os_version="test",
    )


def schema(name: str) -> dict:
    path = Path("schemas/optimizer") / name
    return json.loads(path.read_text(encoding="utf-8"))


class OptimizerFoundationTests(unittest.TestCase):
    def test_dry_run_is_deterministic_bounded_and_schema_valid(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            source = Path(directory) / "source"
            source.mkdir()
            model = InspectedModel(
                model_id="test/model",
                path=source,
                config={"torch_dtype": "float16", "hidden_size": 1024},
                memory_spec=ModelMemorySpec("test/model", 2 * GIB, 131_072),
                kv_dtype_bytes=2,
            )
            calibration = CalibrationManifest(
                "local-v1", "a" * 64, 100, ("en", "ja", "zh-Hans"), ("code",)
            )
            plan = build_dry_run_plan(
                model,
                hardware(),
                Path(directory) / "artifact",
                OptimizationObjective.MEMORY,
                ResourceBudget(4 * GIB, 4 * GIB),
                QualityBudget({"code": 0.02}),
                calibration,
                "test-license",
                plan_id="plan-1",
                created_at="2026-08-22T00:00:00+00:00",
            )
            payload = plan.to_dict()
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["candidates"][0]["target_weight_bits"], 4)
            self.assertFalse(payload["candidates"][0]["executable"])
            self.assertIn(
                "backend_adapter_not_implemented",
                payload["candidates"][0]["blocking_reasons"],
            )
            validate_instance(payload, schema("optimization-plan-v1.schema.json"))
            validate_instance(
                calibration.to_dict(), schema("calibration-manifest-v1.schema.json")
            )
            self.assertFalse((Path(directory) / "artifact").exists())

    def test_resource_budget_blocks_candidates_without_allocating(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            source = Path(directory) / "source"
            source.mkdir()
            model = InspectedModel(
                "large", source, {}, ModelMemorySpec("large", 8 * GIB, 1), 2
            )
            plan = build_dry_run_plan(
                model,
                hardware(),
                Path(directory) / "artifact",
                OptimizationObjective.BALANCED,
                ResourceBudget(1 * GIB, 1 * GIB),
            )
            self.assertTrue(all(not item.within_budget for item in plan.candidates))
            self.assertTrue(
                all("memory_budget_exceeded" in item.blocking_reasons for item in plan.candidates)
            )

    def test_output_path_rejects_overlap_and_existing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            source = Path(directory) / "source"
            source.mkdir()
            with self.assertRaises(OptimizationPathError):
                validate_immutable_output_path(source, source / "artifact")
            existing = Path(directory) / "existing"
            existing.mkdir()
            with self.assertRaises(OptimizationPathError):
                validate_immutable_output_path(source, existing)

    def test_optimizer_events_are_bounded_and_schema_valid(self) -> None:
        bus = OptimizerEventBus(capacity=2)
        bus.publish("plan", "inspect", OptimizerState.PLANNING, 0.0, "optimizer.inspect")
        bus.publish("plan", "plan", OptimizerState.READY, 0.5, "optimizer.ready")
        latest = bus.publish("plan", "done", OptimizerState.COMPLETED, 1.0, "optimizer.done")
        self.assertEqual([event.event_id for event in bus.snapshot()], ["2", "3"])
        validate_instance(latest.to_dict(), schema("optimizer-event-v1.schema.json"))
        with self.assertRaises(ValueError):
            bus.publish("plan", "bad", OptimizerState.RUNNING, float("nan"), "bad")

    def test_artifact_manifest_records_provenance_and_matches_schema(self) -> None:
        manifest = ArtifactManifest(
            artifact_id="artifact-1",
            created_at="2026-08-22T00:00:00+00:00",
            plan_id="plan-1",
            source_hash="a" * 64,
            output_hash="b" * 64,
            output_bytes=1024,
            transforms=({"type": "int8"},),
            tool_versions={"vllm-apple": "0.1.0"},
            calibration_fingerprint=None,
            evaluation={"code": 0.98},
            license="test",
        )
        validate_instance(manifest.to_dict(), schema("artifact-manifest-v1.schema.json"))

    def test_bounded_io_profile_enables_duration_estimate(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            source = Path(directory) / "source"
            workspace = Path(directory) / "workspace"
            source.mkdir()
            workspace.mkdir()
            (source / "weights.safetensors").write_bytes(bytes(1024 * 1024))
            profile = profile_optimizer_io(source, workspace, hardware(), 1024 * 1024)
            validate_instance(profile.to_dict(), schema("performance-profile-v1.schema.json"))
            model = InspectedModel(
                "model", source, {}, ModelMemorySpec("model", GIB, 1), 2
            )
            plan = build_dry_run_plan(
                model,
                hardware(),
                Path(directory) / "artifact",
                OptimizationObjective.BALANCED,
                ResourceBudget(4 * GIB, 4 * GIB, 1),
                performance_profile=profile,
            )
            self.assertIsNotNone(plan.candidates[0].estimated_duration_seconds)
            self.assertFalse(any(workspace.iterdir()))

    def test_structured_optimizer_failure_matches_schema(self) -> None:
        failure = OptimizerFailure(
            OptimizerErrorCode.INSUFFICIENT_DISK,
            "optimizer.error.insufficient_disk",
            Recoverability.USER_ACTION_REQUIRED,
            "free space is below the plan budget",
        )
        validate_instance(failure.to_dict(), schema("optimizer-error-v1.schema.json"))

    def test_cli_emits_structured_error(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            exit_code = optimizer_main(
                ["plan", "/model/does-not-exist", "--output", "/tmp/artifact"]
            )
        payload = json.loads(stderr.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["error"]["code"], "invalid_model")
        validate_instance(payload["error"], schema("optimizer-error-v1.schema.json"))

    def test_performance_profile_rejects_untyped_input(self) -> None:
        with self.assertRaises(ValueError):
            OptimizationPerformanceProfile.from_dict(
                {
                    "schema_version": 1,
                    "profile_id": "profile",
                    "measured_at": "2026-08-22T00:00:00+00:00",
                    "hardware_fingerprint": "a" * 64,
                    "sample_bytes": True,
                    "read_bytes_per_second": 1,
                    "write_bytes_per_second": 1,
                }
            )


if __name__ == "__main__":
    unittest.main()
