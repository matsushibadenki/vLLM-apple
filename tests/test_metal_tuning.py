import json
import stat
import tempfile
import unittest
from pathlib import Path

from tests.schema_validator import validate_instance
from vllm_apple.cli import build_parser
from vllm_apple.daemon import build_parser as build_daemon_parser
from vllm_apple.daemon import install_startup_metal_tuning
from vllm_apple.execution import AppleChipProfile, ExecutionBackend
from vllm_apple.kernel_probe import KernelMeasurement, build_environment_fingerprint
from vllm_apple.kernel_profile import (
    ModelKernelShapeProfile,
    PagedAttentionShape,
    build_model_kernel_shape_profile,
)
from vllm_apple.metal_probe import NativeMetalProbeAdapter
from vllm_apple.metal_tuning import (
    discover_metal_tuning_report,
    load_metal_tuning_report,
    save_metal_tuning_report,
    tune_metal_shape_profile,
)
from vllm_apple.model import inspect_model
from vllm_apple.runtime_probe import RuntimeEnvironmentVersions
from vllm_apple.scheduler import ScheduleRequest
from vllm_apple.service import RuntimeService


class FakeTuningAdapter(NativeMetalProbeAdapter):
    def _candidate_model_shape(self, shape, configuration=None):
        baseline = self._baseline_model_shape(shape)
        latencies = {32: 104, 64: 103, 128: 100, 256: 99}
        return KernelMeasurement(
            baseline.output_digest,
            latencies[configuration.score_width],
            baseline.numeric_values,
        )


class MetalTuningReportTests(unittest.TestCase):
    def profile(self):
        shape = PagedAttentionShape(1024, 1, 32, 8, 128, 16, 64, 4194304)
        return ModelKernelShapeProfile(1, "a" * 24, "model", "test", 2, 2, (shape,))

    def test_median_tie_break_report_is_schema_valid_and_deterministic(self) -> None:
        report = tune_metal_shape_profile(
            self.profile(),
            FakeTuningAdapter(),
            hardware_fingerprint="hardware",
            environment_fingerprint="environment",
            samples=3,
            clock=lambda: 1000,
        )
        self.assertEqual(report.decisions[0].winner.score_width, 128)
        repeated = tune_metal_shape_profile(
            self.profile(),
            FakeTuningAdapter(),
            hardware_fingerprint="hardware",
            environment_fingerprint="environment",
            samples=3,
            clock=lambda: 2000,
        )
        self.assertEqual(report.tuning_id, repeated.tuning_id)
        schema = json.loads(Path("schemas/runtime/metal-tuning-report-v1.schema.json").read_text())
        validate_instance(report.to_dict(), schema)

    def test_report_is_saved_private_and_atomic(self) -> None:
        report = tune_metal_shape_profile(
            self.profile(),
            FakeTuningAdapter(),
            hardware_fingerprint="hardware",
            environment_fingerprint="environment",
            samples=1,
            clock=lambda: 1000,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private" / "tuning.json"
            save_metal_tuning_report(report, path)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(json.loads(path.read_text()), report.to_dict())
            loaded = load_metal_tuning_report(
                path,
                profile_id=report.profile_id,
                hardware_fingerprint="hardware",
                environment_fingerprint="environment",
            )
            self.assertEqual(loaded, report)
            with self.assertRaises(ValueError):
                load_metal_tuning_report(
                    path,
                    profile_id="b" * 24,
                    hardware_fingerprint="hardware",
                    environment_fingerprint="environment",
                )

    def test_cli_contract_defaults_to_median_and_private_save(self) -> None:
        arguments = build_parser().parse_args(["metal-shape-tune", "/model"])
        self.assertEqual(arguments.contexts, "128,1024")
        self.assertEqual(arguments.samples, 3)
        self.assertEqual(arguments.maximum_shapes, 4)
        self.assertFalse(arguments.stdout)
        self.assertIsNone(arguments.output)

    def test_discovery_selects_newest_valid_compatible_report(self) -> None:
        older = tune_metal_shape_profile(
            self.profile(),
            FakeTuningAdapter(),
            hardware_fingerprint="hardware",
            environment_fingerprint="environment",
            samples=1,
            clock=lambda: 1000,
        )
        newer = tune_metal_shape_profile(
            self.profile(),
            FakeTuningAdapter(),
            hardware_fingerprint="hardware",
            environment_fingerprint="environment",
            samples=3,
            clock=lambda: 2000,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "hardware" / "environment"
            save_metal_tuning_report(older, target / f"{older.profile_id}-{older.tuning_id}.json")
            save_metal_tuning_report(newer, target / f"{newer.profile_id}-{newer.tuning_id}.json")
            corrupt = target / f"{newer.profile_id}-{'f' * 24}.json"
            corrupt.write_text("{}")
            corrupt.chmod(0o600)
            discovered = discover_metal_tuning_report(
                profile_id=newer.profile_id,
                hardware_fingerprint="hardware",
                environment_fingerprint="environment",
                root=root,
            )
        self.assertEqual(discovered, newer)
        with self.assertRaises(ValueError):
            discover_metal_tuning_report(
                profile_id=newer.profile_id,
                hardware_fingerprint="../hardware",
                environment_fingerprint="environment",
                root=Path("/tmp"),
            )

    def test_daemon_automatically_installs_matching_report(self) -> None:
        versions = RuntimeEnvironmentVersions("metal", "mlx", "backend")
        chip = AppleChipProfile(
            1,
            "hardware",
            "test-soc",
            16 * 1024**3,
            (ExecutionBackend.NATIVE_METAL,),
            platform="Darwin",
            os_version="macOS",
        )
        environment = build_environment_fingerprint(
            platform=chip.platform,
            os_version=chip.os_version,
            toolchain_version=versions.toolchain_version,
            mlx_version=versions.mlx_version,
            backend_version=versions.backend_version,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = root / "model"
            model_root.mkdir()
            (model_root / "config.json").write_text(
                json.dumps(
                    {
                        "num_hidden_layers": 2,
                        "num_attention_heads": 4,
                        "num_key_value_heads": 2,
                        "hidden_size": 32,
                        "max_position_embeddings": 2048,
                    }
                )
            )
            (model_root / "model.safetensors").write_bytes(b"weights")
            model = inspect_model(str(model_root))
            profile = build_model_kernel_shape_profile(
                model, context_tiers=(128, 1024), block_tokens=16
            )
            report = tune_metal_shape_profile(
                profile,
                FakeTuningAdapter(),
                hardware_fingerprint=chip.hardware_fingerprint,
                environment_fingerprint=environment,
                samples=1,
                clock=lambda: 1000,
            )
            tuning_root = root / "tuning"
            destination = (
                tuning_root
                / chip.hardware_fingerprint
                / environment
                / f"{report.profile_id}-{report.tuning_id}.json"
            )
            save_metal_tuning_report(report, destination)
            runtime = RuntimeService()
            loaded = install_startup_metal_tuning(
                runtime, model, chip, versions, tuning_root=tuning_root
            )
        self.assertEqual(loaded, report)
        self.assertEqual(runtime.metal_tuning_snapshot()["active_tuning_id"], report.tuning_id)

    def test_serve_parsers_expose_explicit_and_disable_tuning_controls(self) -> None:
        cli_arguments = build_parser().parse_args(
            ["serve", "model", "--metal-tuning-report", "/tmp/tuning.json"]
        )
        daemon_arguments = build_daemon_parser().parse_args(["model", "--disable-metal-tuning"])
        self.assertEqual(cli_arguments.metal_tuning_report, Path("/tmp/tuning.json"))
        self.assertTrue(daemon_arguments.disable_metal_tuning)
        native_v2 = build_daemon_parser().parse_args(
            [
                "model",
                "--vllm-metal-source-root",
                "/private/source",
                "--vllm-metal-v2-helper",
                "/private/helper",
            ]
        )
        self.assertEqual(native_v2.vllm_metal_source_root, Path("/private/source"))
        self.assertEqual(native_v2.vllm_metal_v2_helper, Path("/private/helper"))

    def test_runtime_applies_winner_only_after_scheduler_safe_point(self) -> None:
        report = tune_metal_shape_profile(
            self.profile(),
            FakeTuningAdapter(),
            hardware_fingerprint="hardware",
            environment_fingerprint="environment",
            samples=1,
            clock=lambda: 1000,
        )
        runtime = RuntimeService()
        active = runtime.admit_schedule(ScheduleRequest("paged_attention", 1))
        self.assertFalse(runtime.install_metal_tuning(report))
        self.assertIsNone(active.kernel_tuning_id)
        self.assertEqual(runtime.metal_tuning_snapshot()["pending_tuning_id"], report.tuning_id)

        runtime.complete_schedule(active)
        self.assertEqual(runtime.metal_tuning_snapshot()["active_tuning_id"], report.tuning_id)
        self.assertEqual(
            runtime.metal_thread_configuration(report.decisions[0].shape),
            report.decisions[0].winner,
        )
        next_reservation = runtime.admit_schedule(ScheduleRequest("paged_attention", 1))
        self.assertEqual(next_reservation.kernel_tuning_id, report.tuning_id)
        context = runtime.inference_kernel_context(next_reservation)
        self.assertIsNotNone(context)
        assert context is not None
        self.assertEqual(context.tuning_id, report.tuning_id)
        self.assertEqual(context.paged_attention[0].configuration, report.decisions[0].winner)
        runtime.complete_schedule(next_reservation)

    def test_old_reservation_never_observes_new_tuning_context(self) -> None:
        report = tune_metal_shape_profile(
            self.profile(),
            FakeTuningAdapter(),
            hardware_fingerprint="hardware",
            environment_fingerprint="environment",
            samples=1,
            clock=lambda: 1000,
        )
        runtime = RuntimeService()
        old_request = runtime.admit_schedule(ScheduleRequest("paged_attention", 0))
        self.assertFalse(runtime.install_metal_tuning(report))
        self.assertIsNone(runtime.inference_kernel_context(old_request))
        runtime.complete_schedule(old_request)

        new_request = runtime.admit_schedule(ScheduleRequest("paged_attention", 0))
        self.assertEqual(runtime.inference_kernel_context(new_request).tuning_id, report.tuning_id)
        runtime.complete_schedule(new_request)


if __name__ == "__main__":
    unittest.main()
