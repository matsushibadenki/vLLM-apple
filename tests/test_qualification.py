import os
import json
import tempfile
from dataclasses import replace
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_apple.backend_memory import BackendMemorySample
from vllm_apple.model import ModelCapabilityError
from vllm_apple.types import HardwareInfo, MemoryInfo, MemoryPressure
from vllm_apple.qualification import (
    QualificationConfig,
    default_qualification_report_path,
    evaluate_qualification_context,
    qualify_model,
    save_qualification_report,
)
from tests.schema_validator import validate_instance


class FakeBackend:
    def __init__(self, config: object) -> None:
        self.config = config
        self.pid = os.getpid()
        self.base_url = "http://127.0.0.1:8001"
        self.running = False
        self.stopped = False

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False
        self.stopped = True


class QualificationTests(unittest.TestCase):
    @staticmethod
    def hardware_fixture(*, total_bytes: int = 32 * 1024**3, available_bytes: int = 24 * 1024**3) -> HardwareInfo:
        return HardwareInfo(
            platform="Darwin",
            architecture="arm64",
            soc="Apple M4",
            physical_cpu_count=10,
            logical_cpu_count=10,
            gpu_core_count=16,
            memory=MemoryInfo(total_bytes, available_bytes, MemoryPressure.NORMAL),
            is_apple_silicon=True,
            os_version="15.0",
        )

    def test_mlx_capability_failure_prevents_process_construction(self) -> None:
        factory_called = False

        def factory(config):
            nonlocal factory_called
            factory_called = True
            return FakeBackend(config)

        with patch("vllm_apple.qualification.inspect_model", return_value=object()), patch(
            "vllm_apple.qualification.ensure_model_backend_compatible",
            side_effect=ModelCapabilityError(
                "backend_missing_model_capabilities:gated_deltanet"
            ),
        ):
            with self.assertRaisesRegex(
                ModelCapabilityError, "gated_deltanet"
            ):
                qualify_model(
                    QualificationConfig(
                        model="cached-qwen",
                        executable=Path("/tmp/mlx_lm.server"),
                        backend_kind="mlx_lm",
                        duration_seconds=1,
                        require_30_minute_window=False,
                    ),
                    process_factory=factory,
                )
        self.assertFalse(factory_called)

    def test_quality_smoke_failure_stops_before_soak(self) -> None:
        created = []

        def factory(config):
            backend = FakeBackend(config)
            created.append(backend)
            return backend

        with tempfile.TemporaryDirectory() as directory:
            model = self.model_fixture(Path(directory))
            with patch(
                "vllm_apple.qualification.run_serving_promotion_probe",
                return_value={"passed": True},
            ), patch(
                "vllm_apple.qualification.build_v2_hardware_fingerprint",
                return_value="hardware-v1",
            ), patch(
                "vllm_apple.qualification.detect_hardware",
                return_value=self.hardware_fixture(),
            ), patch(
                "vllm_apple.qualification.run_serving_quality_smoke",
                return_value={"passed": False},
            ), patch("vllm_apple.qualification.run_soak") as soak:
                with self.assertRaisesRegex(RuntimeError, "quality smoke failed"):
                    qualify_model(
                        QualificationConfig(
                            model=str(model),
                            executable=Path("/tmp/vllm"),
                            duration_seconds=1,
                            require_30_minute_window=False,
                            quality_smoke=True,
                        ),
                        process_factory=factory,
                    )
        soak.assert_not_called()
        self.assertTrue(created[0].stopped)

    def test_mlx_context_uses_allocator_endpoint_and_reserved_available_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = self.model_fixture(Path(directory))
            config = QualificationConfig(
                model=str(model),
                executable=Path("/tmp/mlx-server"),
                backend_kind="mlx_lm",
                max_model_len=4096,
                duration_seconds=1,
                require_30_minute_window=False,
            )
            sample = BackendMemorySample(
                None,
                None,
                allocator_current_bytes=1024,
                allocator_peak_bytes=2048,
                kv_used_bytes=128,
                source="mlx-wrapper-v1",
            )
            memory = MemoryInfo(
                total_bytes=8 * 1024**3,
                available_bytes=3 * 1024**3,
                pressure=MemoryPressure.NORMAL,
            )
            with patch(
                "vllm_apple.qualification.MLXMemoryMetricsAdapter.sample",
                return_value=sample,
            ), patch("vllm_apple.qualification.detect_memory", return_value=memory):
                report = evaluate_qualification_context(config, "http://127.0.0.1:1")
            self.assertTrue(report["enabled"])
            self.assertEqual(report["status"], "sufficient")
            self.assertEqual(report["kv_capacity_bytes"], 1024**3 + 128)
            self.assertEqual(report["source"], "mlx-allocator-os-available-v1")

    def test_memory_ceiling_prevents_either_backend_process_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = self.model_fixture(Path(directory))
            for backend_kind in ("vllm_metal", "mlx_lm"):
                factory_called = False

                def factory(config):
                    nonlocal factory_called
                    factory_called = True
                    return FakeBackend(config)

                with self.subTest(backend_kind=backend_kind), patch(
                    "vllm_apple.qualification.detect_hardware",
                    return_value=self.hardware_fixture(total_bytes=1024, available_bytes=0),
                ):
                    with self.assertRaisesRegex(
                        ModelCapabilityError, "model_memory_hard_ceiling_exceeded"
                    ):
                        qualify_model(
                            QualificationConfig(
                                model=str(model),
                                executable=Path("/tmp/backend"),
                                backend_kind=backend_kind,
                                duration_seconds=1,
                                require_30_minute_window=False,
                            ),
                            process_factory=factory,
                        )
                    self.assertFalse(factory_called)

    def test_mlx_backend_uses_native_process_contract_and_report_identity(self) -> None:
        captured = []

        def factory(config):
            captured.append(config)
            return FakeBackend(config)

        with tempfile.TemporaryDirectory() as directory:
            model = self.model_fixture(Path(directory))
            with patch(
                "vllm_apple.qualification.run_serving_promotion_probe",
                return_value={"passed": True},
            ), patch(
                "vllm_apple.qualification.run_soak", return_value={"passed": True}
            ), patch(
                "vllm_apple.qualification.detect_hardware",
                return_value=self.hardware_fixture(),
            ), patch(
                "vllm_apple.qualification.build_v2_hardware_fingerprint",
                return_value="hardware-v1",
            ), patch(
                "vllm_apple.qualification.run_phase_probe",
                return_value={"schema_version": 1, "sample_count": 2},
            ) as phase:
                report = qualify_model(
                    QualificationConfig(
                        model=str(model),
                        executable=Path("/tmp/mlx_lm.server"),
                        backend_kind="mlx_lm",
                        duration_seconds=1,
                        warmup_seconds=0,
                        require_30_minute_window=False,
                        phase_samples=2,
                        phase_output_tokens=24,
                    ),
                    process_factory=factory,
                )
        self.assertEqual(captured[0].backend_kind, "mlx_lm")
        self.assertEqual(report["backend"], "mlx_lm")
        self.assertEqual(report["phase_profile"]["sample_count"], 2)
        phase_config = phase.call_args.args[0]
        self.assertEqual(phase_config.backend, "mlx_lm")
        self.assertEqual(phase_config.samples, 2)
        self.assertEqual(phase_config.maximum_output_tokens, 24)
        self.assertEqual(phase_config.target_pid, os.getpid())
        self.assertTrue(report["passed"])

    def test_default_report_path_is_bounded_and_model_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = default_qualification_report_path(
                "org/model with spaces/日本語", application_support=root
            )
            self.assertEqual(path.parent, root / "qualification-reports")
            self.assertEqual(path.suffix, ".json")
            self.assertNotIn("model with spaces", path.name)
            self.assertLess(len(path.name), 80)

    def model_fixture(self, root: Path) -> Path:
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "llama",
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "hidden_size": 8,
                    "max_position_embeddings": 4096,
                }
            )
        )
        (model / "model.safetensors").write_bytes(b"weights")
        return model

    def test_launch_soak_and_shutdown_are_one_fail_closed_report(self) -> None:
        created: list[FakeBackend] = []

        def factory(config: object) -> FakeBackend:
            backend = FakeBackend(config)
            created.append(backend)
            return backend

        soak = {"passed": True, "requests": 20}
        promotion = {"passed": True}
        with tempfile.TemporaryDirectory() as directory:
            model = self.model_fixture(Path(directory))
            with patch(
                "vllm_apple.qualification.run_serving_promotion_probe",
                return_value=promotion,
            ), patch("vllm_apple.qualification.run_soak", return_value=soak) as runner:
                report = qualify_model(
                    QualificationConfig(
                        model=str(model),
                        executable=Path("/tmp/vllm"),
                        duration_seconds=1,
                        warmup_seconds=0,
                        require_30_minute_window=False,
                    ),
                    process_factory=factory,  # type: ignore[arg-type]
                )
        self.assertTrue(report["passed"])
        self.assertTrue(report["shutdown_clean"])
        self.assertEqual(report["promotion_probe"], promotion)
        self.assertTrue(created[0].stopped)
        soak_config = runner.call_args.args[0]
        self.assertEqual(soak_config.mode, "chat-mixed")
        self.assertEqual(soak_config.target_pid, os.getpid())

    def test_reduced_context_fails_unless_explicitly_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = self.model_fixture(Path(directory))
            config = QualificationConfig(
                model=str(model),
                executable=Path("/tmp/vllm"),
                max_model_len=4096,
                duration_seconds=1,
                require_30_minute_window=False,
                vllm_version="0.28.0",
            )
            sample = BackendMemorySample(None, 0.5, kv_used_bytes=64, kv_capacity_bytes=128)
            with patch(
                "vllm_apple.qualification.VLLMMemoryMetricsAdapter.sample",
                return_value=sample,
            ):
                report = evaluate_qualification_context(config, "http://127.0.0.1:1")
                allowed = evaluate_qualification_context(
                    replace(config, allow_context_reduction=True),
                    "http://127.0.0.1:1",
                )
            self.assertEqual(report["status"], "reduced")
            self.assertFalse(report["passed"])
            self.assertTrue(allowed["passed"])

    def test_report_is_private_atomic_and_matches_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qualification.json"
            report = {
                "schema_version": 1,
                "model": "model",
                "backend": "vllm_metal",
                "load_seconds": 1.0,
                "shutdown_clean": True,
                "promotion_probe": {"passed": True},
                "soak": {"passed": True},
                "context_reevaluation": {
                    "enabled": False,
                    "status": "unavailable",
                    "configured_context_tokens": None,
                    "effective_context_tokens": None,
                    "capacity_context_tokens": None,
                    "kv_capacity_bytes": None,
                    "kv_bytes_per_token": None,
                    "weights_bytes": None,
                    "source": None,
                    "reevaluations": 0,
                    "passed": True,
                },
                "passed": True,
            }
            save_qualification_report(report, path)
            loaded = json.loads(path.read_text())
            schema = json.loads(
                Path("schemas/runtime/qualification-report-v1.schema.json").read_text()
            )
            validate_instance(loaded, schema)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_reduced_context_stops_before_expensive_probes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = self.model_fixture(Path(directory))
            config = QualificationConfig(
                model=str(model),
                executable=Path("/tmp/vllm"),
                max_model_len=4096,
                duration_seconds=1,
                require_30_minute_window=False,
                vllm_version="0.28.0",
            )
            backend = FakeBackend(config)
            sample = BackendMemorySample(None, 0.5, kv_used_bytes=64, kv_capacity_bytes=128)
            with patch(
                "vllm_apple.qualification.VLLMMemoryMetricsAdapter.sample",
                return_value=sample,
            ), patch("vllm_apple.qualification.run_serving_promotion_probe") as promotion, patch(
                "vllm_apple.qualification.run_soak"
            ) as soak:
                report = qualify_model(config, process_factory=lambda _: backend)  # type: ignore[arg-type]
            self.assertFalse(report["passed"])
            self.assertEqual(report["context_reevaluation"]["status"], "reduced")
            promotion.assert_not_called()
            soak.assert_not_called()
            self.assertTrue(backend.stopped)

    def test_failed_soak_still_stops_backend(self) -> None:
        created: list[FakeBackend] = []

        def factory(config: object) -> FakeBackend:
            backend = FakeBackend(config)
            created.append(backend)
            return backend

        with tempfile.TemporaryDirectory() as directory:
            model = self.model_fixture(Path(directory))
            with patch(
                "vllm_apple.qualification.run_serving_promotion_probe",
                return_value={"passed": True},
            ), patch(
                "vllm_apple.qualification.run_soak", side_effect=RuntimeError("failed")
            ), patch(
                "vllm_apple.qualification.detect_hardware",
                return_value=self.hardware_fixture(),
            ):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    qualify_model(
                        QualificationConfig(
                            model=str(model),
                            executable=Path("/tmp/vllm"),
                            duration_seconds=1,
                            warmup_seconds=0,
                            require_30_minute_window=False,
                        ),
                        process_factory=factory,  # type: ignore[arg-type]
                    )
        self.assertTrue(created[0].stopped)

    def test_failed_promotion_stops_before_soak(self) -> None:
        created: list[FakeBackend] = []

        def factory(config: object) -> FakeBackend:
            backend = FakeBackend(config)
            created.append(backend)
            return backend

        with tempfile.TemporaryDirectory() as directory:
            model = self.model_fixture(Path(directory))
            with patch(
                "vllm_apple.qualification.run_serving_promotion_probe",
                return_value={"passed": False},
            ), patch("vllm_apple.qualification.run_soak") as soak, patch(
                "vllm_apple.qualification.detect_hardware",
                return_value=self.hardware_fixture(),
            ):
                with self.assertRaisesRegex(RuntimeError, "promotion probe failed"):
                    qualify_model(
                        QualificationConfig(
                            model=str(model),
                            executable=Path("/tmp/vllm"),
                            duration_seconds=1,
                            warmup_seconds=0,
                            require_30_minute_window=False,
                        ),
                        process_factory=factory,  # type: ignore[arg-type]
                    )
        soak.assert_not_called()
        self.assertTrue(created[0].stopped)


if __name__ == "__main__":
    unittest.main()
