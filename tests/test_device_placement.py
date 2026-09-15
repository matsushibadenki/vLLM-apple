import tempfile
import unittest
from pathlib import Path

from vllm_apple.device_benchmark import (
    DeviceBenchmarkConfig,
    DeviceBenchmarkMeasurement,
    DeviceBenchmarkReport,
)
from vllm_apple.device_placement import (
    build_device_placement_plan,
    load_device_placement_plan,
    load_device_placement_with_fallback,
    promote_device_placement_plan,
    save_device_placement_plan,
)
from vllm_apple.device_selection import select_measured_device_backend
from vllm_apple.execution import ExecutionBackend, WorkloadPhase
from vllm_apple.kernel_probe import (
    KernelCapabilityRegistry,
    KernelMeasurement,
    KernelProbeConfig,
    run_kernel_probe,
)
from vllm_apple.operator_dispatch import OperatorDispatcher
from vllm_apple.scheduler import BasicScheduler, ScheduleRequest
from vllm_apple.service import RuntimeService
from vllm_apple.types import Backend, HardwareInfo, MemoryInfo


def benchmark_report(backend, latency, *, operator="matmul", phase=WorkloadPhase.PREFILL):
    config = DeviceBenchmarkConfig(
        operator, backend, phase, "fp32", (8, 8, 8), 1, 3
    )
    measurements = tuple(
        DeviceBenchmarkMeasurement(
            latency, latency, 0, 0, 512, 100, None, "a" * 64
        )
        for _ in range(3)
    )
    return DeviceBenchmarkReport(
        "m4-test",
        "environment-test",
        ("1" if backend is ExecutionBackend.CPU else "2") * 24,
        config,
        measurements,
        None,
    )


def scheduler_with_dispatcher():
    registry = KernelCapabilityRegistry("m4-test", "environment-test")

    def measurement():
        return KernelMeasurement("a" * 64, 10)

    registry.record(run_kernel_probe(
        KernelProbeConfig(
            "m4-test", "environment-test", ExecutionBackend.NATIVE_MLX,
            "matmul", samples=1,
        ),
        measurement,
        measurement,
    ))
    hardware = HardwareInfo(
        "Darwin", "arm64", "M4", 8, 8, 10,
        MemoryInfo(1_000, 800), True, "test",
    )
    return BasicScheduler(hardware, 500, OperatorDispatcher(registry))


class DevicePlacementPlanTests(unittest.TestCase):
    def setUp(self):
        self.reports = (
            benchmark_report(ExecutionBackend.CPU, 100),
            benchmark_report(ExecutionBackend.NATIVE_MLX, 50),
        )
        self.plan = build_device_placement_plan(
            self.reports, select_measured_device_backend(self.reports)
        )

    def test_binds_measured_winner_to_versioned_plan(self):
        placement = self.plan.placements[0]
        self.assertEqual(placement.backend, ExecutionBackend.NATIVE_MLX)
        self.assertEqual(placement.benchmark_report_id, self.reports[1].report_id)
        self.assertEqual(len(self.plan.plan_id), 24)

    def test_scheduler_applies_exact_shape_only(self):
        scheduler = scheduler_with_dispatcher()
        self.assertEqual(scheduler.request_device_placement_plan(self.plan).status, "applied")
        exact = ScheduleRequest(
            "matmul", 10, batch_size=1, phase=WorkloadPhase.PREFILL,
            precision="fp32", dimensions=(8, 8, 8),
        )
        self.assertEqual(scheduler.choose_backend(exact), Backend.MLX_GPU)
        self.assertEqual(
            scheduler.choose_backend(ScheduleRequest(
                "matmul", 10, batch_size=1, phase=WorkloadPhase.PREFILL,
                precision="fp32", dimensions=(16, 16, 16),
            )),
            Backend.CPU,
        )
        reservation = scheduler.admit(exact)
        self.assertEqual(reservation.device_placement_plan_id, self.plan.plan_id)
        scheduler.complete(reservation)

    def test_promoted_ane_route_falls_back_to_gpu_then_cpu(self):
        registry = KernelCapabilityRegistry("m4-test", "environment-test")
        for backend in (ExecutionBackend.COREML_DRAFT, ExecutionBackend.NATIVE_MLX):
            registry.record(run_kernel_probe(
                KernelProbeConfig(
                    "m4-test", "environment-test", backend, "attention", samples=1,
                ),
                lambda: KernelMeasurement("a" * 64, 10),
                lambda: KernelMeasurement("a" * 64, 10),
            ))
        hardware = HardwareInfo(
            "Darwin", "arm64", "M4", 8, 8, 10,
            MemoryInfo(1_000, 800), True, "test",
        )
        scheduler = BasicScheduler(hardware, 500, OperatorDispatcher(registry))
        reports = (
            benchmark_report(
                ExecutionBackend.CPU, 100, operator="attention",
                phase=WorkloadPhase.AUXILIARY,
            ),
            benchmark_report(
                ExecutionBackend.NATIVE_MLX, 70, operator="attention",
                phase=WorkloadPhase.AUXILIARY,
            ),
            benchmark_report(
                ExecutionBackend.COREML_DRAFT, 40, operator="attention",
                phase=WorkloadPhase.AUXILIARY,
            ),
        )
        plan = build_device_placement_plan(
            reports, select_measured_device_backend(reports)
        )
        self.assertEqual(scheduler.request_device_placement_plan(plan).status, "applied")
        request = ScheduleRequest(
            "attention", 10, batch_size=1, phase=WorkloadPhase.AUXILIARY,
            precision="fp32", dimensions=(8, 8, 8),
        )
        invoked = []

        def operation(backend):
            invoked.append(backend)
            if backend is ExecutionBackend.COREML_DRAFT:
                raise TimeoutError("coreml prediction deadline")
            return "reference" if backend is ExecutionBackend.CPU else "mismatch"

        reservation = scheduler.admit(request)
        result = scheduler.execute_with_fallback(
            request, operation, lambda value, _backend: value == "reference",
            reservation=reservation,
        )
        scheduler.complete(reservation)
        self.assertEqual(result.backend, ExecutionBackend.CPU)
        self.assertEqual(
            invoked,
            [
                ExecutionBackend.COREML_DRAFT,
                ExecutionBackend.NATIVE_MLX,
                ExecutionBackend.CPU,
            ],
        )
        self.assertEqual(
            [attempt.error_code for attempt in result.attempts],
            ["backend_timeout", "output_mismatch", None],
        )

    def test_plan_change_is_deferred_until_safe_point(self):
        scheduler = scheduler_with_dispatcher()
        active = scheduler.admit(ScheduleRequest("matmul", 10))
        self.assertEqual(scheduler.request_device_placement_plan(self.plan).status, "deferred")
        self.assertEqual(
            scheduler.device_placement_snapshot()["pending_plan_id"], self.plan.plan_id
        )
        self.assertEqual(
            scheduler.apply_pending_device_placement_plan().status, "deferred"
        )
        scheduler.complete(active)
        self.assertEqual(
            scheduler.apply_pending_device_placement_plan().status, "applied"
        )

    def test_plan_requires_probe_gated_dispatcher(self):
        scheduler = scheduler_with_dispatcher()
        scheduler._operator_dispatcher = None
        with self.assertRaisesRegex(RuntimeError, "probe-gated"):
            scheduler.request_device_placement_plan(self.plan)

    def test_plan_rejects_another_dispatcher_profile(self):
        scheduler = scheduler_with_dispatcher()
        foreign_reports = tuple(
            DeviceBenchmarkReport(
                "other-mac", report.environment_fingerprint, report.capability_id,
                report.config, report.measurements, report.cold_load_nanoseconds,
            )
            for report in self.reports
        )
        foreign_plan = build_device_placement_plan(
            foreign_reports, select_measured_device_backend(foreign_reports)
        )
        with self.assertRaisesRegex(ValueError, "dispatcher profile"):
            scheduler.request_device_placement_plan(foreign_plan)

    def test_runtime_service_installs_plan_at_safe_point(self):
        service = RuntimeService()
        dispatcher = scheduler_with_dispatcher()._operator_dispatcher
        self.assertTrue(service.install_operator_dispatcher(dispatcher))
        self.assertTrue(service.install_device_placement_plan(self.plan))
        self.assertEqual(
            service.scheduler.device_placement_snapshot()["active_plan_id"],
            self.plan.plan_id,
        )

    def test_runtime_service_control_has_three_language_diagnostics(self):
        service = RuntimeService()
        actions = []
        service.configure_device_placement_control(
            lambda action: actions.append(action) is None
        )
        accepted, snapshot = service.control_device_placement("reload")
        self.assertTrue(accepted)
        self.assertEqual(actions, ["reload"])
        self.assertEqual(set(snapshot["messages"]), {"en", "ja", "zh-Hans"})

    def test_private_plan_round_trip_and_expiry(self):
        plan = build_device_placement_plan(
            self.reports,
            select_measured_device_backend(self.reports),
            ttl_seconds=60,
            clock=lambda: 1_000,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            path = save_device_placement_plan(plan, root / "placement.json")
            loaded = load_device_placement_plan(
                path,
                hardware_fingerprint="m4-test",
                environment_fingerprint="environment-test",
                clock=lambda: 1_001,
            )
            self.assertEqual(loaded.plan_id, plan.plan_id)
            with self.assertRaisesRegex(ValueError, "expired"):
                load_device_placement_plan(
                    path,
                    hardware_fingerprint="m4-test",
                    environment_fingerprint="environment-test",
                    clock=lambda: 1_060,
                )

    def test_corrupt_current_rolls_back_to_last_known_good(self):
        plan = build_device_placement_plan(
            self.reports,
            select_measured_device_backend(self.reports),
            ttl_seconds=60,
            clock=lambda: 1_000,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            current = root / "current.json"
            current.write_text("{}")
            current.chmod(0o600)
            last_good = save_device_placement_plan(plan, root / "last-good.json")
            loaded, source = load_device_placement_with_fallback(
                current,
                last_good,
                hardware_fingerprint="m4-test",
                environment_fingerprint="environment-test",
                clock=lambda: 1_001,
            )
            self.assertEqual(source, "last_known_good")
            self.assertEqual(loaded.plan_id, plan.plan_id)

    def test_scheduler_rejects_expired_plan(self):
        expired = build_device_placement_plan(
            self.reports,
            select_measured_device_backend(self.reports),
            ttl_seconds=1,
            clock=lambda: 1,
        )
        with self.assertRaisesRegex(ValueError, "expired"):
            scheduler_with_dispatcher().request_device_placement_plan(expired)

    def test_promotion_archives_only_valid_previous_plan(self):
        previous = build_device_placement_plan(
            self.reports, select_measured_device_backend(self.reports),
            ttl_seconds=60, clock=lambda: 1_000,
        )
        replacement = build_device_placement_plan(
            self.reports, select_measured_device_backend(self.reports),
            ttl_seconds=60, clock=lambda: 1_010,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            current = save_device_placement_plan(previous, root / "current.json")
            last_good = root / "last-good.json"
            promote_device_placement_plan(
                replacement, current, last_good, clock=lambda: 1_011
            )
            restored = load_device_placement_plan(
                last_good,
                hardware_fingerprint="m4-test",
                environment_fingerprint="environment-test",
                clock=lambda: 1_011,
            )
            active = load_device_placement_plan(
                current,
                hardware_fingerprint="m4-test",
                environment_fingerprint="environment-test",
                clock=lambda: 1_011,
            )
            self.assertEqual(restored.plan_id, previous.plan_id)
            self.assertEqual(active.plan_id, replacement.plan_id)


if __name__ == "__main__":
    unittest.main()
