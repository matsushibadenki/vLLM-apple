from __future__ import annotations

import threading
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vllm_apple.daemon import start_observed_native_v2_tuning
from vllm_apple.scheduler import BasicScheduler, MaintenanceInProgressError, ScheduleRequest
from vllm_apple.service import RuntimeService
from vllm_apple.vllm_metal_v2_orchestration import (
    NativeV2IdleTuningCoordinator,
    NativeV2ObservationMonitor,
)
from vllm_apple.vllm_metal_v2_tuning import (
    V2CandidateResult,
    V2DispatchConfiguration,
    V2PagedAttentionFamily,
    V2PagedAttentionShape,
    V2ShapeTuningDecision,
    build_v2_tuning_profile,
    save_v2_tuning_profile,
)

from tests.test_scheduler import hardware


def profile():
    shape = V2PagedAttentionShape(1024, 1024, 1, 8, 4, 256, 16, 10)
    configuration = V2DispatchConfiguration(V2PagedAttentionFamily.PER_TOKEN, 256)
    result = V2CandidateResult(configuration, True, 100, (100,), "a" * 64)
    return build_v2_tuning_profile(
        (V2ShapeTuningDecision(shape, configuration, (result,)),),
        hardware_fingerprint="hardware",
        source_fingerprint="source",
    )


class NativeV2IdleTuningCoordinatorTests(unittest.TestCase):
    def test_applies_profile_while_admission_is_exclusively_blocked(self) -> None:
        scheduler = BasicScheduler(hardware(), 500)
        entered = threading.Event()
        release = threading.Event()
        applied = []
        events = []

        def tune():
            entered.set()
            release.wait(1)
            return profile()

        coordinator = NativeV2IdleTuningCoordinator(
            scheduler, publish=lambda name, payload: events.append((name, payload))
        )
        self.assertTrue(coordinator.start(tune, applied.append))
        self.assertTrue(entered.wait(1))
        with self.assertRaises(MaintenanceInProgressError):
            scheduler.admit(ScheduleRequest("paged_attention", 1))
        self.assertFalse(coordinator.start(tune, applied.append))
        release.set()
        self.assertTrue(coordinator.wait(1))
        self.assertEqual(applied, [profile()])
        self.assertEqual(coordinator.snapshot().status, "applied")
        self.assertEqual(events[-1][1]["status"], "applied")

    def test_failure_releases_maintenance_lease(self) -> None:
        scheduler = BasicScheduler(hardware(), 500)
        coordinator = NativeV2IdleTuningCoordinator(scheduler)

        def fail():
            raise ValueError("sensitive detail must not enter status")

        self.assertTrue(coordinator.start(fail, lambda _profile: None))
        self.assertTrue(coordinator.wait(1))
        snapshot = coordinator.snapshot()
        self.assertEqual(snapshot.status, "failed")
        self.assertEqual(snapshot.error_code, "ValueError")
        reservation = scheduler.admit(ScheduleRequest("paged_attention", 1))
        scheduler.complete(reservation)

    def test_daemon_saves_profile_and_recycles_backend_under_idle_lease(self) -> None:
        service = RuntimeService()
        backend = Mock()
        generated = profile()
        with tempfile.TemporaryDirectory() as directory:
            observation = Path(directory) / "shapes.json"
            observation.write_text("{}")
            with (
                patch(
                    "vllm_apple.daemon.inspect_vllm_metal_integration",
                    return_value=SimpleNamespace(
                        native_v2_detected=True, source_fingerprint="source"
                    ),
                ),
                patch(
                    "vllm_apple.daemon.build_v2_hardware_fingerprint",
                    return_value="hardware",
                ),
                patch(
                    "vllm_apple.daemon.default_v2_observation_path",
                    return_value=observation,
                ),
                patch("vllm_apple.daemon.load_v2_observations", return_value=(generated.decisions[0].shape,)),
                patch(
                    "vllm_apple.daemon.VLLMMetalV2MeasurementAdapter"
                ) as adapter_type,
                patch(
                    "vllm_apple.daemon.tune_v2_observed_shapes",
                    return_value=generated,
                ),
                patch("vllm_apple.daemon.save_v2_tuning_profile") as save,
            ):
                adapter_type.return_value.capability.return_value = {"compatible": True}
                self.assertTrue(
                    start_observed_native_v2_tuning(
                        service,
                        backend,
                        source_root=Path(directory),
                        helper=Path("/private/helper"),
                        samples=1,
                    )
                )
                self.assertTrue(service.native_v2_tuning.wait(1))
        save.assert_called_once_with(generated)
        backend.restart.assert_called_once_with()
        self.assertEqual(service.native_v2_tuning.snapshot().status, "applied")

    def test_daemon_quarantines_failed_profile_and_rolls_backend_back(self) -> None:
        service = RuntimeService()
        backend = Mock()
        backend.restart.side_effect = [RuntimeError("new profile failed"), None]
        generated = profile()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = root / "shapes.json"
            observation.write_text("{}")
            destination = root / "profiles" / f"{generated.profile_id}.json"
            with (
                patch(
                    "vllm_apple.daemon.inspect_vllm_metal_integration",
                    return_value=SimpleNamespace(
                        native_v2_detected=True, source_fingerprint="source"
                    ),
                ),
                patch(
                    "vllm_apple.daemon.build_v2_hardware_fingerprint",
                    return_value="hardware",
                ),
                patch(
                    "vllm_apple.daemon.default_v2_observation_path",
                    return_value=observation,
                ),
                patch(
                    "vllm_apple.daemon.load_v2_observations",
                    return_value=(generated.decisions[0].shape,),
                ),
                patch("vllm_apple.daemon.VLLMMetalV2MeasurementAdapter") as adapter_type,
                patch(
                    "vllm_apple.daemon.tune_v2_observed_shapes",
                    return_value=generated,
                ),
                patch(
                    "vllm_apple.daemon.save_v2_tuning_profile",
                    side_effect=lambda value: save_v2_tuning_profile(value, destination),
                ),
            ):
                adapter_type.return_value.capability.return_value = {"compatible": True}
                self.assertTrue(
                    start_observed_native_v2_tuning(
                        service,
                        backend,
                        source_root=root,
                        helper=Path("/private/helper"),
                        samples=1,
                    )
                )
                self.assertTrue(service.native_v2_tuning.wait(1))
            self.assertFalse(destination.exists())
            self.assertTrue((destination.parent / "quarantine" / destination.name).exists())
        self.assertEqual(backend.restart.call_count, 2)
        self.assertEqual(service.state.value, "ready")
        self.assertEqual(service.native_v2_tuning.snapshot().status, "failed")

    def test_observation_monitor_debounces_and_deduplicates_content(self) -> None:
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            observation = Path(directory) / "shapes.json"
            observation.write_text("first")
            observation.chmod(0o600)
            monitor = NativeV2ObservationMonitor(
                observation,
                lambda: calls.append("started") is None,
                interval_seconds=1,
                debounce_seconds=10,
            )
            self.assertFalse(monitor.poll_once(now=0))
            self.assertFalse(monitor.poll_once(now=9))
            self.assertTrue(monitor.poll_once(now=10))
            self.assertFalse(monitor.poll_once(now=30))
            observation.write_text("first")
            self.assertFalse(monitor.poll_once(now=40))
        self.assertEqual(calls, ["started"])

    def test_observation_monitor_retries_when_scheduler_is_busy(self) -> None:
        outcomes = iter((False, True))
        with tempfile.TemporaryDirectory() as directory:
            observation = Path(directory) / "shapes.json"
            observation.write_text("shape")
            observation.chmod(0o600)
            monitor = NativeV2ObservationMonitor(
                observation,
                lambda: next(outcomes),
                debounce_seconds=0,
            )
            self.assertFalse(monitor.poll_once(now=0))
            self.assertFalse(monitor.poll_once(now=0))
            self.assertTrue(monitor.poll_once(now=1))

    def test_observation_monitor_can_prime_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            observation = Path(directory) / "shapes.json"
            observation.write_text("already-tuned")
            observation.chmod(0o600)
            monitor = NativeV2ObservationMonitor(
                observation,
                lambda: self.fail("primed artifact must not trigger"),
                prime_existing=True,
            )
            self.assertFalse(monitor.poll_once(now=100))

    def test_busy_scheduler_reports_waiting_without_starting_thread(self) -> None:
        scheduler = BasicScheduler(hardware(), 500)
        reservation = scheduler.admit(ScheduleRequest("paged_attention", 1))
        coordinator = NativeV2IdleTuningCoordinator(scheduler)
        self.assertFalse(coordinator.start(profile, lambda _profile: None))
        self.assertEqual(coordinator.snapshot().status, "waiting_for_idle")
        scheduler.complete(reservation)

    def test_disable_enable_and_retry_control_preserve_last_job(self) -> None:
        scheduler = BasicScheduler(hardware(), 500)
        coordinator = NativeV2IdleTuningCoordinator(scheduler)
        applied = []
        coordinator.set_enabled(False)
        self.assertFalse(coordinator.start(profile, applied.append))
        self.assertEqual(coordinator.snapshot().status, "disabled")
        coordinator.set_enabled(True)
        self.assertTrue(coordinator.retry())
        self.assertTrue(coordinator.wait(1))
        self.assertEqual(applied, [profile()])

    def test_quarantine_summary_is_bounded_and_published(self) -> None:
        scheduler = BasicScheduler(hardware(), 500)
        events = []
        coordinator = NativeV2IdleTuningCoordinator(
            scheduler, publish=lambda name, payload: events.append((name, payload))
        )
        profile_id = "a" * 24
        snapshot = coordinator.update_quarantine(2, profile_id)
        self.assertEqual(snapshot.quarantined_profiles, 2)
        self.assertEqual(snapshot.latest_quarantined_profile_id, profile_id)
        self.assertEqual(events[-1][1]["quarantined_profiles"], 2)
        with self.assertRaises(ValueError):
            coordinator.update_quarantine(65, profile_id)


if __name__ == "__main__":
    unittest.main()
