from __future__ import annotations

import threading
import unittest

from vllm_apple.scheduler import BasicScheduler, MaintenanceInProgressError, ScheduleRequest
from vllm_apple.vllm_metal_v2_orchestration import NativeV2IdleTuningCoordinator
from vllm_apple.vllm_metal_v2_tuning import (
    V2CandidateResult,
    V2DispatchConfiguration,
    V2PagedAttentionFamily,
    V2PagedAttentionShape,
    V2ShapeTuningDecision,
    build_v2_tuning_profile,
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

    def test_busy_scheduler_reports_waiting_without_starting_thread(self) -> None:
        scheduler = BasicScheduler(hardware(), 500)
        reservation = scheduler.admit(ScheduleRequest("paged_attention", 1))
        coordinator = NativeV2IdleTuningCoordinator(scheduler)
        self.assertFalse(coordinator.start(profile, lambda _profile: None))
        self.assertEqual(coordinator.snapshot().status, "waiting_for_idle")
        scheduler.complete(reservation)


if __name__ == "__main__":
    unittest.main()
