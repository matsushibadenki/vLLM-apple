import unittest
from dataclasses import replace
import threading
import time

from vllm_apple.execution import (
    AppleExecutionPlan,
    ExecutionBackend,
    PhaseExecutionPlan,
    WorkloadPhase,
)
from vllm_apple.scheduler import (
    BasicScheduler,
    ExecutionPlanAdmissionError,
    MaintenanceInProgressError,
    MemoryCapacityError,
    ScheduleRequest,
)
from vllm_apple.types import Backend, HardwareInfo, MemoryInfo, Priority


def hardware(apple: bool = True) -> HardwareInfo:
    return HardwareInfo(
        platform="Darwin" if apple else "Linux",
        architecture="arm64",
        soc="Test",
        physical_cpu_count=8,
        logical_cpu_count=8,
        gpu_core_count=10,
        memory=MemoryInfo(total_bytes=1_000, available_bytes=800),
        is_apple_silicon=apple,
        os_version="test",
    )


def execution_plan(plan_id: str, prefill_batch: int) -> AppleExecutionPlan:
    return AppleExecutionPlan(
        schema_version=1,
        plan_id=plan_id,
        model_id="model",
        hardware_fingerprint="hardware",
        context_tokens=1024,
        memory_ceiling_bytes=500,
        estimated_peak_bytes=400,
        prefill=PhaseExecutionPlan(
            WorkloadPhase.PREFILL,
            ExecutionBackend.VLLM_METAL,
            prefill_batch,
            "fp16",
        ),
        decode=PhaseExecutionPlan(
            WorkloadPhase.DECODE, ExecutionBackend.VLLM_METAL, 1, "fp16"
        ),
        fallback_chain=(ExecutionBackend.CPU,),
        decision_reasons=("test",),
        dry_run=False,
    )


class SchedulerTests(unittest.TestCase):
    def test_idle_maintenance_is_exclusive_and_blocks_admission(self) -> None:
        scheduler = BasicScheduler(hardware(), 500)
        self.assertTrue(scheduler.begin_idle_maintenance("native-v2"))
        self.assertFalse(scheduler.begin_idle_maintenance("other"))
        with self.assertRaises(MaintenanceInProgressError):
            scheduler.admit(ScheduleRequest("paged_attention", 1))
        scheduler.end_idle_maintenance("native-v2")
        reservation = scheduler.admit(ScheduleRequest("paged_attention", 1))
        scheduler.complete(reservation)

    def test_idle_maintenance_waits_for_active_reservations(self) -> None:
        scheduler = BasicScheduler(hardware(), 500)
        reservation = scheduler.admit(ScheduleRequest("paged_attention", 1))
        self.assertFalse(scheduler.begin_idle_maintenance("native-v2"))
        scheduler.complete(reservation)
        self.assertTrue(scheduler.begin_idle_maintenance("native-v2"))
        scheduler.end_idle_maintenance("native-v2")

    def test_backend_choice_avoids_launch_overhead_for_tiny_decode(self) -> None:
        scheduler = BasicScheduler(hardware(), 500)
        self.assertEqual(scheduler.choose_backend(ScheduleRequest("gemv", 10)), Backend.CPU)
        self.assertEqual(
            scheduler.choose_backend(ScheduleRequest("gemm", 10, batch_size=8)), Backend.MLX_GPU
        )
        self.assertEqual(
            scheduler.choose_backend(ScheduleRequest("paged_attention", 10)), Backend.METAL
        )

    def test_reservations_never_exceed_capacity(self) -> None:
        scheduler = BasicScheduler(hardware(), 100)
        reservation = scheduler.admit(
            ScheduleRequest("attention", 80, priority=Priority.INTERACTIVE)
        )
        with self.assertRaises(MemoryCapacityError):
            scheduler.admit(ScheduleRequest("attention", 21))
        self.assertEqual(scheduler.memory.reserved_bytes, 80)
        scheduler.complete(reservation)
        self.assertEqual(scheduler.memory.reserved_bytes, 0)
        scheduler.complete(reservation)
        self.assertEqual(scheduler.memory.reserved_bytes, 0)

    def test_plan_changes_are_deferred_and_batch_limits_do_not_mix(self) -> None:
        scheduler = BasicScheduler(hardware(), 500)
        initial = execution_plan("a" * 24, 4)
        constrained = execution_plan("b" * 24, 1)
        self.assertEqual(scheduler.request_execution_plan(initial).status, "applied")
        reservation = scheduler.admit(ScheduleRequest("decode", 10))
        self.assertEqual(reservation.execution_plan_id, initial.plan_id)
        self.assertEqual(scheduler.request_execution_plan(constrained).status, "deferred")
        old_policy_work = scheduler.admit(ScheduleRequest("prefill", 10, batch_size=4))
        scheduler.complete(reservation)
        self.assertEqual(scheduler.apply_pending_execution_plan().status, "deferred")
        scheduler.complete(old_policy_work)
        self.assertEqual(scheduler.apply_pending_execution_plan().status, "applied")
        with self.assertRaises(ExecutionPlanAdmissionError):
            scheduler.admit(ScheduleRequest("prefill", 10, batch_size=2))
        self.assertEqual(
            scheduler.execution_plan_snapshot()["active_plan_id"], constrained.plan_id
        )

    def test_dry_run_plan_cannot_be_activated(self) -> None:
        scheduler = BasicScheduler(hardware(), 500)
        plan = execution_plan("c" * 24, 1)
        dry_run = replace(plan, dry_run=True)
        with self.assertRaises(ValueError):
            scheduler.request_execution_plan(dry_run)

    def test_safe_point_blocks_new_admission_until_operation_finishes(self) -> None:
        scheduler = BasicScheduler(hardware(), 500)
        entered = threading.Event()
        release = threading.Event()
        admitted = threading.Event()

        def policy_update() -> None:
            entered.set()
            release.wait(timeout=2)

        update = threading.Thread(target=scheduler.at_safe_point, args=(policy_update,))
        update.start()
        self.assertTrue(entered.wait(timeout=1))

        def admit() -> None:
            scheduler.admit(ScheduleRequest("decode", 1))
            admitted.set()

        admission = threading.Thread(target=admit)
        admission.start()
        time.sleep(0.02)
        self.assertFalse(admitted.is_set())
        release.set()
        update.join(timeout=1)
        admission.join(timeout=1)
        self.assertTrue(admitted.is_set())


if __name__ == "__main__":
    unittest.main()
