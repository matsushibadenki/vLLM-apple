import unittest
from dataclasses import replace
import threading
import time
from unittest.mock import patch

from vllm_apple.execution import (
    AppleExecutionPlan,
    ExecutionBackend,
    PhaseExecutionPlan,
    WorkloadPhase,
)
from vllm_apple.device_resources import (
    BandwidthContentionEvidence,
    DeviceResourceCapacityError,
    DeviceResourceRequest,
    UnifiedDeviceResourceLedger,
)
from vllm_apple.device_pipeline import DevicePipelineStage
from vllm_apple.operator_dispatch import OperatorDispatchDecision
from vllm_apple.scheduler import (
    AdaptiveScheduleCapacityError,
    BasicScheduler,
    ExecutionPlanAdmissionError,
    MaintenanceInProgressError,
    MemoryCapacityError,
    ScheduleQueueFullError,
    ScheduleRequest,
)
from vllm_apple.types import (
    Backend, HardwareInfo, MemoryInfo, MemoryPressure, PowerMode, Priority, ThermalState,
)


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
    def test_preference_never_overrides_thermal_or_memory_safety(self) -> None:
        scheduler = BasicScheduler(hardware(), 500)
        self.assertEqual(scheduler.set_scheduling_preference("low_power"), "applied")
        self.assertEqual(scheduler.adaptive_scheduling_snapshot()["level"], 1)
        active = scheduler.admit(ScheduleRequest("attention", 10))
        self.assertEqual(scheduler.set_scheduling_preference("high_performance"), "deferred")
        self.assertEqual(scheduler.adaptive_scheduling_snapshot()["pending_level"], 0)
        scheduler.update_adaptive_inputs(thermal=ThermalState.CRITICAL)
        self.assertEqual(scheduler.adaptive_scheduling_snapshot()["level"], 2)
        self.assertIsNone(scheduler.adaptive_scheduling_snapshot()["pending_level"])
        scheduler.complete(active)
        self.assertEqual(scheduler.adaptive_scheduling_snapshot()["level"], 2)
        scheduler.update_adaptive_inputs(thermal=ThermalState.NOMINAL)
        scheduler.update_adaptive_inputs(pressure=MemoryPressure.CRITICAL)
        self.assertEqual(scheduler.adaptive_scheduling_snapshot()["level"], 2)
        with self.assertRaises(ValueError):
            scheduler.set_scheduling_preference("unsafe")
        self.assertEqual(
            scheduler.adaptive_scheduling_snapshot()["preference"], "high_performance"
        )

    def test_adaptive_policy_limits_new_work_without_cancelling_active_work(self) -> None:
        scheduler = BasicScheduler(hardware(), 500)
        active = scheduler.admit(ScheduleRequest("attention", 10, batch_size=8))
        self.assertEqual(
            scheduler.update_adaptive_inputs(pressure=MemoryPressure.WARNING), "applied"
        )
        self.assertEqual(scheduler.adaptive_scheduling_snapshot()["level"], 1)
        with self.assertRaises(AdaptiveScheduleCapacityError):
            scheduler.execute_device_pipeline((
                DevicePipelineStage("cpu", ExecutionBackend.CPU, 1, lambda: 1),
                DevicePipelineStage("gpu", ExecutionBackend.NATIVE_MLX, 1, lambda: 2),
            ))
        with self.assertRaisesRegex(ExecutionPlanAdmissionError, "adaptive"):
            scheduler.admit(ScheduleRequest("attention", 10, batch_size=8))
        second = scheduler.admit(ScheduleRequest("attention", 10))
        with self.assertRaises(AdaptiveScheduleCapacityError):
            scheduler.admit(ScheduleRequest("attention", 10))
        self.assertEqual(scheduler.memory.reserved_bytes, 20)
        self.assertEqual(
            scheduler.update_adaptive_inputs(pressure=MemoryPressure.NORMAL), "deferred"
        )
        self.assertEqual(scheduler.adaptive_scheduling_snapshot()["level"], 1)
        scheduler.complete(second)
        self.assertEqual(scheduler.adaptive_scheduling_snapshot()["pending_level"], 0)
        scheduler.complete(active)
        self.assertEqual(scheduler.adaptive_scheduling_snapshot()["level"], 0)

    def test_critical_state_routes_new_work_to_cpu_and_disables_parallelism(self) -> None:
        scheduler = BasicScheduler(hardware(), 500)
        self.assertEqual(
            scheduler.update_adaptive_inputs(
                thermal=ThermalState.CRITICAL, power=PowerMode.LOW_POWER
            ), "applied"
        )
        self.assertEqual(scheduler.choose_backend(ScheduleRequest("attention", 10)), Backend.CPU)
        with self.assertRaisesRegex(ExecutionPlanAdmissionError, "adaptive"):
            scheduler.admit(ScheduleRequest("attention", 10, batch_size=2))
        reservation = scheduler.admit(ScheduleRequest("attention", 10))
        with self.assertRaises(AdaptiveScheduleCapacityError):
            scheduler.admit(ScheduleRequest("attention", 10))
        self.assertEqual(
            scheduler.update_adaptive_inputs(thermal=ThermalState.UNKNOWN), "ignored"
        )
        scheduler.complete(reservation)
        self.assertEqual(
            scheduler.update_adaptive_inputs(
                thermal=ThermalState.NOMINAL, power=PowerMode.AUTOMATIC
            ), "applied"
        )
        self.assertEqual(scheduler.adaptive_scheduling_snapshot()["level"], 0)

    def test_work_steal_is_bounded_and_preserves_priority(self) -> None:
        class ProbedDispatcher:
            def dispatch(self, request):
                return OperatorDispatchDecision(
                    request.operator,
                    ExecutionBackend.NATIVE_MLX,
                    (ExecutionBackend.CPU,),
                    (), (), "preferred_probe_passed",
                )

        scheduler = BasicScheduler(hardware(), 500, ProbedDispatcher())
        profile_id = scheduler.device_resources.snapshot()["contention_profile_id"]
        source = scheduler.device_resources.reserve(
            DeviceResourceRequest.for_backend(ExecutionBackend.NATIVE_MLX, 10)
        )
        background = scheduler.submit(ScheduleRequest("attention", 10, Priority.BACKGROUND))
        realtime = scheduler.submit(ScheduleRequest("attention", 10, Priority.REALTIME))
        self.assertIsNone(scheduler.steal_next(ExecutionBackend.CPU))
        scheduler.device_resources.install_contention_evidence(
            BandwidthContentionEvidence(
                profile_id, ExecutionBackend.NATIVE_MLX, ExecutionBackend.CPU,
                100, 90, 3, True,
            )
        )
        stolen = scheduler.steal_next(ExecutionBackend.CPU)
        self.assertIsNotNone(stolen)
        self.assertEqual(stolen.token, realtime)
        self.assertEqual(stolen.reservation.backend, Backend.CPU)
        self.assertEqual(scheduler.scheduling_observability_snapshot()["steals"], 1)
        self.assertIsNone(scheduler.steal_next(ExecutionBackend.CPU))
        self.assertEqual(scheduler.queue_snapshot()["background"], 1)
        self.assertTrue(scheduler.complete_queued(realtime))
        self.assertEqual(scheduler.admit_next(timeout=0).token, background)
        self.assertTrue(scheduler.complete_queued(background))
        scheduler.device_resources.release(source.reservation_id)

    def test_work_steal_capacity_failure_restores_fifo_and_memory(self) -> None:
        class ProbedDispatcher:
            def dispatch(self, request):
                return OperatorDispatchDecision(
                    request.operator, ExecutionBackend.NATIVE_MLX,
                    (ExecutionBackend.CPU,), (), (), "preferred_probe_passed",
                )

        scheduler = BasicScheduler(hardware(), 10, ProbedDispatcher())
        profile_id = scheduler.device_resources.snapshot()["contention_profile_id"]
        scheduler.device_resources.install_contention_evidence(
            BandwidthContentionEvidence(
                profile_id, ExecutionBackend.NATIVE_MLX, ExecutionBackend.CPU,
                100, 90, 3, True,
            )
        )
        source = scheduler.device_resources.reserve(
            DeviceResourceRequest.for_backend(ExecutionBackend.NATIVE_MLX, 1)
        )
        first = scheduler.submit(ScheduleRequest("attention", 11))
        second = scheduler.submit(ScheduleRequest("attention", 1))
        self.assertIsNone(scheduler.steal_next(ExecutionBackend.CPU))
        self.assertEqual(scheduler.memory.reserved_bytes, 0)
        self.assertEqual(scheduler.queue_snapshot()["queued"], 2)
        self.assertEqual(scheduler._queue.peek()[0], first)
        self.assertTrue(scheduler.cancel(first))
        self.assertEqual(scheduler.steal_next(ExecutionBackend.CPU).token, second)
        self.assertTrue(scheduler.complete_queued(second))
        scheduler.device_resources.release(source.reservation_id)

    def test_executes_only_contention_qualified_device_pipeline(self) -> None:
        scheduler = BasicScheduler(hardware(), 500)
        profile_id = scheduler.device_resources.snapshot()["contention_profile_id"]
        scheduler.device_resources.install_contention_evidence(
            BandwidthContentionEvidence(
                profile_id,
                ExecutionBackend.CPU,
                ExecutionBackend.NATIVE_MLX,
                100,
                90,
                3,
                True,
            )
        )
        result = scheduler.execute_device_pipeline((
            DevicePipelineStage("tokenize", ExecutionBackend.CPU, 10, lambda: "tokens"),
            DevicePipelineStage(
                "prefill", ExecutionBackend.NATIVE_MLX, 20, lambda: "hidden-state"
            ),
        ))
        self.assertEqual(result.outputs, ("tokens", "hidden-state"))
        self.assertEqual(scheduler.device_resources.snapshot()["active_reservations"], 0)

    def test_priority_queue_is_fifo_within_each_priority(self) -> None:
        scheduler = BasicScheduler(hardware(), 500)
        background = scheduler.submit(ScheduleRequest("decode", 1, Priority.BACKGROUND))
        normal_first = scheduler.submit(ScheduleRequest("decode", 1, Priority.NORMAL))
        realtime = scheduler.submit(ScheduleRequest("decode", 1, Priority.REALTIME))
        normal_second = scheduler.submit(ScheduleRequest("decode", 1, Priority.NORMAL))
        expected = (realtime, normal_first, normal_second, background)
        admitted = tuple(scheduler.admit_next(timeout=0) for _ in expected)
        self.assertEqual(tuple(item.token for item in admitted if item), expected)
        for item in admitted:
            self.assertIsNotNone(item)
            scheduler.complete_queued(item.token)

    def test_queue_capacity_and_cancellation_release_reservations(self) -> None:
        scheduler = BasicScheduler(hardware(), 100, maximum_queued_requests=1)
        queued = scheduler.submit(ScheduleRequest("decode", 80))
        with self.assertRaises(ScheduleQueueFullError):
            scheduler.submit(ScheduleRequest("decode", 1))
        self.assertTrue(scheduler.cancel(queued))
        active_token = scheduler.submit(ScheduleRequest("decode", 80))
        admission = scheduler.admit_next(timeout=0)
        self.assertIsNotNone(admission)
        self.assertEqual(scheduler.memory.reserved_bytes, 80)
        self.assertTrue(scheduler.cancel(active_token))
        self.assertEqual(scheduler.memory.reserved_bytes, 0)
        self.assertFalse(scheduler.cancel(active_token))

    def test_queue_snapshot_is_bounded_and_does_not_expose_tokens(self) -> None:
        scheduler = BasicScheduler(hardware(), 100, maximum_queued_requests=4)
        scheduler.submit(ScheduleRequest("decode", 1, Priority.INTERACTIVE))
        snapshot = scheduler.queue_snapshot()
        self.assertEqual(snapshot["queued"], 1)
        self.assertEqual(snapshot["interactive"], 1)
        self.assertEqual(snapshot["capacity"], 4)
        self.assertNotIn("token", snapshot)

    def test_cancellation_during_dispatch_releases_new_reservation(self) -> None:
        scheduler = BasicScheduler(hardware(), 100)
        token = scheduler.submit(ScheduleRequest("decode", 80))
        entered = threading.Event()
        proceed = threading.Event()
        original_admit = scheduler.admit

        def delayed_admit(request: ScheduleRequest):
            entered.set()
            proceed.wait(timeout=2)
            return original_admit(request)

        result: list[object] = []
        with patch.object(scheduler, "admit", side_effect=delayed_admit):
            worker = threading.Thread(target=lambda: result.append(scheduler.admit_next(timeout=0)))
            worker.start()
            self.assertTrue(entered.wait(timeout=1))
            self.assertTrue(scheduler.cancel(token))
            proceed.set()
            worker.join(timeout=1)
        self.assertEqual(result, [None])
        self.assertEqual(scheduler.memory.reserved_bytes, 0)
        self.assertEqual(scheduler.queue_snapshot()["active"], 0)

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

    def test_vllm_metal_dispatch_uses_gpu_backend_and_resource(self) -> None:
        class VLLMMetalDispatcher:
            def dispatch(self, request):
                return OperatorDispatchDecision(
                    request.operator, ExecutionBackend.VLLM_METAL,
                    (ExecutionBackend.CPU,), (), (), "test",
                )

        scheduler = BasicScheduler(hardware(), 100, VLLMMetalDispatcher())
        reservation = scheduler.admit(ScheduleRequest("paged_attention", 40))
        self.assertEqual(reservation.backend, Backend.METAL)
        self.assertEqual(
            scheduler.device_resources.snapshot()["used"]["gpu_command_queues"], 1
        )
        scheduler.complete(reservation)

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

    def test_device_resource_failure_rolls_back_memory_reservation(self) -> None:
        scheduler = BasicScheduler(hardware(), 100)
        scheduler.device_resources = UnifiedDeviceResourceLedger(
            unified_memory_bytes=100, cpu_threads=1,
            gpu_command_queues=0, ane_tasks=0, bandwidth_slots=1,
        )
        with self.assertRaises(DeviceResourceCapacityError):
            scheduler.admit(ScheduleRequest("attention", 80))
        self.assertEqual(scheduler.memory.reserved_bytes, 0)
        self.assertEqual(
            scheduler.device_resources.snapshot()["active_reservations"], 0
        )

    def test_completion_releases_unified_device_resources(self) -> None:
        scheduler = BasicScheduler(hardware(), 100)
        reservation = scheduler.admit(ScheduleRequest("paged_attention", 40))
        used = scheduler.device_resources.snapshot()["used"]
        self.assertEqual(used["gpu_command_queues"], 1)
        self.assertEqual(used["unified_memory_bytes"], 40)
        scheduler.complete(reservation)
        self.assertEqual(
            scheduler.device_resources.snapshot()["active_reservations"], 0
        )

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
