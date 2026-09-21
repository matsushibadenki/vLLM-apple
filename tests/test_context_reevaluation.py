import unittest

from vllm_apple.context_reevaluation import ContextCapacityReevaluator
from vllm_apple.memory_admission import MemoryPressureAdmissionError
from vllm_apple.scheduler import ScheduleRequest
from vllm_apple.service import RuntimeService
from vllm_apple.types import ModelMemorySpec
from vllm_apple.types import PowerMode, ThermalState


class ContextCapacityReevaluatorTests(unittest.TestCase):
    def test_exact_capacity_reduces_configured_context(self) -> None:
        reevaluator = ContextCapacityReevaluator(4096, 100, 1_000)
        self.assertEqual(reevaluator.snapshot().status, "pending")
        reevaluator.update(200_000, source="vllm")
        snapshot = reevaluator.snapshot()
        self.assertEqual(snapshot.status, "reduced")
        self.assertEqual(snapshot.capacity_context_tokens, 2_000)
        self.assertEqual(snapshot.effective_context_tokens, 2_000)
        self.assertEqual(snapshot.reevaluations, 1)

    def test_sufficient_capacity_preserves_configured_context(self) -> None:
        reevaluator = ContextCapacityReevaluator(4096, 100, 1_000)
        reevaluator.update(500_000, source="vllm")
        reevaluator.update(500_000, source="vllm")
        snapshot = reevaluator.snapshot()
        self.assertEqual(snapshot.status, "sufficient")
        self.assertEqual(snapshot.effective_context_tokens, 4096)
        self.assertEqual(snapshot.reevaluations, 1)

    def test_runtime_admission_uses_reduced_context_ceiling(self) -> None:
        service = RuntimeService(
            model_memory_spec=ModelMemorySpec("model", 1_000, 100),
            configured_context_tokens=4096,
        )
        service.record_kv_cache_memory(10_000, 200_000, source="vllm")
        service.admit_schedule(ScheduleRequest("decode", 0, estimated_context_tokens=2_000))
        with self.assertRaisesRegex(
            MemoryPressureAdmissionError, "backend_context_capacity_exceeded"
        ):
            service.admit_schedule(
                ScheduleRequest("decode", 0, estimated_context_tokens=2_001)
            )
        snapshot = service.snapshot().context_reevaluation
        self.assertEqual(snapshot["status"], "reduced")
        self.assertEqual(snapshot["effective_context_tokens"], 2_000)

    def test_runtime_publishes_one_event_per_capacity_change(self) -> None:
        service = RuntimeService(
            model_memory_spec=ModelMemorySpec("model", 1_000, 100),
            configured_context_tokens=4096,
        )
        service.record_kv_cache_memory(10, 200_000, source="vllm")
        sequence = service.events.snapshot()["latest_sequence"]
        service.record_kv_cache_memory(20, 200_000, source="vllm")
        self.assertEqual(service.events.snapshot()["latest_sequence"], sequence)
        subscription = service.events.subscribe(after_sequence=sequence - 1, heartbeat=0.01)
        try:
            event = next(subscription)
            self.assertEqual(event.type, "runtime.context_reevaluation")
            self.assertEqual(event.payload["status"], "reduced")
            self.assertEqual(event.payload["effective_context_tokens"], 2_000)
        finally:
            subscription.close()

    def test_workload_history_and_thermal_state_bound_dynamic_context(self) -> None:
        reevaluator = ContextCapacityReevaluator(4096, 100, 1_000)
        reevaluator.update(500_000, source="vllm")
        reevaluator.observe_workload(1024)
        reevaluator.update_thermal_state(ThermalState.FAIR)
        snapshot = reevaluator.snapshot()
        self.assertEqual(snapshot.status, "dynamic_reduced")
        self.assertEqual(snapshot.effective_context_tokens, 3072)
        reevaluator.observe_workload(3000)
        self.assertEqual(reevaluator.snapshot().effective_context_tokens, 3584)
        reevaluator.update_thermal_state(ThermalState.CRITICAL)
        self.assertEqual(reevaluator.snapshot().effective_context_tokens, 2048)
        reevaluator.update_thermal_state(ThermalState.NOMINAL)
        self.assertEqual(reevaluator.snapshot().effective_context_tokens, 4096)

    def test_runtime_operating_state_updates_context_ceiling(self) -> None:
        service = RuntimeService(
            model_memory_spec=ModelMemorySpec("model", 1_000, 100),
            configured_context_tokens=4096,
        )
        service.record_kv_cache_memory(10_000, 500_000, source="vllm")
        service.apply_operating_state(ThermalState.CRITICAL, PowerMode.AUTOMATIC)
        with self.assertRaisesRegex(
            MemoryPressureAdmissionError, "backend_context_capacity_exceeded"
        ):
            service.admit_schedule(
                ScheduleRequest("decode", 0, estimated_context_tokens=2049)
            )
        self.assertEqual(
            service.snapshot().context_reevaluation["effective_context_tokens"], 2048
        )


if __name__ == "__main__":
    unittest.main()
