import unittest

from vllm_apple.memory_budget import UnifiedMemoryBudgetLedger
from vllm_apple.memory_admission import MemoryPressureAdmissionError
from vllm_apple.scheduler import ScheduleRequest
from vllm_apple.service import RuntimeService


class UnifiedMemoryBudgetLedgerTests(unittest.TestCase):
    def test_additive_components_and_overlap_envelope_are_not_double_counted(self) -> None:
        ledger = UnifiedMemoryBudgetLedger(1_000)
        ledger.update("weights", 400, source="model_manifest")
        ledger.update("kv", 100, source="vllm")
        ledger.update("recurrent", 10, source="model")
        ledger.update("prefix", 50, source="semantic_state")
        ledger.update("window", 15, source="runtime")
        ledger.update("experts", 20, source="model")
        ledger.update("ngram", 5, source="model")
        ledger.update("mtp", 5, source="model")
        ledger.update("scratch", 25, source="scheduler")
        ledger.update("metal_heap", 700, source="ioreg")
        ledger.update("coreml", 75, source="coreml")

        snapshot = ledger.snapshot()
        self.assertEqual(snapshot.known_component_bytes, 705)
        self.assertEqual(snapshot.known_remaining_bytes, 295)
        self.assertEqual(snapshot.overlap_envelope_bytes, 700)
        self.assertEqual(snapshot.unknown_components, ())
        self.assertEqual(snapshot.components["metal_heap"].accounting, "overlap_envelope")

    def test_unknown_values_remain_unknown_and_peaks_are_monotonic(self) -> None:
        ledger = UnifiedMemoryBudgetLedger(100)
        self.assertIn("weights", ledger.snapshot().unknown_components)
        ledger.update("weights", 80, source="manifest")
        ledger.update("weights", 60, source="manifest")
        ledger.update("kv", None, source=None)
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot.components["weights"].current_bytes, 60)
        self.assertEqual(snapshot.components["weights"].peak_bytes, 80)
        self.assertIsNone(snapshot.components["kv"].current_bytes)

    def test_overcommit_and_invalid_updates_are_reported_safely(self) -> None:
        ledger = UnifiedMemoryBudgetLedger(100)
        ledger.update("weights", 120, source="manifest")
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot.known_remaining_bytes, 0)
        self.assertEqual(snapshot.overcommitted_bytes, 20)
        with self.assertRaises(ValueError):
            ledger.update("unknown", 1, source="test")
        with self.assertRaises(ValueError):
            ledger.update("kv", 1, source=None)
        with self.assertRaises(ValueError):
            ledger.update("kv", 1, source="")


class RuntimeMemoryBudgetTests(unittest.TestCase):
    def test_runtime_reconciles_measured_and_owned_components(self) -> None:
        service = RuntimeService()
        service.record_memory_budget_component("weights", 400, source="model_manifest")
        service.record_memory_budget_component("coreml", 20, source="coreml")
        service.record_kv_cache_memory(100, 200, source="vllm")
        service.record_framework_memory(300, source="mlx")

        budget = service.snapshot().memory_budget
        components = budget["components"]
        self.assertEqual(components["weights"]["current_bytes"], 400)
        self.assertEqual(components["kv"]["current_bytes"], 100)
        self.assertEqual(components["prefix"]["current_bytes"], 0)
        self.assertEqual(components["scratch"]["current_bytes"], 0)
        self.assertEqual(components["metal_heap"]["current_bytes"], 300)
        self.assertEqual(components["metal_heap"]["accounting"], "overlap_envelope")

    def test_runtime_rejects_known_model_weight_overcommit(self) -> None:
        service = RuntimeService()
        capacity = service.memory_budget.snapshot().capacity_bytes
        service.record_memory_budget_component(
            "weights", capacity + 1, source="model_weight_files"
        )
        with self.assertRaisesRegex(
            MemoryPressureAdmissionError, "memory_budget_overcommitted"
        ):
            service.admit_schedule(ScheduleRequest("decode", 0))
        self.assertEqual(
            service.memory_admission.snapshot().last_rejection_reason,
            "memory_budget_overcommitted",
        )

    def test_state_spec_accounts_fixed_and_request_dependent_memory(self) -> None:
        from vllm_apple.types import StateMemorySpec

        spec = StateMemorySpec(
            "hybrid",
            "hybrid_moe",
            weights_bytes=100,
            kv_bytes_per_token=2,
            recurrent_state_bytes=30,
            prefix_state_bytes=20,
            attention_window_bytes_per_token=3,
            attention_window_tokens=10,
            expert_working_set_bytes=40,
            ngram_storage_bytes=1_000,
            ngram_working_set_bytes=60,
            mtp_storage_bytes=2_000,
            mtp_working_set_bytes=70,
            scratch_workspace_bytes=50,
        )
        service = RuntimeService(state_memory_spec=spec)
        budget = service.memory_budget_snapshot()
        self.assertEqual(budget.components["weights"].current_bytes, 100)
        self.assertEqual(budget.components["recurrent"].current_bytes, 30)
        self.assertEqual(budget.components["prefix"].current_bytes, 20)
        self.assertEqual(budget.components["experts"].current_bytes, 40)
        self.assertEqual(budget.components["ngram"].current_bytes, 60)
        self.assertEqual(budget.components["mtp"].current_bytes, 70)
        self.assertEqual(budget.components["scratch"].current_bytes, 50)

        reservation = service.admit_schedule(
            ScheduleRequest("decode", 7, batch_size=2, estimated_context_tokens=20)
        )
        # 7 explicit + 2 batches * (20 * 2 KV + 10 * 3 window)
        # + one additional batch of recurrent/prefix state.
        self.assertEqual(reservation.bytes, 247)
        service.complete_schedule(reservation)


if __name__ == "__main__":
    unittest.main()
