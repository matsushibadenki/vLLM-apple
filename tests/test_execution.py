import unittest

from tests.schema_validator import validate_instance
from tests.test_schemas import load_schema
from vllm_apple.context import ContextPolicy, recommend_context, recommend_state_context
from vllm_apple.execution import (
    AppleChipProfile,
    AppleExecutionPlanner,
    ExecutionBackend,
)
from vllm_apple.types import GIB, MemoryInfo, MemoryPressure, ModelMemorySpec, StateMemorySpec


class StateMemorySpecTests(unittest.TestCase):
    def test_legacy_model_spec_produces_same_context(self) -> None:
        memory = MemoryInfo(16 * GIB, 14 * GIB)
        legacy = ModelMemorySpec("model", 4 * GIB, 128 * 1024, GIB)
        self.assertEqual(
            recommend_context(memory, legacy),
            recommend_state_context(memory, legacy.as_state_memory_spec()),
        )

    def test_window_state_is_bounded_but_regular_kv_continues(self) -> None:
        spec = StateMemorySpec(
            model_id="hybrid",
            architecture="hybrid_swa",
            weights_bytes=GIB,
            kv_bytes_per_token=100,
            recurrent_state_bytes=4096,
            attention_window_bytes_per_token=50,
            attention_window_tokens=1024,
        )
        self.assertEqual(spec.state_bytes(2048), 4096 + 2048 * 100 + 1024 * 50)

    def test_sparse_index_grows_but_retrieval_workspace_is_bounded(self) -> None:
        spec = StateMemorySpec(
            model_id="qsa",
            architecture="sparse",
            weights_bytes=GIB,
            kv_bytes_per_token=100,
            sparse_index_bytes_per_token=10,
            sparse_retrieval_bytes_per_token=20,
            sparse_retrieval_tokens=2048,
        )
        self.assertEqual(spec.state_bytes(4096), 4096 * 110 + 2048 * 20)

    def test_fixed_state_requires_a_context_limit(self) -> None:
        with self.assertRaises(ValueError):
            StateMemorySpec("recurrent", "mamba", GIB, recurrent_state_bytes=1024)
        with self.assertRaises(ValueError):
            StateMemorySpec(
                "windowed",
                "swa",
                GIB,
                attention_window_bytes_per_token=1024,
                attention_window_tokens=4096,
            )


class AppleExecutionPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = MemoryInfo(16 * GIB, 14 * GIB, pressure=MemoryPressure.NORMAL)
        self.chip = AppleChipProfile(
            profile_version=1,
            hardware_fingerprint="m4-16g-test",
            soc="Apple M4",
            total_memory_bytes=16 * GIB,
            backends=(ExecutionBackend.VLLM_METAL, ExecutionBackend.CPU),
            precisions=("fp16", "int8"),
        )
        self.model = StateMemorySpec(
            "test/model", "transformer", 4 * GIB, kv_bytes_per_token=128 * 1024
        )

    def test_dry_run_is_deterministic_schema_valid_and_bounded(self) -> None:
        planner = AppleExecutionPlanner()
        first = planner.plan(model=self.model, memory=self.memory, chip=self.chip)
        second = planner.plan(model=self.model, memory=self.memory, chip=self.chip)
        self.assertEqual(first, second)
        self.assertLessEqual(first.estimated_peak_bytes, first.memory_ceiling_bytes)
        self.assertEqual(first.prefill.backend, ExecutionBackend.VLLM_METAL)
        self.assertGreater(first.prefill.batch_size, first.decode.batch_size)
        validate_instance(
            first.to_dict(), load_schema("runtime/apple-execution-plan-v1.schema.json")
        )
        validate_instance(
            self.model.to_dict(), load_schema("runtime/state-memory-spec-v1.schema.json")
        )

    def test_pressure_reduces_batch_and_precision(self) -> None:
        pressured = MemoryInfo(
            16 * GIB, 12 * GIB, pressure=MemoryPressure.WARNING
        )
        plan = AppleExecutionPlanner().plan(
            model=self.model, memory=pressured, chip=self.chip
        )
        self.assertEqual(plan.prefill.batch_size, 1)
        self.assertEqual(plan.decode.state_precision, "int8")

    def test_requested_context_is_clamped(self) -> None:
        plan = AppleExecutionPlanner(ContextPolicy(token_block_size=1)).plan(
            model=self.model,
            memory=self.memory,
            chip=self.chip,
            requested_context_tokens=10**9,
        )
        self.assertLess(plan.context_tokens, 10**9)
        self.assertIn("context:clamped", plan.decision_reasons)

    def test_fixed_memory_over_ceiling_is_rejected(self) -> None:
        oversized = StateMemorySpec(
            "huge", "mamba", 15 * GIB, recurrent_state_bytes=GIB, model_max_context=4096
        )
        with self.assertRaises(ValueError):
            AppleExecutionPlanner().plan(model=oversized, memory=self.memory, chip=self.chip)


if __name__ == "__main__":
    unittest.main()
