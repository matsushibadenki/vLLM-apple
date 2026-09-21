import unittest

from vllm_apple.execution import ExecutionBackend, WorkloadPhase
from vllm_apple.numeric_routing import (
    NumericCapability,
    NumericEligibilityMatrix,
    NumericEligibilityRequest,
    NumericFormat,
    NumericRouteProfile,
    NumericRouteStrategy,
    NumericTensorRole,
    choose_numeric_route,
)


class NumericRoutingTests(unittest.TestCase):
    def capability(self, *, qualified=True):
        return NumericCapability(
            NumericFormat.NVFP4_E2M1, NumericFormat.INT8,
            NumericTensorRole.WEIGHT, ExecutionBackend.NATIVE_MLX,
            "gemv", 16, 4096, "nvfp4.block16.v1", "a" * 64, qualified,
        )

    def request(self, **changes):
        values = dict(
            source_format=NumericFormat.NVFP4_E2M1,
            compute_format=NumericFormat.INT8,
            tensor_role=NumericTensorRole.WEIGHT,
            backend=ExecutionBackend.NATIVE_MLX,
            operator="gemv", elements=1024, recipe_id="nvfp4.block16.v1",
        )
        values.update(changes)
        return NumericEligibilityRequest(**values)

    def test_exact_qualified_recipe_is_eligible(self):
        capability = self.capability()
        decision = NumericEligibilityMatrix((capability,)).decide(self.request())
        self.assertTrue(decision.eligible)
        self.assertEqual(decision.capability_id, capability.capability_id)

    def test_unknown_or_unqualified_recipe_fails_closed(self):
        matrix = NumericEligibilityMatrix((self.capability(qualified=False),))
        self.assertEqual(matrix.decide(self.request()).reason, "capability_not_qualified")
        self.assertEqual(
            matrix.decide(self.request(recipe_id="unknown.v1")).reason,
            "unsupported_recipe",
        )

    def test_route_uses_measured_total_and_memory_ceiling(self):
        capability_id = self.capability().capability_id
        common = dict(
            capability_id=capability_id, phase=WorkloadPhase.DECODE,
            synchronization_nanoseconds=10, compute_nanoseconds=100,
            amortized_uses=10, output_digest="b" * 64,
        )
        profiles = (
            NumericRouteProfile(
                strategy=NumericRouteStrategy.CACHED_CONVERT,
                conversion_nanoseconds=1000, peak_memory_bytes=2000, **common,
            ),
            NumericRouteProfile(
                strategy=NumericRouteStrategy.FUSED_EVERY_USE,
                conversion_nanoseconds=20, peak_memory_bytes=1000, **common,
            ),
        )
        decision = choose_numeric_route(
            profiles, capability_id=capability_id, phase=WorkloadPhase.DECODE,
            memory_ceiling_bytes=1500,
        )
        self.assertEqual(decision.strategy, NumericRouteStrategy.FUSED_EVERY_USE)
        with self.assertRaisesRegex(ValueError, "output mismatch"):
            choose_numeric_route(
                profiles + (NumericRouteProfile(
                    strategy=NumericRouteStrategy.LOAD_CONVERT,
                    conversion_nanoseconds=1, peak_memory_bytes=1000,
                    **(common | {"output_digest": "c" * 64}),
                ),), capability_id=capability_id, phase=WorkloadPhase.DECODE,
                memory_ceiling_bytes=1500,
            )


if __name__ == "__main__":
    unittest.main()
