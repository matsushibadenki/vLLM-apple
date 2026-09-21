import unittest

from vllm_apple.execution import ExecutionBackend
from vllm_apple.graph_fusion import CapabilityGatedGraphFusionPass
from vllm_apple.kernel_probe import (
    KernelCapabilityRegistry,
    KernelMeasurement,
    KernelProbeConfig,
    run_kernel_probe,
)


def result(operator, *, passed=True):
    digest = "a" * 64

    def baseline():
        return KernelMeasurement(digest, 100)

    def candidate():
        return KernelMeasurement(digest if passed else "b" * 64, 90)

    return run_kernel_probe(
        KernelProbeConfig(
            "hardware", "environment", ExecutionBackend.NATIVE_MLX,
            operator, samples=1,
        ),
        baseline,
        candidate,
    )


class GraphFusionTests(unittest.TestCase):
    def registry(self, *operators):
        registry = KernelCapabilityRegistry("hardware", "environment")
        for operator in operators:
            registry.record(result(operator))
        return registry

    def test_applies_only_measured_non_overlapping_fusions(self):
        graph = (
            "rms_norm", "rope", "attention", "dequant_q4", "matmul", "silu",
            "moe_route", "expert_gemm",
        )
        fused = CapabilityGatedGraphFusionPass(self.registry(
            "fused_rmsnorm_rope", "fused_q4_silu", "fused_moe"
        )).apply(graph)
        self.assertEqual(fused.operators, (
            "fused_rmsnorm_rope", "attention", "fused_q4_silu", "fused_moe"
        ))
        self.assertEqual(fused.applied, (
            "fused_rmsnorm_rope", "fused_q4_silu", "fused_moe"
        ))

    def test_missing_or_quarantined_evidence_preserves_original_graph(self):
        registry = self.registry("fused_q4_silu")
        registry.record(result("fused_rmsnorm_rope", passed=False))
        graph = ("rms_norm", "rope", "moe_route", "expert_gemm")
        fused = CapabilityGatedGraphFusionPass(registry).apply(graph)
        self.assertEqual(fused.operators, graph)
        self.assertEqual(fused.applied, ())

    def test_rejects_unbounded_or_invalid_graph(self):
        fusion = CapabilityGatedGraphFusionPass(self.registry())
        with self.assertRaises(ValueError):
            fusion.apply(("operator",) * 4097)
        with self.assertRaises(ValueError):
            fusion.apply(("",))


if __name__ == "__main__":
    unittest.main()
