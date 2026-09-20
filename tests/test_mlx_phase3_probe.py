import unittest
from unittest.mock import patch

from vllm_apple.kernel_probe import KernelMeasurement
from vllm_apple.mlx_phase3_probe import MLXPhase3ProbeAdapter


class MLXPhase3ProbeTests(unittest.TestCase):
    def test_fusion_suite_compares_staged_and_lazy_graph_candidates(self):
        adapter = MLXPhase3ProbeAdapter()

        def measurement(_adapter, operator):
            latency = 100 if operator.startswith("staged") else 90
            return KernelMeasurement("a" * 64, latency, (1.0, 2.0))

        with patch.object(
            MLXPhase3ProbeAdapter, "_measure", autospec=True,
            side_effect=measurement,
        ):
            results = adapter.probe_fusion_suite(
                hardware_fingerprint="hardware",
                environment_fingerprint="environment",
                samples=2,
            )
        self.assertEqual([result.operator for result in results], [
            "fused_q4_silu", "fused_rmsnorm_rope", "fused_moe"
        ])
        self.assertTrue(all(result.passed for result in results))

    def test_correctness_mismatch_quarantines_candidate(self):
        adapter = MLXPhase3ProbeAdapter()

        def measurement(_adapter, operator):
            values = (1.0,) if operator.startswith("staged") else (1.1,)
            return KernelMeasurement(operator[0].encode().hex().ljust(64, "0"), 100, values)

        with patch.object(
            MLXPhase3ProbeAdapter, "_measure", autospec=True,
            side_effect=measurement,
        ):
            results = adapter.probe_fusion_suite(
                hardware_fingerprint="hardware",
                environment_fingerprint="environment",
                samples=1,
            )
        self.assertTrue(all(result.quarantined for result in results))
        self.assertTrue(all(result.reason == "correctness_mismatch" for result in results))


if __name__ == "__main__":
    unittest.main()
