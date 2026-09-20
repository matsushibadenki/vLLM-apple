import unittest
from unittest.mock import patch

from vllm_apple.kernel_probe import KernelMeasurement
from vllm_apple.mlx_vision_probe import MLXVisionFusionProbeAdapter


class MLXVisionFusionProbeTests(unittest.TestCase):
    def test_compares_staged_and_single_lazy_graph(self):
        adapter = MLXVisionFusionProbeAdapter()

        def measurement(_adapter, operator):
            latency = 100 if operator.startswith("staged") else 80
            return KernelMeasurement("a" * 64, latency, (1.0, 2.0))

        with patch.object(
            MLXVisionFusionProbeAdapter,
            "_measure",
            autospec=True,
            side_effect=measurement,
        ):
            result = adapter.probe_fusion(
                hardware_fingerprint="hardware",
                environment_fingerprint="environment",
                samples=2,
            )
        self.assertEqual(result.operator, "fused_vision_preprocess_projection")
        self.assertTrue(result.passed)

    def test_correctness_mismatch_quarantines_fusion(self):
        adapter = MLXVisionFusionProbeAdapter()

        def measurement(_adapter, operator):
            values = (1.0,) if operator.startswith("staged") else (1.01,)
            digest = "a" * 64 if operator.startswith("staged") else "b" * 64
            return KernelMeasurement(digest, 100, values)

        with patch.object(
            MLXVisionFusionProbeAdapter,
            "_measure",
            autospec=True,
            side_effect=measurement,
        ):
            result = adapter.probe_fusion(
                hardware_fingerprint="hardware",
                environment_fingerprint="environment",
                samples=1,
            )
        self.assertTrue(result.quarantined)
        self.assertEqual(result.reason, "correctness_mismatch")


if __name__ == "__main__":
    unittest.main()
