import unittest
from unittest.mock import patch

from vllm_apple.kernel_probe import KernelMeasurement
from vllm_apple.mlx_probe import NativeMLXProbeAdapter


class MLXQuantizedProbeTests(unittest.TestCase):
    def test_q4_q8_suite_compares_against_same_shape_dense_mlx(self):
        dense = KernelMeasurement("a" * 64, 100, (1.0, 2.0))

        def measurement(_adapter, operator):
            if operator == "_dense_matmul_quant_shape":
                return dense
            error = 0.2 if operator.endswith("q4") else 0.02
            return KernelMeasurement("b" * 64, 110, (1.0 + error, 2.0 - error))

        adapter = NativeMLXProbeAdapter()
        with patch.object(
            NativeMLXProbeAdapter, "_candidate", autospec=True,
            side_effect=measurement,
        ) as candidate:
            results = adapter.probe_quantized_matmul_suite(
                hardware_fingerprint="hardware",
                environment_fingerprint="environment",
                samples=2,
            )
        self.assertEqual([item.operator for item in results], [
            "quantized_matmul_q4", "quantized_matmul_q8"
        ])
        self.assertTrue(all(item.passed for item in results))
        self.assertEqual(candidate.call_count, 8)

    def test_q4_error_outside_fixed_budget_is_quarantined(self):
        dense = KernelMeasurement("a" * 64, 100, (1.0,))

        def measurement(_adapter, operator):
            return dense if operator.startswith("_") else KernelMeasurement(
                "b" * 64, 100, (1.31,)
            )

        adapter = NativeMLXProbeAdapter()
        with patch.object(
            NativeMLXProbeAdapter, "_candidate", autospec=True,
            side_effect=measurement,
        ):
            results = adapter.probe_quantized_matmul_suite(
                hardware_fingerprint="hardware",
                environment_fingerprint="environment",
                samples=1,
            )
        self.assertTrue(results[0].quarantined)
        self.assertEqual(results[0].reason, "correctness_mismatch")

    def test_private_dense_fixture_is_not_publicly_measurable(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            NativeMLXProbeAdapter().measure_operator("_dense_matmul_quant_shape")


if __name__ == "__main__":
    unittest.main()
