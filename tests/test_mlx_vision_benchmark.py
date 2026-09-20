import unittest
from unittest.mock import patch

from vllm_apple.mlx_vision_benchmark import (
    MLXVisionBenchmarkAdapter,
    VisionBenchmarkMeasurement,
)


class MLXVisionBenchmarkTests(unittest.TestCase):
    def test_reports_latency_throughput_and_memory_per_image(self):
        adapter = MLXVisionBenchmarkAdapter()

        def measurement(_adapter, batch_size):
            return VisionBenchmarkMeasurement(batch_size, batch_size * 100, 1000, "a" * 64)

        with patch.object(
            MLXVisionBenchmarkAdapter, "_measure", autospec=True, side_effect=measurement
        ):
            report = adapter.benchmark(
                environment_fingerprint="mlx-test", batch_sizes=(1, 2), samples=3
            )
        self.assertTrue(report.passed)
        self.assertEqual(report.measurements[0].images_per_second, 10_000_000.0)
        self.assertEqual(report.measurements[1].maximum_memory_per_image_bytes, 500)

    def test_digest_mismatch_fails_closed(self):
        adapter = MLXVisionBenchmarkAdapter()
        calls = []

        def measurement(_adapter, batch_size):
            calls.append(True)
            return VisionBenchmarkMeasurement(
                batch_size, 100, 1000, ("a" if len(calls) == 1 else "b") * 64
            )

        with patch.object(
            MLXVisionBenchmarkAdapter, "_measure", autospec=True, side_effect=measurement
        ):
            report = adapter.benchmark(
                environment_fingerprint="mlx-test", batch_sizes=(1,), samples=2
            )
        self.assertFalse(report.passed)
        self.assertFalse(report.measurements[0].deterministic)

    def test_rejects_unbounded_configuration(self):
        with self.assertRaisesRegex(ValueError, "configuration"):
            MLXVisionBenchmarkAdapter().benchmark(
                environment_fingerprint="mlx-test", batch_sizes=(17,), samples=1
            )


if __name__ == "__main__":
    unittest.main()
