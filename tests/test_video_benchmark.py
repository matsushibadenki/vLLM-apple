import unittest

from vllm_apple.video_benchmark import (
    VideoBenchmarkThresholds,
    VideoDecodeMeasurement,
    run_video_benchmark,
)


class VideoBenchmarkTests(unittest.TestCase):
    def test_calculates_throughput_and_memory_per_video_minute(self):
        report = run_video_benchmark(
            lambda: VideoDecodeMeasurement(300, 10, 2, 10_000, 50_000),
            VideoBenchmarkThresholds(100, 4, 100_000),
        )
        self.assertEqual(report.frames_per_second, 150)
        self.assertEqual(report.video_seconds_per_second, 5)
        self.assertEqual(report.buffer_bytes_per_video_minute, 60_000)
        self.assertTrue(report.passed)

    def test_any_failed_threshold_fails_report(self):
        report = run_video_benchmark(
            lambda: VideoDecodeMeasurement(10, 10, 10, 1_000),
            VideoBenchmarkThresholds(2, 2, 100_000),
        )
        self.assertFalse(report.passed)

    def test_invalid_measurement_and_adapter_type_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "measurement"):
            VideoDecodeMeasurement(0, 1, 1, 0)
        with self.assertRaisesRegex(TypeError, "invalid measurement"):
            run_video_benchmark(lambda: object())


if __name__ == "__main__":
    unittest.main()
