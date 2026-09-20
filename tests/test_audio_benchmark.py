import unittest

from vllm_apple.audio_benchmark import AudioBenchmarkConfig, run_audio_benchmark


class AudioBenchmarkTests(unittest.TestCase):
    def test_reports_dropout_latency_and_realtime_factor(self):
        now = [0.0]

        def clock():
            value = now[0]
            now[0] += 0.001
            return value

        config = AudioBenchmarkConfig(
            sample_rate=8000,
            chunk_frames=80,
            total_frames=800,
            maximum_p95_latency_seconds=0.002,
        )
        report = run_audio_benchmark(
            config, lambda sequence, samples, frames, final: frames, clock=clock
        )
        self.assertEqual(report.chunks, 10)
        self.assertEqual(report.dropout_ratio, 0)
        self.assertEqual(report.p95_latency_seconds, 0.001)
        self.assertEqual(report.realtime_factor, 0.1)
        self.assertTrue(report.passed)

    def test_partial_and_failed_chunks_are_counted_as_dropout(self):
        now = [0.0]

        def clock():
            return now[0]

        def process(sequence, samples, frames, final):
            if sequence == 1:
                raise RuntimeError("worker failed")
            return frames - 1

        report = run_audio_benchmark(
            AudioBenchmarkConfig(
                sample_rate=8000, chunk_frames=80, total_frames=160,
                maximum_dropout_ratio=1,
            ),
            process,
            clock=clock,
        )
        self.assertEqual(report.failures, 1)
        self.assertEqual(report.processed_frames, 79)
        self.assertEqual(report.dropped_frames, 81)
        self.assertFalse(report.passed)

    def test_invalid_callback_frame_count_is_a_failure(self):
        report = run_audio_benchmark(
            AudioBenchmarkConfig(sample_rate=8000, chunk_frames=80, total_frames=80),
            lambda sequence, samples, frames, final: frames + 1,
            clock=lambda: 0.0,
        )
        self.assertEqual(report.failures, 1)
        self.assertFalse(report.passed)


if __name__ == "__main__":
    unittest.main()
