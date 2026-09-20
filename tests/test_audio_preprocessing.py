import unittest

from vllm_apple.audio_preprocessing import (
    StreamingLinearResampler,
    StreamingLogBandEncoder,
)


class AudioPreprocessingTests(unittest.TestCase):
    def test_resampler_is_chunk_boundary_stable(self):
        samples = [float(index) / 10 for index in range(12)]
        complete = StreamingLinearResampler(12, 8).process(samples, 12, final=True)
        chunked = StreamingLinearResampler(12, 8)
        output = (
            chunked.process(samples[:5], 5)
            + chunked.process(samples[5:9], 4)
            + chunked.process(samples[9:], 3, final=True)
        )
        self.assertEqual(output, complete)
        self.assertEqual(len(output), 8)

    def test_resampler_preserves_interleaved_channels(self):
        result = StreamingLinearResampler(2, 4, channels=2).process(
            [0, 10, 2, 12], 2, final=True
        )
        self.assertEqual(result, (0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 2.0, 12.0))

    def test_feature_encoder_is_chunk_boundary_stable_and_bounded(self):
        samples = [float((index % 7) - 3) / 3 for index in range(24)]
        complete_encoder = StreamingLogBandEncoder(frame_length=8, hop_length=4, bands=2)
        complete = complete_encoder.process(samples)
        chunked_encoder = StreamingLogBandEncoder(frame_length=8, hop_length=4, bands=2)
        chunked = (
            chunked_encoder.process(samples[:5])
            + chunked_encoder.process(samples[5:13])
            + chunked_encoder.process(samples[13:])
        )
        self.assertEqual(chunked, complete)
        self.assertLess(chunked_encoder.retained_samples, 8)
        self.assertEqual([frame.start_sample for frame in chunked], [0, 4, 8, 12, 16])

    def test_nonfinite_samples_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            StreamingLinearResampler(16_000, 16_000).process([float("inf")], 1)
        with self.assertRaisesRegex(ValueError, "finite"):
            StreamingLogBandEncoder().process([float("nan")])


if __name__ == "__main__":
    unittest.main()
