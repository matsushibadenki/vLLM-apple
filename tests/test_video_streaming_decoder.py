import io
import unittest

from vllm_apple.video_frame_scheduler import VideoFrameScheduler
from vllm_apple.video_streaming_decoder import IncrementalFFmpegVideoDecoder


class FakeInput(io.BytesIO):
    def close(self):
        self.closed_by_decoder = True


class FakeProcess:
    def __init__(self, frames):
        self.stdin = FakeInput()
        self.stdout = io.BytesIO(frames)
        self.stderr = io.BytesIO(b"")
        self.exit_code = 0

    def wait(self, timeout=None):
        return self.exit_code

    def kill(self):
        self.exit_code = -9


class VideoStreamingDecoderTests(unittest.TestCase):
    def test_incremental_input_feeds_decoded_frames_to_scheduler(self):
        process = FakeProcess(bytes(range(32)))
        scheduler = VideoFrameScheduler[bytes](maximum_frames=4, maximum_lateness_seconds=1)
        decoder = IncrementalFFmpegVideoDecoder(
            scheduler, width=2, height=2, frame_rate=2,
            maximum_frames=2, process_factory=lambda *args, **kwargs: process,
        )
        decoder.append(0, b"compressed-")
        decoder.append(1, b"video", final=True)
        report = decoder.finish()
        decisions = decoder.drain_ready(0.5)
        self.assertTrue(report.passed)
        self.assertEqual(report.decoded_frames, 2)
        self.assertEqual([item.frame.frame_id for item in decisions], ["frame-0"])
        self.assertEqual(len(decisions[0].frame.payload), 16)

    def test_sequence_and_input_budget_fail_closed(self):
        decoder = IncrementalFFmpegVideoDecoder(
            VideoFrameScheduler(), width=2, height=2, frame_rate=1,
            maximum_input_bytes=2, maximum_frames=1,
            process_factory=lambda *args, **kwargs: FakeProcess(bytes(range(16))),
        )
        with self.assertRaisesRegex(ValueError, "sequence 0"):
            decoder.append(1, b"a")
        with self.assertRaisesRegex(ValueError, "budget"):
            decoder.append(0, b"abc")
        decoder.append(0, b"ok", final=True)
        self.assertTrue(decoder.finish().passed)

    def test_scheduler_backpressure_is_reported(self):
        scheduler = VideoFrameScheduler[bytes](maximum_frames=1)
        decoder = IncrementalFFmpegVideoDecoder(
            scheduler, width=2, height=2, frame_rate=1, maximum_frames=2,
            process_factory=lambda *args, **kwargs: FakeProcess(bytes(range(32))),
        )
        decoder.append(0, b"stream", final=True)
        report = decoder.finish()
        self.assertEqual(report.scheduler_rejections, 1)
        self.assertFalse(report.passed)


if __name__ == "__main__":
    unittest.main()
