import inspect
import unittest

from vllm_apple.audio_ring_buffer import AudioRingBuffer


class AudioRingBufferTests(unittest.TestCase):
    def test_wraparound_preserves_interleaved_frame_order(self):
        buffer = AudioRingBuffer(3, channels=2)
        self.assertEqual(buffer.write_from([1, 2, 3, 4], 2), 2)
        output = [0.0] * 2
        self.assertEqual(buffer.read_into(output, 1), 1)
        self.assertEqual(output, [1.0, 2.0])
        self.assertEqual(buffer.write_from([5, 6, 7, 8], 2), 2)
        output = [0.0] * 6
        self.assertEqual(buffer.read_into(output, 3), 3)
        self.assertEqual(output, [3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    def test_overflow_rejects_new_frames_without_overwriting_unread_audio(self):
        buffer = AudioRingBuffer(2)
        self.assertEqual(buffer.write_from([1, 2, 3], 3), 2)
        output = [0.0] * 2
        self.assertEqual(buffer.read_into(output, 2), 2)
        self.assertEqual(output, [1.0, 2.0])
        self.assertEqual(buffer.snapshot.overflow_frames, 1)

    def test_underrun_is_nonblocking_and_observable(self):
        buffer = AudioRingBuffer(4)
        output = [9.0] * 3
        self.assertEqual(buffer.read_into(output, 3), 0)
        self.assertEqual(output, [9.0, 9.0, 9.0])
        self.assertEqual(buffer.snapshot.underrun_frames, 3)

    def test_invalid_and_nonfinite_input_fails_closed(self):
        buffer = AudioRingBuffer(2)
        with self.assertRaisesRegex(ValueError, "finite"):
            buffer.write_from([float("nan")], 1)
        with self.assertRaisesRegex(ValueError, "producer"):
            buffer.write_from([], 1)

    def test_realtime_methods_do_not_use_explicit_locks(self):
        source = inspect.getsource(AudioRingBuffer)
        self.assertNotIn("threading", source)
        self.assertNotIn("Lock(", source)


if __name__ == "__main__":
    unittest.main()
