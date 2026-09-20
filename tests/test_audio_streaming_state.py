import unittest

from vllm_apple.audio_streaming_state import (
    StreamingAudioSession,
    StreamingAudioSessionRegistry,
)


class AudioStreamingStateTests(unittest.TestCase):
    def session(self, **kwargs):
        return StreamingAudioSession(
            "session-1",
            input_rate=8,
            model_rate=8,
            frame_length=4,
            hop_length=2,
            feature_bands=2,
            **kwargs,
        )

    def test_tracks_ordered_chunks_features_and_final_state(self):
        session = self.session()
        first = session.ingest(0, [0, 1, 0], 3)
        second = session.ingest(1, [1, 0, 1], 3, final=True)
        self.assertEqual(first.features, ())
        self.assertEqual([item.start_sample for item in second.features], [0, 2])
        self.assertEqual(second.total_input_frames, 6)
        self.assertTrue(session.snapshot.closed)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.ingest(2, [], 0)

    def test_rejects_missing_sequence_and_input_budget(self):
        session = self.session(maximum_input_frames=2)
        with self.assertRaisesRegex(ValueError, "sequence 0"):
            session.ingest(1, [0], 1)
        with self.assertRaisesRegex(ValueError, "frame budget"):
            session.ingest(0, [0, 0, 0], 3)

    def test_recurrent_state_is_named_immutable_and_bounded(self):
        session = self.session(maximum_recurrent_state_bytes=4)
        session.set_recurrent_state("encoder.kv", b"1234")
        self.assertEqual(session.recurrent_state("encoder.kv"), b"1234")
        self.assertEqual(session.snapshot.recurrent_state_bytes, 4)
        with self.assertRaisesRegex(ValueError, "byte budget"):
            session.set_recurrent_state("codec", b"x")

    def test_registry_bounds_and_reaps_idle_sessions(self):
        now = [0.0]
        registry = StreamingAudioSessionRegistry(
            maximum_sessions=1, idle_timeout_seconds=5, clock=lambda: now[0]
        )
        first = registry.create("first", input_rate=8, model_rate=8)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            registry.create("second", input_rate=8, model_rate=8)
        now[0] = 5.0
        self.assertEqual(registry.reap_idle(), ("first",))
        self.assertTrue(first.snapshot.closed)
        registry.create("second", input_rate=8, model_rate=8)
        self.assertTrue(registry.close("second"))
        self.assertFalse(registry.close("second"))


if __name__ == "__main__":
    unittest.main()
