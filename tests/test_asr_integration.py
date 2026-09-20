import unittest

from vllm_apple.asr_integration import (
    ASRTranscript,
    CallableASRBackend,
    StreamingASRIntegrator,
)
from vllm_apple.audio_deadline_scheduler import AudioDeadlineScheduler
from vllm_apple.audio_streaming_state import StreamingAudioSessionRegistry


class ASRIntegrationTests(unittest.TestCase):
    def integrator(self, callback, now):
        return StreamingASRIntegrator(
            CallableASRBackend(callback),
            sessions=StreamingAudioSessionRegistry(clock=lambda: now[0]),
            scheduler=AudioDeadlineScheduler(clock=lambda: now[0]),
        )

    def test_streaming_features_reach_backend_through_deadline_scheduler(self):
        now = [0.0]
        seen = []

        def transcribe(work):
            seen.append(work)
            return ASRTranscript("hello", "en", 0, 0.5, 0.9, work.final)

        integrator = self.integrator(transcribe, now)
        integrator.open_session(
            "stream", input_rate=8, model_rate=8,
            frame_length=4, hop_length=2, feature_bands=2,
        )
        first = integrator.submit_audio(
            "stream", 0, [0, 1, 0], 3, deadline=1, estimated_duration_seconds=0.1
        )
        self.assertIsNone(first.task_id)
        second = integrator.submit_audio(
            "stream", 1, [1, 0, 1], 3, deadline=1,
            estimated_duration_seconds=0.1, final=True,
        )
        self.assertTrue(second.admitted)
        outcome = integrator.run_next()
        self.assertEqual(outcome.status, "completed")
        self.assertEqual(outcome.result.text, "hello")
        self.assertEqual(len(seen[0].features), 2)
        self.assertTrue(seen[0].final)

    def test_infeasible_asr_work_is_not_executed(self):
        now = [10.0]
        calls = []
        integrator = self.integrator(lambda work: calls.append(work), now)
        integrator.open_session("stream", input_rate=8, model_rate=8)
        submission = integrator.submit_audio(
            "stream", 0, [], 0, deadline=10.01,
            estimated_duration_seconds=0.1, final=True,
        )
        self.assertFalse(submission.admitted)
        self.assertIsNone(integrator.run_next())
        self.assertEqual(calls, [])

    def test_backend_result_must_match_final_marker(self):
        now = [0.0]
        integrator = self.integrator(
            lambda work: ASRTranscript("partial", "ja", 0, 0, None, False), now
        )
        integrator.open_session("stream", input_rate=8, model_rate=8)
        integrator.submit_audio(
            "stream", 0, [], 0, deadline=1,
            estimated_duration_seconds=0.1, final=True,
        )
        with self.assertRaisesRegex(ValueError, "final marker"):
            integrator.run_next()

    def test_transcript_validation_is_bounded(self):
        with self.assertRaisesRegex(ValueError, "invalid ASR"):
            ASRTranscript("text", "en", 1, 0, 0.5, True)
        with self.assertRaisesRegex(ValueError, "invalid ASR"):
            ASRTranscript("text", "en", 0, 1, 1.1, True)


if __name__ == "__main__":
    unittest.main()
