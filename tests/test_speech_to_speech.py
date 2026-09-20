import unittest

from vllm_apple.asr_integration import ASRTranscript
from vllm_apple.speech_to_speech import (
    DialogueResponse,
    EchoDialogueBackend,
    SpeechToSpeechPipeline,
    SynthesizedSpeech,
)


class FixtureSynthesizer:
    def __init__(self, *, language="en", frames=16000):
        self.language = language
        self.frames = frames

    def synthesize(self, response):
        return SynthesizedSpeech(
            b"\0\0" * self.frames, 16000, 1, self.language
        )


def transcript(*, final=True, language="en"):
    return ASRTranscript("hello", language, 0, 1, 0.9, final)


class SpeechToSpeechTests(unittest.TestCase):
    def test_final_transcript_produces_validated_audio_and_rtf(self):
        pipeline = SpeechToSpeechPipeline(EchoDialogueBackend("reply: "), FixtureSynthesizer())
        result = pipeline.respond(transcript(), elapsed_seconds=0.25)
        self.assertEqual(result.response.text, "reply: hello")
        self.assertEqual(result.speech.duration_seconds, 1.0)
        self.assertEqual(result.realtime_factor, 0.25)

    def test_partial_transcript_never_triggers_backends(self):
        calls = []

        class Dialogue:
            def respond(self, value):
                calls.append(value)

        pipeline = SpeechToSpeechPipeline(Dialogue(), FixtureSynthesizer())
        with self.assertRaisesRegex(ValueError, "final ASR"):
            pipeline.respond(transcript(final=False), elapsed_seconds=0.1)
        self.assertEqual(calls, [])

    def test_language_mismatch_and_output_budgets_fail_closed(self):
        mismatch = SpeechToSpeechPipeline(EchoDialogueBackend(), FixtureSynthesizer(language="ja"))
        with self.assertRaisesRegex(ValueError, "language"):
            mismatch.respond(transcript(), elapsed_seconds=0.1)
        bounded = SpeechToSpeechPipeline(
            EchoDialogueBackend(), FixtureSynthesizer(frames=10), maximum_output_bytes=4
        )
        with self.assertRaisesRegex(ValueError, "byte budget"):
            bounded.respond(transcript(), elapsed_seconds=0.1)

    def test_pcm_must_contain_complete_interleaved_frames(self):
        with self.assertRaisesRegex(ValueError, "synthesized speech"):
            SynthesizedSpeech(b"\0\0\0", 16000, 1, "en")
        with self.assertRaisesRegex(ValueError, "dialogue"):
            DialogueResponse("", "en")


if __name__ == "__main__":
    unittest.main()
