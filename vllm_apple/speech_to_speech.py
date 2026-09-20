"""Backend-neutral ASR-to-dialogue-to-speech pipeline foundation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .asr_integration import ASRTranscript


@dataclass(frozen=True, slots=True)
class DialogueResponse:
    text: str
    language: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.text, str)
            or not self.text.strip()
            or len(self.text) > 32_768
            or not isinstance(self.language, str)
            or not 1 <= len(self.language) <= 35
            or not self.language.isascii()
        ):
            raise ValueError("invalid dialogue response")


@dataclass(frozen=True, slots=True)
class SynthesizedSpeech:
    pcm_s16le: bytes
    sample_rate: int
    channels: int
    language: str

    def __post_init__(self) -> None:
        frame_bytes = self.channels * 2 if type(self.channels) is int else 0
        if (
            not isinstance(self.pcm_s16le, bytes)
            or not self.pcm_s16le
            or type(self.sample_rate) is not int
            or not 8_000 <= self.sample_rate <= 192_000
            or type(self.channels) is not int
            or not 1 <= self.channels <= 8
            or len(self.pcm_s16le) % frame_bytes
            or not isinstance(self.language, str)
            or not 1 <= len(self.language) <= 35
            or not self.language.isascii()
        ):
            raise ValueError("invalid synthesized speech")

    @property
    def frame_count(self) -> int:
        return len(self.pcm_s16le) // (self.channels * 2)

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.sample_rate


class DialogueBackend(Protocol):
    def respond(self, transcript: ASRTranscript) -> DialogueResponse: ...


class SpeechSynthesisBackend(Protocol):
    def synthesize(self, response: DialogueResponse) -> SynthesizedSpeech: ...


@dataclass(frozen=True, slots=True)
class SpeechToSpeechResult:
    transcript: ASRTranscript
    response: DialogueResponse
    speech: SynthesizedSpeech
    end_to_end_latency_seconds: float
    realtime_factor: float


class SpeechToSpeechPipeline:
    """Validated final-transcript pipeline; partial ASR never triggers audible output."""

    def __init__(
        self,
        dialogue: DialogueBackend,
        synthesizer: SpeechSynthesisBackend,
        *,
        maximum_output_bytes: int = 32 * 1024 * 1024,
        maximum_duration_seconds: float = 300,
    ) -> None:
        if (
            not callable(getattr(dialogue, "respond", None))
            or not callable(getattr(synthesizer, "synthesize", None))
            or type(maximum_output_bytes) is not int
            or not 1 <= maximum_output_bytes <= 1 << 30
            or not math.isfinite(maximum_duration_seconds)
            or maximum_duration_seconds <= 0
        ):
            raise ValueError("invalid speech-to-speech pipeline configuration")
        self._dialogue = dialogue
        self._synthesizer = synthesizer
        self._maximum_output_bytes = maximum_output_bytes
        self._maximum_duration_seconds = maximum_duration_seconds

    def respond(
        self,
        transcript: ASRTranscript,
        *,
        elapsed_seconds: float,
    ) -> SpeechToSpeechResult:
        if not transcript.final:
            raise ValueError("speech-to-speech requires a final ASR transcript")
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("speech-to-speech elapsed time is invalid")
        response = self._dialogue.respond(transcript)
        if not isinstance(response, DialogueResponse):
            raise TypeError("dialogue backend returned an invalid response type")
        speech = self._synthesizer.synthesize(response)
        if not isinstance(speech, SynthesizedSpeech):
            raise TypeError("speech backend returned an invalid result type")
        if response.language != speech.language:
            raise ValueError("speech language does not match dialogue response")
        if len(speech.pcm_s16le) > self._maximum_output_bytes:
            raise ValueError("synthesized speech exceeds its byte budget")
        duration = speech.duration_seconds
        if duration > self._maximum_duration_seconds:
            raise ValueError("synthesized speech exceeds its duration budget")
        realtime_factor = elapsed_seconds / duration
        return SpeechToSpeechResult(
            transcript,
            response,
            speech,
            elapsed_seconds,
            realtime_factor,
        )


@dataclass(frozen=True, slots=True)
class EchoDialogueBackend:
    """Deterministic local fixture adapter, not a production dialogue model."""

    prefix: str = ""

    def respond(self, transcript: ASRTranscript) -> DialogueResponse:
        return DialogueResponse(self.prefix + transcript.text, transcript.language)
