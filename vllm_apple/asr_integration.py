"""Backend-neutral streaming ASR integration over audio sessions and deadlines."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from .audio_deadline_scheduler import (
    AudioDeadlineScheduler,
    AudioSchedulingPriority,
    AudioTaskOutcome,
)
from .audio_preprocessing import AudioFeatureFrame
from .audio_streaming_state import (
    AudioStreamUpdate,
    StreamingAudioSession,
    StreamingAudioSessionRegistry,
)


@dataclass(frozen=True, slots=True)
class ASRWorkItem:
    session_id: str
    sequence: int
    features: tuple[AudioFeatureFrame, ...]
    final: bool


@dataclass(frozen=True, slots=True)
class ASRTranscript:
    text: str
    language: str
    start_seconds: float
    end_seconds: float
    confidence: float | None
    final: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.text, str)
            or len(self.text) > 32_768
            or not isinstance(self.language, str)
            or not 1 <= len(self.language) <= 35
            or not self.language.isascii()
            or not math.isfinite(self.start_seconds)
            or not math.isfinite(self.end_seconds)
            or self.start_seconds < 0
            or self.end_seconds < self.start_seconds
            or self.confidence is not None
            and (not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1)
        ):
            raise ValueError("invalid ASR transcript")


class ASRBackend(Protocol):
    def transcribe(self, work: ASRWorkItem) -> ASRTranscript: ...


@dataclass(frozen=True, slots=True)
class ASRSubmission:
    update: AudioStreamUpdate
    task_id: str | None
    admitted: bool


class StreamingASRIntegrator:
    """Connect ordered PCM chunks to a bounded feature/deadline/backend pipeline."""

    def __init__(
        self,
        backend: ASRBackend,
        *,
        sessions: StreamingAudioSessionRegistry | None = None,
        scheduler: AudioDeadlineScheduler[ASRWorkItem] | None = None,
    ) -> None:
        if not callable(getattr(backend, "transcribe", None)):
            raise ValueError("ASR backend must implement transcribe")
        self._backend = backend
        self._sessions = sessions or StreamingAudioSessionRegistry()
        self._scheduler = scheduler or AudioDeadlineScheduler()

    def open_session(self, session_id: str, **configuration: object) -> StreamingAudioSession:
        return self._sessions.create(session_id, **configuration)

    def submit_audio(
        self,
        session_id: str,
        sequence: int,
        samples: Sequence[float],
        frame_count: int,
        *,
        deadline: float,
        estimated_duration_seconds: float,
        final: bool = False,
        priority: AudioSchedulingPriority = AudioSchedulingPriority.REALTIME,
    ) -> ASRSubmission:
        session = self._sessions.get(session_id)
        update = session.ingest(sequence, samples, frame_count, final=final)
        if not update.features and not final:
            return ASRSubmission(update, None, True)
        task_id = f"{session_id}:{sequence}"
        from .audio_deadline_scheduler import AudioScheduledTask

        admitted = self._scheduler.submit(AudioScheduledTask(
            task_id=task_id,
            priority=priority,
            deadline=deadline,
            estimated_duration_seconds=estimated_duration_seconds,
            payload=ASRWorkItem(session_id, sequence, update.features, final),
        ))
        return ASRSubmission(update, task_id, admitted)

    def run_next(self) -> AudioTaskOutcome[ASRTranscript] | None:
        return self._scheduler.run_next(self._transcribe_validated)

    def close_session(self, session_id: str) -> bool:
        return self._sessions.close(session_id)

    def _transcribe_validated(self, work: ASRWorkItem) -> ASRTranscript:
        transcript = self._backend.transcribe(work)
        if not isinstance(transcript, ASRTranscript):
            raise TypeError("ASR backend returned an invalid transcript type")
        if transcript.final != work.final:
            raise ValueError("ASR transcript final marker does not match its work item")
        return transcript


@dataclass(frozen=True, slots=True)
class CallableASRBackend:
    """Small adapter for local model workers without imposing a package dependency."""

    callback: Callable[[ASRWorkItem], ASRTranscript]

    def transcribe(self, work: ASRWorkItem) -> ASRTranscript:
        return self.callback(work)
