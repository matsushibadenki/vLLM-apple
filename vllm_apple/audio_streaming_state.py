"""Bounded session state for streaming audio inference."""
from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from .audio_preprocessing import (
    AudioFeatureFrame,
    StreamingLinearResampler,
    StreamingLogBandEncoder,
)

_STATE_NAME = re.compile(r"[a-z][a-z0-9_.-]{0,63}\Z")


@dataclass(frozen=True, slots=True)
class AudioStreamUpdate:
    session_id: str
    sequence: int
    input_frames: int
    resampled_frames: int
    total_input_frames: int
    total_resampled_frames: int
    features: tuple[AudioFeatureFrame, ...]
    final: bool


@dataclass(frozen=True, slots=True)
class AudioStreamSnapshot:
    session_id: str
    next_sequence: int
    total_input_frames: int
    total_resampled_frames: int
    retained_feature_samples: int
    recurrent_state_bytes: int
    recurrent_state_names: tuple[str, ...]
    closed: bool


class StreamingAudioSession:
    """Owns all cross-chunk state for one ordered audio stream."""

    def __init__(
        self,
        session_id: str,
        *,
        input_rate: int,
        model_rate: int,
        channels: int = 1,
        frame_length: int = 400,
        hop_length: int = 160,
        feature_bands: int = 20,
        maximum_input_frames: int = 384_000 * 60 * 60,
        maximum_recurrent_state_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if (
            not _valid_session_id(session_id)
            or type(maximum_input_frames) is not int
            or maximum_input_frames <= 0
            or type(maximum_recurrent_state_bytes) is not int
            or not 1 <= maximum_recurrent_state_bytes <= 1 << 30
        ):
            raise ValueError("invalid streaming audio session configuration")
        self.session_id = session_id
        self._resampler = StreamingLinearResampler(
            input_rate, model_rate, channels=channels
        )
        self._encoder = StreamingLogBandEncoder(
            frame_length=frame_length,
            hop_length=hop_length,
            bands=feature_bands,
        )
        self._channels = channels
        self._maximum_input_frames = maximum_input_frames
        self._maximum_recurrent_state_bytes = maximum_recurrent_state_bytes
        self._next_sequence = 0
        self._total_input_frames = 0
        self._total_resampled_frames = 0
        self._states: dict[str, bytes] = {}
        self._state_bytes = 0
        self._closed = False

    def ingest(
        self,
        sequence: int,
        samples: Sequence[float],
        frame_count: int,
        *,
        final: bool = False,
    ) -> AudioStreamUpdate:
        if self._closed:
            raise RuntimeError("streaming audio session is closed")
        if type(sequence) is not int or sequence != self._next_sequence:
            raise ValueError(f"expected audio chunk sequence {self._next_sequence}")
        if (
            type(frame_count) is not int
            or frame_count < 0
            or self._total_input_frames + frame_count > self._maximum_input_frames
        ):
            raise ValueError("streaming audio input exceeds its frame budget")
        resampled = self._resampler.process(samples, frame_count, final=final)
        resampled_frames = len(resampled) // self._channels
        mono = _downmix(resampled, self._channels)
        features = self._encoder.process(mono)
        self._next_sequence += 1
        self._total_input_frames += frame_count
        self._total_resampled_frames += resampled_frames
        if final:
            self._closed = True
        return AudioStreamUpdate(
            session_id=self.session_id,
            sequence=sequence,
            input_frames=frame_count,
            resampled_frames=resampled_frames,
            total_input_frames=self._total_input_frames,
            total_resampled_frames=self._total_resampled_frames,
            features=features,
            final=final,
        )

    def set_recurrent_state(self, name: str, payload: bytes) -> None:
        if self._closed:
            raise RuntimeError("streaming audio session is closed")
        if not isinstance(name, str) or not _STATE_NAME.fullmatch(name):
            raise ValueError("invalid recurrent state name")
        if not isinstance(payload, bytes):
            raise ValueError("recurrent state payload must be immutable bytes")
        previous = self._states.get(name, b"")
        resulting_size = self._state_bytes - len(previous) + len(payload)
        if resulting_size > self._maximum_recurrent_state_bytes:
            raise ValueError("recurrent state exceeds its byte budget")
        self._states[name] = payload
        self._state_bytes = resulting_size

    def recurrent_state(self, name: str) -> bytes | None:
        return self._states.get(name)

    @property
    def snapshot(self) -> AudioStreamSnapshot:
        return AudioStreamSnapshot(
            session_id=self.session_id,
            next_sequence=self._next_sequence,
            total_input_frames=self._total_input_frames,
            total_resampled_frames=self._total_resampled_frames,
            retained_feature_samples=self._encoder.retained_samples,
            recurrent_state_bytes=self._state_bytes,
            recurrent_state_names=tuple(sorted(self._states)),
            closed=self._closed,
        )

    def close(self) -> None:
        self._closed = True
        self._resampler.reset()
        self._encoder.reset()
        self._states.clear()
        self._state_bytes = 0


@dataclass(slots=True)
class _RegistryEntry:
    session: StreamingAudioSession
    touched_at: float


class StreamingAudioSessionRegistry:
    """Bounded registry with explicit close and idle-session reaping."""

    def __init__(
        self,
        *,
        maximum_sessions: int = 32,
        idle_timeout_seconds: float = 300,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            type(maximum_sessions) is not int
            or not 1 <= maximum_sessions <= 1024
            or not math.isfinite(idle_timeout_seconds)
            or idle_timeout_seconds <= 0
        ):
            raise ValueError("invalid streaming audio registry configuration")
        self._maximum_sessions = maximum_sessions
        self._idle_timeout_seconds = idle_timeout_seconds
        self._clock = clock
        self._entries: dict[str, _RegistryEntry] = {}
        self._lock = threading.RLock()

    def create(self, session_id: str, **configuration: object) -> StreamingAudioSession:
        with self._lock:
            self._reap_locked(self._clock())
            if session_id in self._entries:
                raise ValueError("streaming audio session already exists")
            if len(self._entries) >= self._maximum_sessions:
                raise RuntimeError("streaming audio session capacity reached")
            session = StreamingAudioSession(session_id, **configuration)
            self._entries[session_id] = _RegistryEntry(session, self._clock())
            return session

    def get(self, session_id: str) -> StreamingAudioSession:
        with self._lock:
            entry = self._entries.get(session_id)
            if entry is None:
                raise KeyError("streaming audio session not found")
            entry.touched_at = self._clock()
            return entry.session

    def close(self, session_id: str) -> bool:
        with self._lock:
            entry = self._entries.pop(session_id, None)
        if entry is None:
            return False
        entry.session.close()
        return True

    def reap_idle(self) -> tuple[str, ...]:
        with self._lock:
            return self._reap_locked(self._clock())

    def _reap_locked(self, now: float) -> tuple[str, ...]:
        expired = tuple(sorted(
            session_id
            for session_id, entry in self._entries.items()
            if now - entry.touched_at >= self._idle_timeout_seconds
        ))
        for session_id in expired:
            self._entries.pop(session_id).session.close()
        return expired


def _downmix(samples: tuple[float, ...], channels: int) -> tuple[float, ...]:
    if channels == 1:
        return samples
    return tuple(
        sum(samples[offset:offset + channels]) / channels
        for offset in range(0, len(samples), channels)
    )


def _valid_session_id(value: str) -> bool:
    return isinstance(value, str) and 1 <= len(value) <= 128 and value.isascii()
