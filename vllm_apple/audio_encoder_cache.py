"""Bounded revision- and segment-aware cache for audio encoder outputs."""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class AudioEncoderCacheKey:
    audio_sha256: str
    encoder_fingerprint: str
    feature_fingerprint: str
    sample_rate: int
    channels: int
    start_sample: int
    end_sample: int

    def __post_init__(self) -> None:
        if (
            any(not _is_digest(value) for value in (
                self.audio_sha256,
                self.encoder_fingerprint,
                self.feature_fingerprint,
            ))
            or type(self.sample_rate) is not int
            or not 1 <= self.sample_rate <= 384_000
            or type(self.channels) is not int
            or not 1 <= self.channels <= 32
            or type(self.start_sample) is not int
            or type(self.end_sample) is not int
            or self.start_sample < 0
            or self.end_sample <= self.start_sample
        ):
            raise ValueError("invalid audio encoder cache key")

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {
                "audio": self.audio_sha256,
                "channels": self.channels,
                "encoder": self.encoder_fingerprint,
                "end": self.end_sample,
                "feature": self.feature_fingerprint,
                "rate": self.sample_rate,
                "start": self.start_sample,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class AudioEncoderCacheSnapshot:
    entries: int
    resident_bytes: int
    hits: int
    misses: int
    evictions: int
    rejected_oversize: int


@dataclass(slots=True)
class _Entry(Generic[T]):
    value: T
    size_bytes: int


class AudioEncoderCache(Generic[T]):
    """Thread-safe LRU for immutable encoded audio segments."""

    def __init__(self, *, maximum_entries: int = 256, maximum_bytes: int = 512 << 20) -> None:
        if (
            type(maximum_entries) is not int
            or not 1 <= maximum_entries <= 8192
            or type(maximum_bytes) is not int
            or not 1 <= maximum_bytes <= 1 << 40
        ):
            raise ValueError("invalid audio encoder cache limits")
        self._maximum_entries = maximum_entries
        self._maximum_bytes = maximum_bytes
        self._entries: OrderedDict[str, _Entry[T]] = OrderedDict()
        self._resident_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._rejected_oversize = 0
        self._lock = threading.RLock()

    def get(self, key: AudioEncoderCacheKey) -> T | None:
        digest = key.digest
        with self._lock:
            entry = self._entries.get(digest)
            if entry is None:
                self._misses += 1
                return None
            self._entries.move_to_end(digest)
            self._hits += 1
            return entry.value

    def put(self, key: AudioEncoderCacheKey, value: T, *, size_bytes: int) -> bool:
        if type(size_bytes) is not int or not 1 <= size_bytes <= self._maximum_bytes:
            with self._lock:
                self._rejected_oversize += 1
            return False
        digest = key.digest
        with self._lock:
            previous = self._entries.pop(digest, None)
            if previous is not None:
                self._resident_bytes -= previous.size_bytes
            self._entries[digest] = _Entry(value, size_bytes)
            self._resident_bytes += size_bytes
            while (
                len(self._entries) > self._maximum_entries
                or self._resident_bytes > self._maximum_bytes
            ):
                _, removed = self._entries.popitem(last=False)
                self._resident_bytes -= removed.size_bytes
                self._evictions += 1
            return digest in self._entries

    def get_or_compute(
        self,
        key: AudioEncoderCacheKey,
        compute: Callable[[], tuple[T, int]],
    ) -> T:
        cached = self.get(key)
        if cached is not None:
            return cached
        value, size_bytes = compute()
        self.put(key, value, size_bytes=size_bytes)
        return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._resident_bytes = 0

    @property
    def snapshot(self) -> AudioEncoderCacheSnapshot:
        with self._lock:
            return AudioEncoderCacheSnapshot(
                entries=len(self._entries),
                resident_bytes=self._resident_bytes,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                rejected_oversize=self._rejected_oversize,
            )


def audio_feature_fingerprint(
    *,
    model_rate: int,
    frame_length: int,
    hop_length: int,
    feature_bands: int,
) -> str:
    values = (model_rate, frame_length, hop_length, feature_bands)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("invalid audio feature fingerprint input")
    payload = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _is_digest(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
