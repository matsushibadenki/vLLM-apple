"""Partitioned bounded cache for frame, patch, embedding, and scene artifacts."""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Mapping, TypeVar


T = TypeVar("T")


class VideoCacheKind(str, Enum):
    FRAME = "frame"
    PATCH = "patch"
    EMBEDDING = "embedding"
    SCENE = "scene"


@dataclass(frozen=True, slots=True)
class VideoCacheKey:
    kind: VideoCacheKind
    video_sha256: str
    transform_fingerprint: str
    model_fingerprint: str
    start_microseconds: int
    end_microseconds: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.kind, VideoCacheKind)
            or any(not _is_digest(value) for value in (
                self.video_sha256,
                self.transform_fingerprint,
                self.model_fingerprint,
            ))
            or type(self.start_microseconds) is not int
            or type(self.end_microseconds) is not int
            or self.start_microseconds < 0
            or self.end_microseconds <= self.start_microseconds
        ):
            raise ValueError("invalid video cache key")

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {
                "end": self.end_microseconds,
                "kind": self.kind.value,
                "model": self.model_fingerprint,
                "start": self.start_microseconds,
                "transform": self.transform_fingerprint,
                "video": self.video_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class VideoCacheTierSnapshot:
    kind: VideoCacheKind
    entries: int
    resident_bytes: int
    hits: int
    misses: int
    evictions: int


@dataclass(frozen=True, slots=True)
class VideoCacheSnapshot:
    entries: int
    resident_bytes: int
    rejected_oversize: int
    tiers: tuple[VideoCacheTierSnapshot, ...]


@dataclass(slots=True)
class _Entry(Generic[T]):
    key: VideoCacheKey
    value: T
    size_bytes: int


class VideoArtifactCache(Generic[T]):
    """Thread-safe global LRU with independent per-artifact byte budgets."""

    def __init__(
        self,
        *,
        maximum_entries: int = 512,
        maximum_bytes: int = 1024 << 20,
        tier_maximum_bytes: Mapping[VideoCacheKind, int] | None = None,
    ) -> None:
        defaults = {kind: maximum_bytes for kind in VideoCacheKind}
        if tier_maximum_bytes is not None:
            defaults.update(tier_maximum_bytes)
        if (
            type(maximum_entries) is not int
            or not 1 <= maximum_entries <= 65_536
            or type(maximum_bytes) is not int
            or not 1 <= maximum_bytes <= 1 << 42
            or set(defaults) != set(VideoCacheKind)
            or any(type(value) is not int or not 1 <= value <= maximum_bytes for value in defaults.values())
        ):
            raise ValueError("invalid video cache limits")
        self._maximum_entries = maximum_entries
        self._maximum_bytes = maximum_bytes
        self._tier_maximum_bytes = defaults
        self._entries: OrderedDict[str, _Entry[T]] = OrderedDict()
        self._resident_bytes = 0
        self._tier_bytes = {kind: 0 for kind in VideoCacheKind}
        self._hits = {kind: 0 for kind in VideoCacheKind}
        self._misses = {kind: 0 for kind in VideoCacheKind}
        self._evictions = {kind: 0 for kind in VideoCacheKind}
        self._rejected_oversize = 0
        self._lock = threading.RLock()

    def get(self, key: VideoCacheKey) -> T | None:
        digest = key.digest
        with self._lock:
            entry = self._entries.get(digest)
            if entry is None:
                self._misses[key.kind] += 1
                return None
            self._entries.move_to_end(digest)
            self._hits[key.kind] += 1
            return entry.value

    def put(self, key: VideoCacheKey, value: T, *, size_bytes: int) -> bool:
        tier_limit = self._tier_maximum_bytes[key.kind]
        if type(size_bytes) is not int or not 1 <= size_bytes <= min(
            tier_limit, self._maximum_bytes
        ):
            with self._lock:
                self._rejected_oversize += 1
            return False
        digest = key.digest
        with self._lock:
            previous = self._entries.pop(digest, None)
            if previous is not None:
                self._remove_accounting(previous)
            self._entries[digest] = _Entry(key, value, size_bytes)
            self._resident_bytes += size_bytes
            self._tier_bytes[key.kind] += size_bytes
            self._evict_tier(key.kind)
            self._evict_global()
            return digest in self._entries

    def clear(self, kind: VideoCacheKind | None = None) -> None:
        with self._lock:
            if kind is None:
                self._entries.clear()
                self._resident_bytes = 0
                self._tier_bytes = {item: 0 for item in VideoCacheKind}
                return
            for digest in [
                digest for digest, entry in self._entries.items() if entry.key.kind is kind
            ]:
                self._remove_entry(digest, count_eviction=False)

    def _evict_tier(self, kind: VideoCacheKind) -> None:
        while self._tier_bytes[kind] > self._tier_maximum_bytes[kind]:
            digest = next(
                digest for digest, entry in self._entries.items() if entry.key.kind is kind
            )
            self._remove_entry(digest)

    def _evict_global(self) -> None:
        while (
            len(self._entries) > self._maximum_entries
            or self._resident_bytes > self._maximum_bytes
        ):
            digest = next(iter(self._entries))
            self._remove_entry(digest)

    def _remove_entry(self, digest: str, *, count_eviction: bool = True) -> None:
        entry = self._entries.pop(digest)
        self._remove_accounting(entry)
        if count_eviction:
            self._evictions[entry.key.kind] += 1

    def _remove_accounting(self, entry: _Entry[T]) -> None:
        self._resident_bytes -= entry.size_bytes
        self._tier_bytes[entry.key.kind] -= entry.size_bytes

    @property
    def snapshot(self) -> VideoCacheSnapshot:
        with self._lock:
            tiers = tuple(
                VideoCacheTierSnapshot(
                    kind,
                    sum(entry.key.kind is kind for entry in self._entries.values()),
                    self._tier_bytes[kind],
                    self._hits[kind],
                    self._misses[kind],
                    self._evictions[kind],
                )
                for kind in VideoCacheKind
            )
            return VideoCacheSnapshot(
                len(self._entries), self._resident_bytes, self._rejected_oversize, tiers
            )


def _is_digest(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
