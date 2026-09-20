"""Bounded, revision-bound cache for reusable vision encoder outputs."""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class VisionCacheKey:
    image_sha256: str
    model_revision: str
    preprocessing_fingerprint: str
    encoder_fingerprint: str

    def __post_init__(self) -> None:
        values = (
            self.image_sha256,
            self.model_revision,
            self.preprocessing_fingerprint,
            self.encoder_fingerprint,
        )
        if any(not _is_digest(value) for value in values):
            raise ValueError("vision cache key fields must be SHA-256 digests")

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {
                "image": self.image_sha256,
                "model": self.model_revision,
                "preprocessing": self.preprocessing_fingerprint,
                "encoder": self.encoder_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class VisionCacheStats:
    entries: int
    resident_bytes: int
    hits: int
    misses: int
    evictions: int


@dataclass(slots=True)
class _CacheEntry(Generic[T]):
    value: T
    size_bytes: int


class VisionEncoderCache(Generic[T]):
    """Thread-safe LRU that never admits an output larger than its byte budget."""

    def __init__(self, *, maximum_entries: int = 128, maximum_bytes: int = 512 << 20) -> None:
        if not 1 <= maximum_entries <= 4096 or not 1 <= maximum_bytes <= 1 << 40:
            raise ValueError("invalid vision encoder cache limits")
        self._maximum_entries = maximum_entries
        self._maximum_bytes = maximum_bytes
        self._entries: OrderedDict[str, _CacheEntry[T]] = OrderedDict()
        self._resident_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = threading.RLock()

    def get(self, key: VisionCacheKey) -> T | None:
        with self._lock:
            entry = self._entries.get(key.digest)
            if entry is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key.digest)
            self._hits += 1
            return entry.value

    def put(self, key: VisionCacheKey, value: T, *, size_bytes: int) -> bool:
        if not 1 <= size_bytes <= self._maximum_bytes:
            return False
        with self._lock:
            previous = self._entries.pop(key.digest, None)
            if previous is not None:
                self._resident_bytes -= previous.size_bytes
            self._entries[key.digest] = _CacheEntry(value, size_bytes)
            self._resident_bytes += size_bytes
            self._evict_to_limits()
            return key.digest in self._entries

    def get_or_compute(
        self,
        key: VisionCacheKey,
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
    def stats(self) -> VisionCacheStats:
        with self._lock:
            return VisionCacheStats(
                len(self._entries),
                self._resident_bytes,
                self._hits,
                self._misses,
                self._evictions,
            )

    def _evict_to_limits(self) -> None:
        while (
            len(self._entries) > self._maximum_entries
            or self._resident_bytes > self._maximum_bytes
        ):
            _, removed = self._entries.popitem(last=False)
            self._resident_bytes -= removed.size_bytes
            self._evictions += 1


def preprocessing_fingerprint(spec: object) -> str:
    if not hasattr(spec, "__dataclass_fields__"):
        raise ValueError("preprocessing specification must be a dataclass")
    values = {
        name: getattr(spec, name)
        for name in sorted(spec.__dataclass_fields__)
    }
    payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _is_digest(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
