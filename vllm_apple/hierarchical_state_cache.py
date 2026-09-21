"""Private hot/cold cache for KV, prefix and multimodal embedding state."""
from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

MAX_HIERARCHICAL_CACHE_ENTRIES = 65_536
MAX_HIERARCHICAL_CACHE_BYTES = 1 << 40


class HierarchicalStateKind(str, Enum):
    KV = "kv"
    PREFIX = "prefix"
    VISION_EMBEDDING = "vision_embedding"
    VIDEO_EMBEDDING = "video_embedding"
    AUDIO_EMBEDDING = "audio_embedding"


@dataclass(frozen=True, slots=True)
class HierarchicalStateKey:
    kind: HierarchicalStateKind
    fingerprint: str

    def __post_init__(self) -> None:
        if (not isinstance(self.kind, HierarchicalStateKind)
                or len(self.fingerprint) != 64
                or any(character not in "0123456789abcdef"
                       for character in self.fingerprint)):
            raise ValueError("invalid hierarchical state key")

    @property
    def cache_id(self) -> str:
        return hashlib.sha256(
            f"{self.kind.value}:{self.fingerprint}".encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class HierarchicalStateValue:
    payload: bytes
    tier: str
    sha256: str


@dataclass(slots=True)
class _ColdEntry:
    path: Path
    size: int
    sha256: str


class HierarchicalStateCache:
    def __init__(
        self,
        root: Path,
        *,
        hot_entries: int,
        hot_bytes: int,
        cold_entries: int,
        cold_bytes: int,
    ) -> None:
        bounds = (hot_entries, hot_bytes, cold_entries, cold_bytes)
        if (any(type(value) is not int or value < 0 for value in bounds)
                or hot_entries > MAX_HIERARCHICAL_CACHE_ENTRIES
                or cold_entries > MAX_HIERARCHICAL_CACHE_ENTRIES
                or hot_bytes > MAX_HIERARCHICAL_CACHE_BYTES
                or cold_bytes > MAX_HIERARCHICAL_CACHE_BYTES
                or (hot_entries == 0) != (hot_bytes == 0)
                or (cold_entries == 0) != (cold_bytes == 0)
                or hot_entries + cold_entries == 0):
            raise ValueError("invalid hierarchical cache bounds")
        self.root = root.expanduser().resolve(strict=False)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        attributes = self.root.lstat()
        if (not stat.S_ISDIR(attributes.st_mode) or attributes.st_uid != os.getuid()
                or attributes.st_mode & 0o077):
            raise ValueError("hierarchical cache root must be private")
        self._hot_limit = (hot_entries, hot_bytes)
        self._cold_limit = (cold_entries, cold_bytes)
        self._hot: OrderedDict[HierarchicalStateKey, bytes] = OrderedDict()
        self._cold: OrderedDict[HierarchicalStateKey, _ColdEntry] = OrderedDict()
        self._hot_bytes = self._cold_bytes = 0
        self._hits_hot = self._hits_cold = self._misses = 0
        self._promotions = self._spills = self._evictions = 0
        self._lock = threading.RLock()

    def put(self, key: HierarchicalStateKey, payload: bytes) -> str:
        if (not isinstance(key, HierarchicalStateKey) or not isinstance(payload, bytes)
                or not 1 <= len(payload) <= MAX_HIERARCHICAL_CACHE_BYTES):
            raise ValueError("invalid hierarchical cache value")
        with self._lock:
            self._remove_locked(key)
            if self._fits_hot(payload):
                self._hot[key] = payload
                self._hot_bytes += len(payload)
                self._trim_hot()
                return "hot"
            if self._store_cold(key, payload):
                return "cold"
            self._evictions += 1
            return "dropped"

    def get(self, key: HierarchicalStateKey) -> HierarchicalStateValue | None:
        if not isinstance(key, HierarchicalStateKey):
            raise ValueError("invalid hierarchical cache key")
        with self._lock:
            payload = self._hot.get(key)
            if payload is not None:
                self._hot.move_to_end(key)
                self._hits_hot += 1
                return HierarchicalStateValue(
                    payload, "hot", hashlib.sha256(payload).hexdigest()
                )
            entry = self._cold.get(key)
            if entry is None:
                self._misses += 1
                return None
            try:
                payload = self._read_cold(entry)
            except ValueError:
                self._delete_cold(key)
                self._evictions += 1
                raise
            self._cold.move_to_end(key)
            self._hits_cold += 1
            if self._fits_hot(payload):
                self._delete_cold(key)
                self._hot[key] = payload
                self._hot_bytes += len(payload)
                self._promotions += 1
                self._trim_hot(exclude=key)
                tier = "hot"
            else:
                tier = "cold"
            return HierarchicalStateValue(payload, tier, entry.sha256)

    def remove(self, key: HierarchicalStateKey) -> bool:
        with self._lock:
            return self._remove_locked(key)

    def clear(self) -> None:
        with self._lock:
            self._hot.clear()
            self._hot_bytes = 0
            for key in tuple(self._cold):
                self._delete_cold(key)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "hot_entries": len(self._hot), "hot_bytes": self._hot_bytes,
                "cold_entries": len(self._cold), "cold_bytes": self._cold_bytes,
                "hot_hits": self._hits_hot, "cold_hits": self._hits_cold,
                "misses": self._misses, "promotions": self._promotions,
                "spills": self._spills, "evictions": self._evictions,
            }

    def _fits_hot(self, payload: bytes) -> bool:
        return bool(self._hot_limit[0] and len(payload) <= self._hot_limit[1])

    def _trim_hot(self, exclude: HierarchicalStateKey | None = None) -> None:
        while (len(self._hot) > self._hot_limit[0]
               or self._hot_bytes > self._hot_limit[1]):
            key = next(candidate for candidate in self._hot if candidate != exclude)
            payload = self._hot.pop(key)
            self._hot_bytes -= len(payload)
            if self._store_cold(key, payload):
                self._spills += 1
            else:
                self._evictions += 1

    def _store_cold(self, key: HierarchicalStateKey, payload: bytes) -> bool:
        if not self._cold_limit[0] or len(payload) > self._cold_limit[1]:
            return False
        digest = hashlib.sha256(payload).hexdigest()
        descriptor, temporary_name = tempfile.mkstemp(prefix=".state-", dir=self.root)
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            destination = self.root / f"{key.cache_id}-{digest}.bin"
            os.replace(temporary, destination)
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            temporary.unlink(missing_ok=True)
            raise
        self._cold[key] = _ColdEntry(destination, len(payload), digest)
        self._cold_bytes += len(payload)
        while (len(self._cold) > self._cold_limit[0]
               or self._cold_bytes > self._cold_limit[1]):
            victim = next(iter(self._cold))
            self._delete_cold(victim)
            self._evictions += 1
        return key in self._cold

    def _read_cold(self, entry: _ColdEntry) -> bytes:
        descriptor = os.open(entry.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            attributes = os.fstat(descriptor)
            if (not stat.S_ISREG(attributes.st_mode) or attributes.st_uid != os.getuid()
                    or attributes.st_mode & 0o077 or attributes.st_size != entry.size):
                raise ValueError("cold state cache entry is unsafe")
            chunks = []
            remaining = entry.size
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
        finally:
            os.close(descriptor)
        if len(payload) != entry.size or hashlib.sha256(payload).hexdigest() != entry.sha256:
            raise ValueError("cold state cache digest mismatch")
        return payload

    def _delete_cold(self, key: HierarchicalStateKey) -> None:
        entry = self._cold.pop(key)
        self._cold_bytes -= entry.size
        entry.path.unlink(missing_ok=True)

    def _remove_locked(self, key: HierarchicalStateKey) -> bool:
        payload = self._hot.pop(key, None)
        if payload is not None:
            self._hot_bytes -= len(payload)
            return True
        if key in self._cold:
            self._delete_cold(key)
            return True
        return False
