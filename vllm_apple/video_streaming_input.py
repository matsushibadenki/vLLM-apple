"""Bounded ordered streaming-video ingestion into private disk-backed artifacts."""
from __future__ import annotations

import hashlib
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


@dataclass(frozen=True, slots=True)
class StreamingVideoUpdate:
    session_id: str
    sequence: int
    chunk_bytes: int
    total_bytes: int
    chunks: int
    final: bool


@dataclass(frozen=True, slots=True)
class FinalizedVideoArtifact:
    session_id: str
    path: Path
    size_bytes: int
    sha256: str
    chunks: int


class StreamingVideoInputSession:
    """Append-only ordered stream that exposes an artifact only after final integrity checks."""

    def __init__(
        self,
        session_id: str,
        workspace: Path,
        *,
        maximum_chunk_bytes: int = 4 * 1024 * 1024,
        maximum_total_bytes: int = 512 * 1024 * 1024,
        maximum_chunks: int = 65_536,
    ) -> None:
        if (
            not isinstance(session_id, str)
            or not _SESSION_ID.fullmatch(session_id)
            or not isinstance(workspace, Path)
            or workspace.is_symlink()
            or not workspace.is_dir()
            or type(maximum_chunk_bytes) is not int
            or not 1 <= maximum_chunk_bytes <= 64 * 1024 * 1024
            or type(maximum_total_bytes) is not int
            or not maximum_chunk_bytes <= maximum_total_bytes <= 1 << 40
            or type(maximum_chunks) is not int
            or not 1 <= maximum_chunks <= 1_000_000
        ):
            raise ValueError("invalid streaming video session configuration")
        self.session_id = session_id
        self._maximum_chunk_bytes = maximum_chunk_bytes
        self._maximum_total_bytes = maximum_total_bytes
        self._maximum_chunks = maximum_chunks
        self._path = workspace.resolve() / f".{session_id}.video-stream"
        descriptor = os.open(
            self._path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        self._stream = os.fdopen(descriptor, "wb")
        self._digest = hashlib.sha256()
        self._next_sequence = 0
        self._total_bytes = 0
        self._chunks = 0
        self._finalized: FinalizedVideoArtifact | None = None
        self._closed = False

    def append(
        self,
        sequence: int,
        chunk: bytes,
        *,
        final: bool = False,
        expected_sha256: str | None = None,
    ) -> StreamingVideoUpdate:
        if self._closed or self._finalized is not None:
            raise RuntimeError("streaming video session is not writable")
        if type(sequence) is not int or sequence != self._next_sequence:
            raise ValueError(f"expected video chunk sequence {self._next_sequence}")
        if (
            not isinstance(chunk, bytes)
            or not 1 <= len(chunk) <= self._maximum_chunk_bytes
            or self._total_bytes + len(chunk) > self._maximum_total_bytes
            or self._chunks + 1 > self._maximum_chunks
        ):
            raise ValueError("streaming video chunk exceeds its budget")
        self._stream.write(chunk)
        self._digest.update(chunk)
        self._total_bytes += len(chunk)
        self._chunks += 1
        self._next_sequence += 1
        if final:
            actual = self._digest.hexdigest()
            if expected_sha256 is not None and expected_sha256 != actual:
                self.close()
                raise ValueError("streaming video digest mismatch")
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()
            self._finalized = FinalizedVideoArtifact(
                self.session_id, self._path, self._total_bytes, actual, self._chunks
            )
        return StreamingVideoUpdate(
            self.session_id,
            sequence,
            len(chunk),
            self._total_bytes,
            self._chunks,
            final,
        )

    @property
    def artifact(self) -> FinalizedVideoArtifact:
        if self._finalized is None:
            raise RuntimeError("streaming video artifact is not finalized")
        return self._finalized

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._stream.closed:
            self._stream.close()
        self._path.unlink(missing_ok=True)

    def __enter__(self) -> "StreamingVideoInputSession":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


@dataclass(slots=True)
class _Entry:
    session: StreamingVideoInputSession
    touched_at: float


class StreamingVideoInputRegistry:
    """Bounded session registry with idle artifact cleanup."""

    def __init__(
        self,
        *,
        maximum_sessions: int = 16,
        idle_timeout_seconds: float = 300,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            type(maximum_sessions) is not int
            or not 1 <= maximum_sessions <= 1024
            or not math.isfinite(idle_timeout_seconds)
            or idle_timeout_seconds <= 0
        ):
            raise ValueError("invalid streaming video registry configuration")
        self._maximum_sessions = maximum_sessions
        self._idle_timeout_seconds = idle_timeout_seconds
        self._clock = clock
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.RLock()

    def create(self, session_id: str, workspace: Path, **limits: int) -> StreamingVideoInputSession:
        with self._lock:
            self._reap_locked(self._clock())
            if session_id in self._entries:
                raise ValueError("streaming video session already exists")
            if len(self._entries) >= self._maximum_sessions:
                raise RuntimeError("streaming video session capacity reached")
            session = StreamingVideoInputSession(session_id, workspace, **limits)
            self._entries[session_id] = _Entry(session, self._clock())
            return session

    def get(self, session_id: str) -> StreamingVideoInputSession:
        with self._lock:
            entry = self._entries.get(session_id)
            if entry is None:
                raise KeyError("streaming video session not found")
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
            identifier for identifier, entry in self._entries.items()
            if now - entry.touched_at >= self._idle_timeout_seconds
        ))
        for identifier in expired:
            self._entries.pop(identifier).session.close()
        return expired
