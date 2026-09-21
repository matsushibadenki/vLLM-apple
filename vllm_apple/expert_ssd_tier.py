"""Private, integrity-checked SSD tier for inactive MoE experts."""
from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from .expert_residency import ExpertKey

MAX_EXPERT_SSD_ENTRIES = 65_536
MAX_EXPERT_SSD_BYTES = 1 << 42
MAX_SINGLE_EXPERT_BYTES = 1 << 40


@dataclass(frozen=True, slots=True)
class ExpertSSDValue:
    key: ExpertKey
    data: bytes
    sha256: str


@dataclass(slots=True)
class _Record:
    path: Path
    size: int
    sha256: str
    inode: int


class ExpertSSDStore:
    def __init__(self, root: Path, *, maximum_entries: int, maximum_bytes: int) -> None:
        if (not 1 <= maximum_entries <= MAX_EXPERT_SSD_ENTRIES
                or not 1 <= maximum_bytes <= MAX_EXPERT_SSD_BYTES):
            raise ValueError("invalid expert SSD bounds")
        self._root = Path(root)
        self._root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root_stat = self._root.lstat()
        if (not stat.S_ISDIR(root_stat.st_mode)
                or stat.S_IMODE(root_stat.st_mode) & 0o077
                or root_stat.st_uid != os.getuid()):
            raise ValueError("expert SSD root must be private and owner-controlled")
        self._maximum_entries = maximum_entries
        self._maximum_bytes = maximum_bytes
        self._records: OrderedDict[ExpertKey, _Record] = OrderedDict()
        self._bytes = self._hits = self._misses = self._evictions = 0
        self._integrity_failures = 0
        self._lock = threading.RLock()

    def put(self, key: ExpertKey, data: bytes) -> None:
        if not isinstance(key, ExpertKey) or not isinstance(data, bytes):
            raise ValueError("invalid expert SSD value")
        if not 1 <= len(data) <= min(self._maximum_bytes, MAX_SINGLE_EXPERT_BYTES):
            raise ValueError("expert SSD value exceeds bounds")
        digest = hashlib.sha256(data).hexdigest()
        final_path = self._root / f"layer-{key.layer}-expert-{key.expert}.bin"
        with self._lock:
            old = self._records.pop(key, None)
            if old is not None:
                self._unlink_record(old)
                self._bytes -= old.size
            fd, temporary_name = tempfile.mkstemp(prefix=".expert-", dir=self._root)
            temporary_path = Path(temporary_name)
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "wb", closefd=True) as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_path, final_path)
                directory_fd = os.open(self._root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                item_stat = final_path.lstat()
                self._records[key] = _Record(final_path, len(data), digest, item_stat.st_ino)
                self._bytes += len(data)
                self._evict_to_bounds()
            except BaseException:
                try:
                    temporary_path.unlink(missing_ok=True)
                finally:
                    if key not in self._records:
                        final_path.unlink(missing_ok=True)
                raise

    def get(self, key: ExpertKey) -> ExpertSSDValue | None:
        if not isinstance(key, ExpertKey):
            raise ValueError("invalid expert key")
        with self._lock:
            record = self._records.get(key)
            if record is None:
                self._misses += 1
                return None
            try:
                data = self._read_verified(record)
            except (OSError, ValueError):
                self._records.pop(key, None)
                self._bytes -= record.size
                self._integrity_failures += 1
                self._unlink_record(record)
                return None
            self._records.move_to_end(key)
            self._hits += 1
            return ExpertSSDValue(key, data, record.sha256)

    def remove(self, key: ExpertKey) -> bool:
        with self._lock:
            record = self._records.pop(key, None)
            if record is None:
                return False
            self._bytes -= record.size
            self._unlink_record(record)
            return True

    def clear(self) -> None:
        with self._lock:
            records = tuple(self._records.values())
            self._records.clear()
            self._bytes = 0
            for record in records:
                self._unlink_record(record)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._records),
                "bytes": self._bytes,
                "maximum_entries": self._maximum_entries,
                "maximum_bytes": self._maximum_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "integrity_failures": self._integrity_failures,
            }

    def _read_verified(self, record: _Record) -> bytes:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(record.path, flags)
        try:
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                    or stat.S_IMODE(before.st_mode) != 0o600
                    or before.st_ino != record.inode or before.st_size != record.size):
                raise ValueError("expert SSD identity mismatch")
            data = b""
            chunks: list[bytes] = []
            remaining = record.size
            while remaining:
                chunk = os.read(fd, min(8 << 20, remaining))
                if not chunk:
                    raise ValueError("expert SSD file truncated")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(fd, 1):
                raise ValueError("expert SSD file grew during read")
            data = b"".join(chunks)
            after = os.fstat(fd)
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                    after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError("expert SSD file changed during read")
            if hashlib.sha256(data).hexdigest() != record.sha256:
                raise ValueError("expert SSD digest mismatch")
            return data
        finally:
            os.close(fd)

    def _evict_to_bounds(self) -> None:
        while (len(self._records) > self._maximum_entries
               or self._bytes > self._maximum_bytes):
            _, record = self._records.popitem(last=False)
            self._bytes -= record.size
            self._evictions += 1
            self._unlink_record(record)

    @staticmethod
    def _unlink_record(record: _Record) -> None:
        try:
            record.path.unlink(missing_ok=True)
        except OSError:
            pass
