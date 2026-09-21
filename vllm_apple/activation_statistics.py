"""Constant-memory activation statistics and private aggregate streaming."""
from __future__ import annotations

import json
import math
import os
import stat
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

MAX_ACTIVATION_VALUES_PER_UPDATE = 1_048_576
MAX_ACTIVATION_COUNT = (1 << 63) - 1
MAX_STATISTICS_STREAM_BYTES = 1 << 30


@dataclass(frozen=True, slots=True)
class ActivationStatisticsSnapshot:
    count: int
    mean: float | None
    variance: float | None
    minimum: float | None
    maximum: float | None
    absolute_maximum: float | None
    zeros: int


class OnlineActivationStatistics:
    """Welford accumulator that never retains individual activation values."""

    def __init__(self) -> None:
        self._count = self._zeros = 0
        self._mean = self._m2 = 0.0
        self._minimum = math.inf
        self._maximum = -math.inf
        self._absolute_maximum = 0.0
        self._lock = threading.Lock()

    def update(self, values: Iterable[int | float]) -> int:
        local_count = local_zeros = 0
        local_mean = local_m2 = 0.0
        local_minimum = math.inf
        local_maximum = -math.inf
        local_absolute_maximum = 0.0
        for value in values:
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value)):
                raise ValueError("activation values must be finite numbers")
            local_count += 1
            if local_count > MAX_ACTIVATION_VALUES_PER_UPDATE:
                raise ValueError("activation update exceeds value bound")
            number = float(value)
            delta = number - local_mean
            local_mean += delta / local_count
            local_m2 += delta * (number - local_mean)
            local_minimum = min(local_minimum, number)
            local_maximum = max(local_maximum, number)
            local_absolute_maximum = max(local_absolute_maximum, abs(number))
            local_zeros += number == 0.0
        if local_count == 0:
            raise ValueError("activation update must not be empty")
        with self._lock:
            combined_count = self._count + local_count
            if combined_count > MAX_ACTIVATION_COUNT:
                raise ValueError("activation count overflow")
            if self._count:
                delta = local_mean - self._mean
                self._m2 += local_m2 + delta * delta * self._count * local_count / combined_count
                self._mean += delta * local_count / combined_count
            else:
                self._mean = local_mean
                self._m2 = local_m2
            self._count = combined_count
            self._zeros += local_zeros
            self._minimum = min(self._minimum, local_minimum)
            self._maximum = max(self._maximum, local_maximum)
            self._absolute_maximum = max(self._absolute_maximum, local_absolute_maximum)
        return local_count

    def snapshot(self) -> ActivationStatisticsSnapshot:
        with self._lock:
            if not self._count:
                return ActivationStatisticsSnapshot(0, None, None, None, None, None, 0)
            return ActivationStatisticsSnapshot(
                self._count,
                self._mean,
                self._m2 / self._count,
                self._minimum,
                self._maximum,
                self._absolute_maximum,
                self._zeros,
            )


class ActivationStatisticsStream:
    """Append-only private JSONL stream containing aggregate snapshots only."""

    def __init__(self, path: Path, *, maximum_bytes: int = 64 << 20) -> None:
        if not 1 <= maximum_bytes <= MAX_STATISTICS_STREAM_BYTES:
            raise ValueError("invalid statistics stream bound")
        self._path = Path(path)
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        parent = self._path.parent.lstat()
        if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid()
                or stat.S_IMODE(parent.st_mode) & 0o077):
            raise ValueError("statistics stream parent must be private")
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        self._fd = os.open(self._path, flags, 0o600)
        file_stat = os.fstat(self._fd)
        if (not stat.S_ISREG(file_stat.st_mode) or file_stat.st_uid != os.getuid()
                or stat.S_IMODE(file_stat.st_mode) != 0o600
                or file_stat.st_size > maximum_bytes):
            os.close(self._fd)
            raise ValueError("statistics stream file is unsafe")
        self._maximum_bytes = maximum_bytes
        self._bytes = file_stat.st_size
        self._records = 0
        self._closed = False
        self._lock = threading.Lock()

    def append(self, tensor_fingerprint: str, snapshot: ActivationStatisticsSnapshot) -> None:
        if (len(tensor_fingerprint) != 64
                or any(character not in "0123456789abcdef" for character in tensor_fingerprint)
                or not isinstance(snapshot, ActivationStatisticsSnapshot)):
            raise ValueError("invalid statistics stream record")
        payload = json.dumps({
            "schema_version": 1,
            "tensor_fingerprint": tensor_fingerprint,
            "statistics": asdict(snapshot),
        }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
        with self._lock:
            if self._closed:
                raise RuntimeError("statistics stream is closed")
            if self._bytes + len(payload) > self._maximum_bytes:
                raise ValueError("statistics stream capacity exceeded")
            written = 0
            while written < len(payload):
                count = os.write(self._fd, payload[written:])
                if count <= 0:
                    raise OSError("statistics stream write failed")
                written += count
            os.fsync(self._fd)
            self._bytes += len(payload)
            self._records += 1

    def snapshot(self) -> dict[str, int | bool]:
        with self._lock:
            return {
                "bytes": self._bytes,
                "maximum_bytes": self._maximum_bytes,
                "records": self._records,
                "closed": self._closed,
            }

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                os.close(self._fd)
                self._closed = True

    def __enter__(self) -> "ActivationStatisticsStream":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
