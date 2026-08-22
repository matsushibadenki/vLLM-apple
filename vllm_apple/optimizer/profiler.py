from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..types import HardwareInfo, MIB
from .types import OPTIMIZER_SCHEMA_VERSION


WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin")


@dataclass(frozen=True, slots=True)
class OptimizationPerformanceProfile:
    profile_id: str
    measured_at: str
    hardware_fingerprint: str
    sample_bytes: int
    read_bytes_per_second: int
    write_bytes_per_second: int

    def __post_init__(self) -> None:
        if not self.profile_id or not self.measured_at or len(self.hardware_fingerprint) < 16:
            raise ValueError("invalid optimization performance profile")
        if min(self.sample_bytes, self.read_bytes_per_second, self.write_bytes_per_second) <= 0:
            raise ValueError("performance measurements must be positive")

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": OPTIMIZER_SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "OptimizationPerformanceProfile":
        if payload.get("schema_version") != OPTIMIZER_SCHEMA_VERSION:
            raise ValueError("unsupported optimizer performance profile schema")
        fields = tuple(cls.__dataclass_fields__)
        if set(payload) != {"schema_version", *fields}:
            raise ValueError("invalid optimizer performance profile fields")
        string_fields = ("profile_id", "measured_at", "hardware_fingerprint")
        integer_fields = ("sample_bytes", "read_bytes_per_second", "write_bytes_per_second")
        if any(not isinstance(payload[field], str) for field in string_fields):
            raise ValueError("invalid optimizer performance profile strings")
        if any(
            not isinstance(payload[field], int) or isinstance(payload[field], bool)
            for field in integer_fields
        ):
            raise ValueError("invalid optimizer performance profile measurements")
        return cls(
            profile_id=payload["profile_id"],
            measured_at=payload["measured_at"],
            hardware_fingerprint=payload["hardware_fingerprint"],
            sample_bytes=payload["sample_bytes"],
            read_bytes_per_second=payload["read_bytes_per_second"],
            write_bytes_per_second=payload["write_bytes_per_second"],
        )


def profile_optimizer_io(
    model_path: Path,
    workspace: Path,
    hardware: HardwareInfo,
    sample_bytes: int = 64 * MIB,
) -> OptimizationPerformanceProfile:
    if not MIB <= sample_bytes <= 256 * MIB:
        raise ValueError("profile sample must be between 1 MiB and 256 MiB")
    source = model_path.expanduser().resolve(strict=True)
    destination = workspace.expanduser().resolve(strict=True)
    if not source.is_dir() or not destination.is_dir():
        raise ValueError("model and profiler workspace must be directories")
    if _overlap(source, destination):
        raise ValueError("profiler workspace must not overlap the source model")
    weight_files = sorted(
        path for path in source.rglob("*") if path.is_file() and path.name.endswith(WEIGHT_SUFFIXES)
    )
    if not weight_files:
        raise ValueError("no model weights available for profiling")
    read_bytes, read_seconds = _measure_read(weight_files, sample_bytes)
    write_bytes, write_seconds = _measure_write(destination, sample_bytes)
    fingerprint = hardware_fingerprint(hardware)
    profile_id = hashlib.sha256(
        f"{fingerprint}:{read_bytes}:{write_bytes}:{read_seconds}:{write_seconds}".encode()
    ).hexdigest()
    return OptimizationPerformanceProfile(
        profile_id=profile_id,
        measured_at=datetime.now(timezone.utc).isoformat(),
        hardware_fingerprint=fingerprint,
        sample_bytes=min(read_bytes, write_bytes),
        read_bytes_per_second=max(1, int(read_bytes / read_seconds)),
        write_bytes_per_second=max(1, int(write_bytes / write_seconds)),
    )


def hardware_fingerprint(hardware: HardwareInfo) -> str:
    payload = {
        "platform": hardware.platform,
        "architecture": hardware.architecture,
        "soc": hardware.soc,
        "gpu_core_count": hardware.gpu_core_count,
        "total_memory": hardware.memory.total_bytes,
        "os_version": hardware.os_version,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _measure_read(paths: list[Path], limit: int) -> tuple[int, float]:
    started = time.monotonic()
    total = 0
    for path in paths:
        with path.open("rb", buffering=0) as handle:
            while total < limit:
                chunk = handle.read(min(MIB, limit - total))
                if not chunk:
                    break
                total += len(chunk)
        if total >= limit:
            break
    elapsed = max(time.monotonic() - started, 1e-9)
    if total <= 0:
        raise ValueError("model weights are empty")
    return total, elapsed


def _measure_write(workspace: Path, limit: int) -> tuple[int, float]:
    descriptor, name = tempfile.mkstemp(prefix=".vllm-apple-profile-", dir=workspace)
    total = 0
    started = time.monotonic()
    try:
        block = bytes(MIB)
        with os.fdopen(descriptor, "wb", buffering=0) as handle:
            descriptor = -1
            while total < limit:
                chunk = block[: min(MIB, limit - total)]
                handle.write(chunk)
                total += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
    return total, max(time.monotonic() - started, 1e-9)


def _overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents
