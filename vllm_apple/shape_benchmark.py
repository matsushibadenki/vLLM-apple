from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .execution import ExecutionBackend
from .hardware import default_application_support
from .kernel_probe import KernelProbeResult, parse_kernel_probe_result
from .kernel_profile import ModelKernelShapeProfile
from .metal_probe import NativeMetalProbeAdapter

SHAPE_BENCHMARK_SCHEMA_VERSION = 1
MAX_SHAPE_BENCHMARK_BYTES = 256 * 1024


@dataclass(frozen=True, slots=True)
class MetalShapeBenchmark:
    schema_version: int
    benchmark_id: str
    profile_id: str
    model_id: str
    hardware_fingerprint: str
    environment_fingerprint: str
    created_at_unix_seconds: int
    results: tuple[KernelProbeResult, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SHAPE_BENCHMARK_SCHEMA_VERSION:
            raise ValueError("unsupported shape benchmark schema")
        if len(self.benchmark_id) != 24 or any(
            character not in "0123456789abcdef" for character in self.benchmark_id
        ):
            raise ValueError("invalid shape benchmark ID")
        if len(self.profile_id) != 24 or not self.model_id:
            raise ValueError("invalid shape benchmark model identity")
        if not self.hardware_fingerprint or not self.environment_fingerprint:
            raise ValueError("shape benchmark fingerprints cannot be empty")
        if self.created_at_unix_seconds <= 0:
            raise ValueError("shape benchmark creation time must be positive")
        if not 1 <= len(self.results) <= 16:
            raise ValueError("shape benchmark must contain 1 to 16 results")
        for result in self.results:
            if (
                result.backend is not ExecutionBackend.NATIVE_METAL
                or not result.operator.startswith("paged_attention:c")
                or result.hardware_fingerprint != self.hardware_fingerprint
                or result.environment_fingerprint != self.environment_fingerprint
            ):
                raise ValueError("shape result does not match benchmark identity")
        if self.benchmark_id != _benchmark_id(
            self.profile_id,
            self.hardware_fingerprint,
            self.environment_fingerprint,
            self.results,
        ):
            raise ValueError("shape benchmark ID does not match its results")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "benchmark_id": self.benchmark_id,
            "profile_id": self.profile_id,
            "model_id": self.model_id,
            "hardware_fingerprint": self.hardware_fingerprint,
            "environment_fingerprint": self.environment_fingerprint,
            "created_at_unix_seconds": self.created_at_unix_seconds,
            "results": [result.to_dict() for result in self.results],
        }


def run_metal_shape_benchmark(
    profile: ModelKernelShapeProfile,
    adapter: NativeMetalProbeAdapter,
    *,
    hardware_fingerprint: str,
    environment_fingerprint: str,
    samples: int = 1,
    maximum_shapes: int = 4,
    clock: Callable[[], float] = time.time,
) -> MetalShapeBenchmark:
    created_at = int(clock())
    if created_at <= 0:
        raise ValueError("shape benchmark clock must be positive")
    results = adapter.probe_shape_profile(
        profile,
        hardware_fingerprint=hardware_fingerprint,
        environment_fingerprint=environment_fingerprint,
        samples=samples,
        maximum_shapes=maximum_shapes,
    )
    benchmark_id = _benchmark_id(
        profile.profile_id, hardware_fingerprint, environment_fingerprint, results
    )
    return MetalShapeBenchmark(
        schema_version=SHAPE_BENCHMARK_SCHEMA_VERSION,
        benchmark_id=benchmark_id,
        profile_id=profile.profile_id,
        model_id=profile.model_id,
        hardware_fingerprint=hardware_fingerprint,
        environment_fingerprint=environment_fingerprint,
        created_at_unix_seconds=created_at,
        results=results,
    )


def save_metal_shape_benchmark(benchmark: MetalShapeBenchmark, path: Path) -> Path:
    encoded = (
        json.dumps(benchmark.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_SHAPE_BENCHMARK_BYTES:
        raise ValueError("shape benchmark exceeded 256 KiB")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.getuid()
        or parent.st_mode & 0o077
    ):
        raise ValueError("shape benchmark directory must be private")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return path


def default_metal_shape_benchmark_path(benchmark: MetalShapeBenchmark) -> Path:
    return (
        default_application_support()
        / "profiles"
        / "metal-shapes"
        / benchmark.hardware_fingerprint
        / benchmark.environment_fingerprint
        / f"{benchmark.profile_id}-{benchmark.benchmark_id}.json"
    )


def load_metal_shape_benchmark(
    path: Path,
    *,
    profile_id: str,
    hardware_fingerprint: str,
    environment_fingerprint: str,
) -> MetalShapeBenchmark:
    attributes = path.lstat()
    if (
        not stat.S_ISREG(attributes.st_mode)
        or attributes.st_uid != os.getuid()
        or attributes.st_mode & 0o077
    ):
        raise ValueError("shape benchmark must be a private current-user regular file")
    if attributes.st_size > MAX_SHAPE_BENCHMARK_BYTES:
        raise ValueError("shape benchmark exceeded 256 KiB")
    payload = json.loads(path.read_text(encoding="utf-8"))
    fields = {
        "schema_version",
        "benchmark_id",
        "profile_id",
        "model_id",
        "hardware_fingerprint",
        "environment_fingerprint",
        "created_at_unix_seconds",
        "results",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("invalid shape benchmark fields")
    if (
        payload["profile_id"] != profile_id
        or payload["hardware_fingerprint"] != hardware_fingerprint
        or payload["environment_fingerprint"] != environment_fingerprint
    ):
        raise ValueError("shape benchmark identity mismatch")
    created_at = payload["created_at_unix_seconds"]
    if not isinstance(created_at, int) or isinstance(created_at, bool) or created_at <= 0:
        raise ValueError("invalid shape benchmark creation time")
    values = payload["results"]
    if not isinstance(values, list) or not 1 <= len(values) <= 16:
        raise ValueError("invalid shape benchmark result count")
    try:
        return MetalShapeBenchmark(
            schema_version=payload["schema_version"],
            benchmark_id=payload["benchmark_id"],
            profile_id=payload["profile_id"],
            model_id=payload["model_id"],
            hardware_fingerprint=payload["hardware_fingerprint"],
            environment_fingerprint=payload["environment_fingerprint"],
            created_at_unix_seconds=created_at,
            results=tuple(parse_kernel_probe_result(value) for value in values),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid shape benchmark values") from error


def _benchmark_id(
    profile_id: str,
    hardware_fingerprint: str,
    environment_fingerprint: str,
    results: tuple[KernelProbeResult, ...],
) -> str:
    identity = {
        "schema_version": SHAPE_BENCHMARK_SCHEMA_VERSION,
        "profile_id": profile_id,
        "hardware_fingerprint": hardware_fingerprint,
        "environment_fingerprint": environment_fingerprint,
        "results": [result.probe_id for result in results],
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
