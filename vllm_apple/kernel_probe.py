from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from .execution import ExecutionBackend

KERNEL_PROBE_SCHEMA_VERSION = 1
MAX_PROBE_SAMPLES = 32
MAX_REGISTRY_ENTRIES = 512
MAX_PROBE_CACHE_BYTES = 1024 * 1024
PROBE_SUITE_VERSION = 8
DEFAULT_PROBE_CACHE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class KernelMeasurement:
    output_digest: str
    latency_nanoseconds: int
    numeric_values: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.output_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.output_digest
        ):
            raise ValueError("output_digest must be lowercase SHA-256")
        if self.latency_nanoseconds <= 0:
            raise ValueError("latency must be positive")
        if self.numeric_values is not None and (
            not 1 <= len(self.numeric_values) <= 4096
            or any(not math.isfinite(value) for value in self.numeric_values)
        ):
            raise ValueError("numeric probe values must be finite and bounded")


@dataclass(frozen=True, slots=True)
class KernelProbeConfig:
    hardware_fingerprint: str
    environment_fingerprint: str
    backend: ExecutionBackend
    operator: str
    samples: int = 3
    maximum_slowdown_ratio: float = 1.25
    maximum_absolute_error: float | None = None

    def __post_init__(self) -> None:
        if (
            not 1 <= len(self.hardware_fingerprint) <= 128
            or not 1 <= len(self.environment_fingerprint) <= 128
        ):
            raise ValueError("probe fingerprints cannot be empty")
        if not self.operator or len(self.operator) > 128:
            raise ValueError("operator must contain 1 to 128 characters")
        if not 1 <= self.samples <= MAX_PROBE_SAMPLES:
            raise ValueError("samples must be between 1 and 32")
        if (
            not math.isfinite(self.maximum_slowdown_ratio)
            or self.maximum_slowdown_ratio < 1
        ):
            raise ValueError("maximum slowdown ratio must be finite and at least 1")
        if self.maximum_absolute_error is not None and (
            not math.isfinite(self.maximum_absolute_error)
            or self.maximum_absolute_error < 0
        ):
            raise ValueError("maximum absolute error must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class KernelProbeResult:
    schema_version: int
    probe_id: str
    hardware_fingerprint: str
    environment_fingerprint: str
    backend: ExecutionBackend
    operator: str
    samples_completed: int
    baseline_latency_nanoseconds: int | None
    candidate_latency_nanoseconds: int | None
    slowdown_ratio: float | None
    passed: bool
    quarantined: bool
    reason: str

    def __post_init__(self) -> None:
        if self.schema_version != KERNEL_PROBE_SCHEMA_VERSION:
            raise ValueError("unsupported kernel probe schema version")
        if self.passed == self.quarantined:
            raise ValueError("probe must be either passed or quarantined")
        if self.passed != (self.reason == "passed"):
            raise ValueError("probe reason does not match status")
        if not 0 <= self.samples_completed <= MAX_PROBE_SAMPLES:
            raise ValueError("invalid completed sample count")
        if len(self.probe_id) != 24 or any(
            character not in "0123456789abcdef" for character in self.probe_id
        ):
            raise ValueError("invalid probe ID")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["backend"] = self.backend.value
        return result


def run_kernel_probe(
    config: KernelProbeConfig,
    baseline: Callable[[], KernelMeasurement],
    candidate: Callable[[], KernelMeasurement],
) -> KernelProbeResult:
    baseline_latencies: list[int] = []
    candidate_latencies: list[int] = []
    reason = "passed"
    completed = 0
    try:
        for _ in range(config.samples):
            reference = baseline()
            measured = candidate()
            if not _measurements_match(
                reference, measured, config.maximum_absolute_error
            ):
                reason = "correctness_mismatch"
                break
            baseline_latencies.append(reference.latency_nanoseconds)
            candidate_latencies.append(measured.latency_nanoseconds)
            completed += 1
    except Exception:
        reason = "probe_error"

    baseline_median = _median(baseline_latencies)
    candidate_median = _median(candidate_latencies)
    ratio = (
        candidate_median / baseline_median
        if baseline_median is not None and candidate_median is not None
        else None
    )
    if reason == "passed" and ratio is not None and ratio > config.maximum_slowdown_ratio:
        reason = "performance_regression"
    passed = reason == "passed" and completed == config.samples
    identity = {
        "schema_version": KERNEL_PROBE_SCHEMA_VERSION,
        "hardware_fingerprint": config.hardware_fingerprint,
        "environment_fingerprint": config.environment_fingerprint,
        "backend": config.backend.value,
        "operator": config.operator,
        "samples": config.samples,
        "maximum_slowdown_ratio": config.maximum_slowdown_ratio,
        "maximum_absolute_error": config.maximum_absolute_error,
    }
    probe_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return KernelProbeResult(
        schema_version=KERNEL_PROBE_SCHEMA_VERSION,
        probe_id=probe_id,
        hardware_fingerprint=config.hardware_fingerprint,
        environment_fingerprint=config.environment_fingerprint,
        backend=config.backend,
        operator=config.operator,
        samples_completed=completed,
        baseline_latency_nanoseconds=baseline_median,
        candidate_latency_nanoseconds=candidate_median,
        slowdown_ratio=round(ratio, 6) if ratio is not None else None,
        passed=passed,
        quarantined=not passed,
        reason=reason,
    )


def _measurements_match(
    reference: KernelMeasurement,
    measured: KernelMeasurement,
    maximum_absolute_error: float | None,
) -> bool:
    if reference.output_digest == measured.output_digest:
        return True
    if (
        maximum_absolute_error is None
        or reference.numeric_values is None
        or measured.numeric_values is None
        or len(reference.numeric_values) != len(measured.numeric_values)
    ):
        return False
    return all(
        abs(expected - actual) <= maximum_absolute_error
        for expected, actual in zip(reference.numeric_values, measured.numeric_values)
    )


class KernelCapabilityRegistry:
    """Bounded, profile-specific registry with sticky fail-closed quarantine."""

    def __init__(
        self,
        hardware_fingerprint: str,
        environment_fingerprint: str,
        capacity: int = MAX_REGISTRY_ENTRIES,
    ) -> None:
        if not hardware_fingerprint or not environment_fingerprint:
            raise ValueError("registry fingerprints cannot be empty")
        if not 1 <= capacity <= MAX_REGISTRY_ENTRIES:
            raise ValueError("registry capacity must be between 1 and 512")
        self.hardware_fingerprint = hardware_fingerprint
        self.environment_fingerprint = environment_fingerprint
        self.capacity = capacity
        self._results: dict[tuple[ExecutionBackend, str], KernelProbeResult] = {}
        self._lock = threading.RLock()

    def record(self, result: KernelProbeResult) -> None:
        if (
            result.hardware_fingerprint != self.hardware_fingerprint
            or result.environment_fingerprint != self.environment_fingerprint
        ):
            raise ValueError("probe result does not match registry profile")
        key = (result.backend, result.operator)
        with self._lock:
            existing = self._results.get(key)
            if existing is not None and existing.quarantined and result.passed:
                raise ValueError("quarantine is sticky for the current environment")
            if existing is None and len(self._results) >= self.capacity:
                raise ValueError("kernel capability registry is full")
            self._results[key] = result

    def is_usable(self, backend: ExecutionBackend, operator: str) -> bool:
        with self._lock:
            result = self._results.get((backend, operator))
            return bool(result and result.passed and not result.quarantined)

    def snapshot(self) -> tuple[KernelProbeResult, ...]:
        with self._lock:
            return tuple(
                self._results[key]
                for key in sorted(
                    self._results, key=lambda item: (item[0].value, item[1])
                )
            )


def build_environment_fingerprint(
    *,
    platform: str,
    os_version: str,
    toolchain_version: str,
    mlx_version: str,
    backend_version: str,
) -> str:
    values = {
        "probe_suite_version": str(PROBE_SUITE_VERSION),
        "platform": platform,
        "os_version": os_version,
        "toolchain_version": toolchain_version,
        "mlx_version": mlx_version,
        "backend_version": backend_version,
    }
    if any(not value or len(value) > 128 for value in values.values()):
        raise ValueError("environment components must contain 1 to 128 characters")
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]


class KernelProbeCache:
    def __init__(
        self,
        path: Path,
        hardware_fingerprint: str,
        environment_fingerprint: str,
        max_age_seconds: float = DEFAULT_PROBE_CACHE_MAX_AGE_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not math.isfinite(max_age_seconds) or max_age_seconds <= 0:
            raise ValueError("probe cache max age must be positive and finite")
        self.path = path
        self.hardware_fingerprint = hardware_fingerprint
        self.environment_fingerprint = environment_fingerprint
        self.max_age_seconds = max_age_seconds
        self._clock = clock

    def save(self, registry: KernelCapabilityRegistry) -> None:
        if (
            registry.hardware_fingerprint != self.hardware_fingerprint
            or registry.environment_fingerprint != self.environment_fingerprint
        ):
            raise ValueError("registry does not match probe cache profile")
        payload = {
            "schema_version": 1,
            "probe_suite_version": PROBE_SUITE_VERSION,
            "hardware_fingerprint": self.hardware_fingerprint,
            "environment_fingerprint": self.environment_fingerprint,
            "created_at_unix_seconds": int(self._clock()),
            "results": [result.to_dict() for result in registry.snapshot()],
        }
        encoded = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        if len(encoded) > MAX_PROBE_CACHE_BYTES:
            raise ValueError("kernel probe cache exceeded 1 MiB")
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        parent_attributes = self.path.parent.lstat()
        if (
            not stat.S_ISDIR(parent_attributes.st_mode)
            or parent_attributes.st_uid != os.getuid()
            or parent_attributes.st_mode & 0o077
        ):
            raise ValueError("kernel probe cache directory must be private")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    def load(self) -> KernelCapabilityRegistry:
        attributes = self.path.lstat()
        if not stat.S_ISREG(attributes.st_mode) or attributes.st_uid != os.getuid():
            raise ValueError("kernel probe cache must be a current-user regular file")
        if attributes.st_size > MAX_PROBE_CACHE_BYTES:
            raise ValueError("kernel probe cache exceeded 1 MiB")
        if attributes.st_mode & 0o077:
            raise ValueError("kernel probe cache must be private")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        required = {
            "schema_version",
            "probe_suite_version",
            "hardware_fingerprint",
            "environment_fingerprint",
            "created_at_unix_seconds",
            "results",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("invalid kernel probe cache fields")
        if payload["schema_version"] != 1 or payload["probe_suite_version"] != PROBE_SUITE_VERSION:
            raise ValueError("unsupported kernel probe cache version")
        if (
            payload["hardware_fingerprint"] != self.hardware_fingerprint
            or payload["environment_fingerprint"] != self.environment_fingerprint
        ):
            raise ValueError("kernel probe cache profile mismatch")
        created_at = payload["created_at_unix_seconds"]
        if (
            not isinstance(created_at, int)
            or isinstance(created_at, bool)
            or created_at <= 0
        ):
            raise ValueError("invalid kernel probe cache creation time")
        now = self._clock()
        if created_at > now + 300:
            raise ValueError("kernel probe cache creation time is in the future")
        if now - created_at > self.max_age_seconds:
            raise ValueError("kernel probe cache expired")
        values = payload["results"]
        if not isinstance(values, list) or len(values) > MAX_REGISTRY_ENTRIES:
            raise ValueError("invalid kernel probe cache result count")
        registry = KernelCapabilityRegistry(
            self.hardware_fingerprint, self.environment_fingerprint
        )
        for value in values:
            registry.record(parse_kernel_probe_result(value))
        return registry


def parse_kernel_probe_result(value: object) -> KernelProbeResult:
    fields = {
        "schema_version",
        "probe_id",
        "hardware_fingerprint",
        "environment_fingerprint",
        "backend",
        "operator",
        "samples_completed",
        "baseline_latency_nanoseconds",
        "candidate_latency_nanoseconds",
        "slowdown_ratio",
        "passed",
        "quarantined",
        "reason",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("invalid cached kernel probe fields")
    integer_fields = (
        "schema_version",
        "samples_completed",
    )
    if any(
        not isinstance(value[name], int) or isinstance(value[name], bool)
        for name in integer_fields
    ):
        raise ValueError("invalid cached kernel probe integer")
    for name in ("baseline_latency_nanoseconds", "candidate_latency_nanoseconds"):
        item = value[name]
        if item is not None and (
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
        ):
            raise ValueError("invalid cached kernel probe latency")
    ratio = value["slowdown_ratio"]
    if ratio is not None and (
        not isinstance(ratio, (int, float))
        or isinstance(ratio, bool)
        or not math.isfinite(ratio)
        or ratio < 0
    ):
        raise ValueError("invalid cached kernel probe ratio")
    if not isinstance(value["passed"], bool) or not isinstance(value["quarantined"], bool):
        raise ValueError("invalid cached kernel probe status")
    try:
        return KernelProbeResult(
            schema_version=value["schema_version"],
            probe_id=value["probe_id"],
            hardware_fingerprint=value["hardware_fingerprint"],
            environment_fingerprint=value["environment_fingerprint"],
            backend=ExecutionBackend(value["backend"]),
            operator=value["operator"],
            samples_completed=value["samples_completed"],
            baseline_latency_nanoseconds=value["baseline_latency_nanoseconds"],
            candidate_latency_nanoseconds=value["candidate_latency_nanoseconds"],
            slowdown_ratio=float(ratio) if ratio is not None else None,
            passed=value["passed"],
            quarantined=value["quarantined"],
            reason=value["reason"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError("invalid cached kernel probe values") from error


def _median(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[len(ordered) // 2]
