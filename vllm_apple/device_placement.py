"""Versioned, benchmark-evidenced operator placement plans."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from .device_benchmark import DeviceBenchmarkReport
from .device_selection import DevicePlacementDecision
from .execution import ExecutionBackend, WorkloadPhase
from .hardware import default_application_support

DEVICE_PLACEMENT_SCHEMA_VERSION = 1
MAX_DEVICE_PLACEMENTS = 64
MAX_DEVICE_PLACEMENT_BYTES = 256 * 1024
MAX_DEVICE_PLACEMENT_TTL_SECONDS = 30 * 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class DevicePlacement:
    operator: str
    phase: WorkloadPhase
    precision: str
    dimensions: tuple[int, ...]
    batch_size: int
    backend: ExecutionBackend
    baseline_backend: ExecutionBackend
    capability_id: str
    benchmark_report_id: str
    improvement_ratio: float

    def __post_init__(self) -> None:
        if (
            not self.operator
            or not self.precision
            or not self.dimensions
            or self.batch_size <= 0
            or len(self.capability_id) != 24
            or len(self.benchmark_report_id) != 24
            or not 0 <= self.improvement_ratio < 1
        ):
            raise ValueError("invalid device placement")


@dataclass(frozen=True, slots=True)
class DevicePlacementPlan:
    hardware_fingerprint: str
    environment_fingerprint: str
    placements: tuple[DevicePlacement, ...]
    created_at_unix_seconds: int
    valid_until_unix_seconds: int
    plan_id: str
    schema_version: int = DEVICE_PLACEMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != DEVICE_PLACEMENT_SCHEMA_VERSION
            or not self.hardware_fingerprint
            or not self.environment_fingerprint
            or not 1 <= len(self.placements) <= MAX_DEVICE_PLACEMENTS
            or type(self.created_at_unix_seconds) is not int
            or type(self.valid_until_unix_seconds) is not int
            or not 0 < self.created_at_unix_seconds < self.valid_until_unix_seconds
            or self.valid_until_unix_seconds - self.created_at_unix_seconds
            > MAX_DEVICE_PLACEMENT_TTL_SECONDS
            or len(self.plan_id) != 24
            or self.plan_id != _placement_plan_id(
                self.hardware_fingerprint,
                self.environment_fingerprint,
                self.placements,
                self.created_at_unix_seconds,
                self.valid_until_unix_seconds,
            )
        ):
            raise ValueError("invalid device placement plan")
        keys = {
            (value.operator, value.phase, value.precision, value.dimensions, value.batch_size)
            for value in self.placements
        }
        if len(keys) != len(self.placements):
            raise ValueError("device placement plan contains duplicate workload identities")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "hardware_fingerprint": self.hardware_fingerprint,
            "environment_fingerprint": self.environment_fingerprint,
            "created_at_unix_seconds": self.created_at_unix_seconds,
            "valid_until_unix_seconds": self.valid_until_unix_seconds,
            "placements": [_placement_dict(value) for value in self.placements],
        }


def build_device_placement_plan(
    reports: tuple[DeviceBenchmarkReport, ...],
    decision: DevicePlacementDecision,
    *,
    ttl_seconds: int = 24 * 60 * 60,
    clock: Callable[[], float] = time.time,
) -> DevicePlacementPlan:
    selected = next(
        (report for report in reports if report.config.backend is decision.selected),
        None,
    )
    candidate = next(
        (value for value in decision.candidates if value.backend is decision.selected),
        None,
    )
    if selected is None or candidate is None or not candidate.eligible:
        raise ValueError("device placement winner lacks eligible benchmark evidence")
    created_at = int(clock())
    if (
        created_at <= 0
        or type(ttl_seconds) is not int
        or not 1 <= ttl_seconds <= MAX_DEVICE_PLACEMENT_TTL_SECONDS
    ):
        raise ValueError("invalid device placement lifetime")
    valid_until = created_at + ttl_seconds
    placement = DevicePlacement(
        selected.config.operator,
        selected.config.phase,
        selected.config.precision,
        selected.config.dimensions,
        selected.config.batch_size,
        selected.config.backend,
        decision.baseline,
        selected.capability_id,
        selected.report_id,
        decision.improvement_ratio,
    )
    plan_id = _placement_plan_id(
        selected.hardware_fingerprint,
        selected.environment_fingerprint,
        (placement,),
        created_at,
        valid_until,
    )
    return DevicePlacementPlan(
        selected.hardware_fingerprint,
        selected.environment_fingerprint,
        (placement,),
        created_at,
        valid_until,
        plan_id,
    )


def _placement_plan_id(
    hardware_fingerprint: str,
    environment_fingerprint: str,
    placements: tuple[DevicePlacement, ...],
    created_at_unix_seconds: int,
    valid_until_unix_seconds: int,
) -> str:
    return hashlib.sha256(json.dumps(
        {
            "schema_version": DEVICE_PLACEMENT_SCHEMA_VERSION,
            "hardware_fingerprint": hardware_fingerprint,
            "environment_fingerprint": environment_fingerprint,
            "created_at_unix_seconds": created_at_unix_seconds,
            "valid_until_unix_seconds": valid_until_unix_seconds,
            "placements": [_placement_dict(value) for value in placements],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()).hexdigest()[:24]


def _placement_dict(placement: DevicePlacement) -> dict[str, object]:
    value = asdict(placement)
    value["phase"] = placement.phase.value
    value["backend"] = placement.backend.value
    value["baseline_backend"] = placement.baseline_backend.value
    value["dimensions"] = list(placement.dimensions)
    return value


def save_device_placement_plan(plan: DevicePlacementPlan, path: Path) -> Path:
    encoded = (json.dumps(plan.to_dict(), sort_keys=True, indent=2) + "\n").encode()
    if len(encoded) > MAX_DEVICE_PLACEMENT_BYTES:
        raise ValueError("device placement plan exceeded 256 KiB")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = path.parent.lstat()
    if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid() or parent.st_mode & 0o077:
        raise ValueError("device placement directory must be private")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


def load_device_placement_plan(
    path: Path,
    *,
    hardware_fingerprint: str,
    environment_fingerprint: str,
    clock: Callable[[], float] = time.time,
) -> DevicePlacementPlan:
    attributes = path.lstat()
    if (
        not stat.S_ISREG(attributes.st_mode)
        or attributes.st_uid != os.getuid()
        or attributes.st_mode & 0o077
        or attributes.st_size > MAX_DEVICE_PLACEMENT_BYTES
    ):
        raise ValueError("device placement plan must be a bounded private regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "plan_id", "hardware_fingerprint", "environment_fingerprint",
        "created_at_unix_seconds", "valid_until_unix_seconds", "placements",
    }:
        raise ValueError("invalid device placement plan fields")
    if (
        payload["hardware_fingerprint"] != hardware_fingerprint
        or payload["environment_fingerprint"] != environment_fingerprint
    ):
        raise ValueError("device placement plan identity mismatch")
    try:
        values = payload["placements"]
        fields = {
            "operator", "phase", "precision", "dimensions", "batch_size", "backend",
            "baseline_backend", "capability_id", "benchmark_report_id", "improvement_ratio",
        }
        if not isinstance(values, list) or any(
            not isinstance(value, dict) or set(value) != fields for value in values
        ):
            raise ValueError("invalid device placement values")
        plan = DevicePlacementPlan(
            payload["hardware_fingerprint"],
            payload["environment_fingerprint"],
            tuple(DevicePlacement(
                operator=value["operator"],
                phase=WorkloadPhase(value["phase"]),
                precision=value["precision"],
                dimensions=tuple(value["dimensions"]),
                batch_size=value["batch_size"],
                backend=ExecutionBackend(value["backend"]),
                baseline_backend=ExecutionBackend(value["baseline_backend"]),
                capability_id=value["capability_id"],
                benchmark_report_id=value["benchmark_report_id"],
                improvement_ratio=value["improvement_ratio"],
            ) for value in values),
            payload["created_at_unix_seconds"],
            payload["valid_until_unix_seconds"],
            payload["plan_id"],
            payload["schema_version"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid device placement plan values") from error
    now = int(clock())
    if plan.created_at_unix_seconds > now + 300:
        raise ValueError("device placement plan creation time is in the future")
    if now >= plan.valid_until_unix_seconds:
        raise ValueError("device placement plan expired")
    return plan


def load_device_placement_with_fallback(
    current: Path,
    last_known_good: Path,
    **identity,
) -> tuple[DevicePlacementPlan, str]:
    try:
        return load_device_placement_plan(current, **identity), "current"
    except (OSError, ValueError, json.JSONDecodeError):
        return load_device_placement_plan(last_known_good, **identity), "last_known_good"


def promote_device_placement_plan(
    plan: DevicePlacementPlan,
    current: Path,
    last_known_good: Path,
    *,
    clock: Callable[[], float] = time.time,
) -> Path:
    """Archive only a still-valid current plan before atomically promoting its replacement."""
    if current == last_known_good:
        raise ValueError("device placement current and rollback paths must differ")
    if current.exists():
        try:
            previous = load_device_placement_plan(
                current,
                hardware_fingerprint=plan.hardware_fingerprint,
                environment_fingerprint=plan.environment_fingerprint,
                clock=clock,
            )
        except (OSError, ValueError, json.JSONDecodeError):
            previous = None
        if previous is not None and previous.plan_id != plan.plan_id:
            save_device_placement_plan(previous, last_known_good)
    return save_device_placement_plan(plan, current)


def default_device_placement_paths(
    hardware_fingerprint: str,
    environment_fingerprint: str,
    *,
    application_support: Path | None = None,
) -> tuple[Path, Path]:
    if not hardware_fingerprint or not environment_fingerprint:
        raise ValueError("device placement path identity cannot be empty")
    if any(
        len(value) > 128
        or any(not (character.isalnum() or character in "-_.") for character in value)
        for value in (hardware_fingerprint, environment_fingerprint)
    ):
        raise ValueError("device placement path identity is unsafe")
    root = (application_support or default_application_support()) / "profiles" / "device-placement"
    name = f"{hardware_fingerprint}-{environment_fingerprint}"
    return root / f"{name}.json", root / f"{name}.last-known-good.json"
