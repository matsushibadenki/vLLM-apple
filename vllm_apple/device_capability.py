"""Versioned CPU/GPU/ANE eligibility contracts for Apple runtime planning."""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass
from enum import Enum

from .execution import ExecutionBackend, WorkloadPhase
from .kernel_probe import KernelProbeResult

DEVICE_CAPABILITY_SCHEMA_VERSION = 1
MAX_DEVICE_CAPABILITIES = 64
MAX_ELIGIBILITY_VALUES = 128


class ComputeDevice(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    ANE = "ane"


_BACKEND_DEVICE = {
    ExecutionBackend.CPU: ComputeDevice.CPU,
    ExecutionBackend.VLLM_METAL: ComputeDevice.GPU,
    ExecutionBackend.NATIVE_MLX: ComputeDevice.GPU,
    ExecutionBackend.NATIVE_METAL: ComputeDevice.GPU,
    ExecutionBackend.COREML_DRAFT: ComputeDevice.ANE,
}


def _bounded_values(values: tuple[str, ...], label: str) -> None:
    if (
        not values
        or len(values) > MAX_ELIGIBILITY_VALUES
        or len(set(values)) != len(values)
        or any(
            not isinstance(value, str)
            or not 1 <= len(value) <= 128
            or any(ord(character) < 0x20 for character in value)
            for value in values
        )
    ):
        raise ValueError(f"invalid device capability {label}")


@dataclass(frozen=True, slots=True)
class DeviceCapability:
    backend: ExecutionBackend
    device: ComputeDevice
    backend_version: str
    hardware_fingerprint: str
    environment_fingerprint: str
    operators: tuple[str, ...]
    phases: tuple[WorkloadPhase, ...]
    precisions: tuple[str, ...]
    status: str
    reason: str
    evidence_ids: tuple[str, ...]
    schema_version: int = DEVICE_CAPABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DEVICE_CAPABILITY_SCHEMA_VERSION:
            raise ValueError("unsupported device capability schema version")
        if _BACKEND_DEVICE.get(self.backend) is not self.device:
            raise ValueError("backend and compute device do not match")
        for value, label in (
            (self.backend_version, "backend version"),
            (self.hardware_fingerprint, "hardware fingerprint"),
            (self.environment_fingerprint, "environment fingerprint"),
            (self.reason, "reason"),
        ):
            if not isinstance(value, str) or not 1 <= len(value) <= 128:
                raise ValueError(f"invalid device capability {label}")
        _bounded_values(self.operators, "operators")
        _bounded_values(self.precisions, "precisions")
        if (
            not self.phases
            or len(set(self.phases)) != len(self.phases)
            or any(not isinstance(phase, WorkloadPhase) for phase in self.phases)
        ):
            raise ValueError("invalid device capability phases")
        if self.status not in {"available", "unavailable", "quarantined"}:
            raise ValueError("invalid device capability status")
        if self.status == "available" and self.reason != "probe_passed":
            raise ValueError("available capability requires passing probe evidence")
        if self.status != "available" and self.reason == "probe_passed":
            raise ValueError("unusable capability cannot have passing evidence")
        if (
            len(self.evidence_ids) > MAX_ELIGIBILITY_VALUES
            or len(set(self.evidence_ids)) != len(self.evidence_ids)
            or any(
                len(value) != 24
                or any(character not in "0123456789abcdef" for character in value)
                for value in self.evidence_ids
            )
            or (self.status == "available" and not self.evidence_ids)
        ):
            raise ValueError("invalid device capability evidence")

    @property
    def capability_id(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["backend"] = self.backend.value
        result["device"] = self.device.value
        result["phases"] = [phase.value for phase in self.phases]
        result["operators"] = list(self.operators)
        result["precisions"] = list(self.precisions)
        result["evidence_ids"] = list(self.evidence_ids)
        return result


def device_capability_from_probe(
    result: KernelProbeResult,
    *,
    device: ComputeDevice,
    backend_version: str,
    phases: tuple[WorkloadPhase, ...],
    precisions: tuple[str, ...],
) -> DeviceCapability:
    """Promote one profile-bound kernel probe without weakening its quarantine."""
    if not isinstance(result, KernelProbeResult):
        raise ValueError("device capability probe result is invalid")
    return DeviceCapability(
        backend=result.backend,
        device=device,
        backend_version=backend_version,
        hardware_fingerprint=result.hardware_fingerprint,
        environment_fingerprint=result.environment_fingerprint,
        operators=(result.operator,),
        phases=phases,
        precisions=precisions,
        status="available" if result.passed else "quarantined",
        reason="probe_passed" if result.passed else result.reason,
        evidence_ids=(result.probe_id,),
    )


@dataclass(frozen=True, slots=True)
class DeviceEligibilityRequest:
    operator: str
    phase: WorkloadPhase
    precision: str
    candidates: tuple[ExecutionBackend, ...]

    def __post_init__(self) -> None:
        _bounded_values((self.operator,), "operator")
        _bounded_values((self.precision,), "precision")
        if (
            not isinstance(self.phase, WorkloadPhase)
            or not self.candidates
            or len(self.candidates) > len(ExecutionBackend)
            or len(set(self.candidates)) != len(self.candidates)
        ):
            raise ValueError("invalid device eligibility request")


@dataclass(frozen=True, slots=True)
class DeviceEligibilityDecision:
    selected: ExecutionBackend
    fallback_chain: tuple[ExecutionBackend, ...]
    rejected: tuple[tuple[ExecutionBackend, str], ...]


class DeviceCapabilityRegistry:
    """Profile-bound registry; unknown accelerators never become eligible implicitly."""

    def __init__(self, hardware_fingerprint: str, environment_fingerprint: str) -> None:
        if not hardware_fingerprint or not environment_fingerprint:
            raise ValueError("device registry fingerprints cannot be empty")
        self.hardware_fingerprint = hardware_fingerprint
        self.environment_fingerprint = environment_fingerprint
        self._entries: dict[tuple[ExecutionBackend, str], DeviceCapability] = {}
        self._lock = threading.RLock()

    def record(self, capability: DeviceCapability) -> None:
        if (
            not isinstance(capability, DeviceCapability)
            or capability.hardware_fingerprint != self.hardware_fingerprint
            or capability.environment_fingerprint != self.environment_fingerprint
        ):
            raise ValueError("device capability does not match registry profile")
        if len(capability.operators) != 1:
            raise ValueError("device registry entries require exactly one operator")
        key = (capability.backend, capability.operators[0])
        with self._lock:
            existing = self._entries.get(key)
            if (
                existing is not None
                and existing.status == "quarantined"
                and capability.status == "available"
            ):
                raise ValueError("device capability quarantine is sticky")
            if existing is None and len(self._entries) >= MAX_DEVICE_CAPABILITIES:
                raise ValueError("device capability registry is full")
            self._entries[key] = capability

    def decide(self, request: DeviceEligibilityRequest) -> DeviceEligibilityDecision:
        eligible: list[ExecutionBackend] = []
        rejected: list[tuple[ExecutionBackend, str]] = []
        with self._lock:
            for backend in request.candidates:
                capability = self._entries.get((backend, request.operator))
                if capability is None:
                    rejected.append((backend, "unprobed"))
                elif capability.status != "available":
                    rejected.append((backend, capability.status))
                elif request.operator not in capability.operators:
                    rejected.append((backend, "operator_ineligible"))
                elif request.phase not in capability.phases:
                    rejected.append((backend, "phase_ineligible"))
                elif request.precision not in capability.precisions:
                    rejected.append((backend, "precision_ineligible"))
                else:
                    eligible.append(backend)
        if not eligible:
            raise RuntimeError("device capability fallback chain is exhausted")
        return DeviceEligibilityDecision(
            eligible[0], tuple(eligible[1:]), tuple(rejected)
        )

    def snapshot(self) -> tuple[DeviceCapability, ...]:
        with self._lock:
            return tuple(
                self._entries[key]
                for key in sorted(self._entries, key=lambda value: (value[0].value, value[1]))
            )


def compose_device_capability_registry(
    kernel_registry,
    *,
    backend_versions: dict[ExecutionBackend, str],
    phases_by_operator: dict[str, tuple[WorkloadPhase, ...]],
    precisions_by_operator: dict[str, tuple[str, ...]],
) -> DeviceCapabilityRegistry:
    """Lift measured kernel results into device placement eligibility."""
    if not hasattr(kernel_registry, "snapshot"):
        raise ValueError("kernel capability registry is invalid")
    registry = DeviceCapabilityRegistry(
        kernel_registry.hardware_fingerprint,
        kernel_registry.environment_fingerprint,
    )
    for result in kernel_registry.snapshot():
        version = backend_versions.get(result.backend)
        phases = phases_by_operator.get(result.operator)
        precisions = precisions_by_operator.get(result.operator)
        if version is None or phases is None or precisions is None:
            raise ValueError("device capability composition metadata is incomplete")
        registry.record(device_capability_from_probe(
            result,
            device=_BACKEND_DEVICE[result.backend],
            backend_version=version,
            phases=phases,
            precisions=precisions,
        ))
    return registry
