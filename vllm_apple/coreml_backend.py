"""Probe-gated lifecycle for bounded Core ML fixed-graph execution."""
from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass
from pathlib import Path

from .ane_probe import (
    CoreMLANEModelProbeConfig,
    CoreMLPrediction,
    run_coreml_prediction,
)
from .device_capability import (
    ComputeDevice,
    DeviceCapability,
    DeviceCapabilityRegistry,
    DeviceEligibilityRequest,
)
from .execution import ExecutionBackend, WorkloadPhase
from .model_integrity import verify_model_integrity


MAX_COREML_RESOURCES = 32


@dataclass(frozen=True, slots=True)
class CoreMLFixedGraphResource:
    resource_id: str
    operator: str
    capability_id: str
    model_root_sha256: str


@dataclass(frozen=True, slots=True)
class CoreMLFixedGraphResult:
    values: tuple[float, ...]
    latency_nanoseconds: int
    backend: ExecutionBackend
    capability_id: str


class CoreMLFixedGraphBackend:
    """Own resources whose model and execution evidence are cryptographically bound."""

    def __init__(
        self,
        *,
        swift_executable: Path = Path("/usr/bin/swift"),
        timeout_seconds: float = 30,
        maximum_output_bytes: int = 128 * 1024,
    ) -> None:
        self.swift_executable = swift_executable
        self.timeout_seconds = timeout_seconds
        self.maximum_output_bytes = maximum_output_bytes
        self._resources: dict[str, tuple[CoreMLANEModelProbeConfig, DeviceCapability]] = {}
        self._lock = threading.RLock()

    def load(
        self,
        config: CoreMLANEModelProbeConfig,
        capability: DeviceCapability,
    ) -> CoreMLFixedGraphResource:
        operator = f"coreml_fixed_graph@{config.model_root_sha256[:16]}"
        if (
            capability.backend is not ExecutionBackend.COREML_DRAFT
            or capability.device is not ComputeDevice.ANE
            or capability.status != "available"
            or capability.reason != "probe_passed"
            or capability.operators != (operator,)
            or WorkloadPhase.AUXILIARY not in capability.phases
            or "fp32" not in capability.precisions
        ):
            raise ValueError("Core ML resource capability is not execution eligible")
        manifest = verify_model_integrity(config.model_path, config.integrity_manifest_path)
        if manifest.get("root_sha256") != config.model_root_sha256:
            raise ValueError("Core ML resource integrity digest mismatch")
        with self._lock:
            if len(self._resources) >= MAX_COREML_RESOURCES:
                raise RuntimeError("Core ML resource limit reached")
            resource_id = secrets.token_hex(16)
            self._resources[resource_id] = (config, capability)
        return CoreMLFixedGraphResource(
            resource_id, operator, capability.capability_id, config.model_root_sha256
        )

    def execute(
        self,
        resource: CoreMLFixedGraphResource,
        input_values: tuple[float, ...],
    ) -> CoreMLFixedGraphResult:
        with self._lock:
            entry = self._resources.get(resource.resource_id)
        if entry is None:
            raise RuntimeError("Core ML resource is not loaded")
        config, capability = entry
        if (
            resource.operator != capability.operators[0]
            or resource.capability_id != capability.capability_id
            or resource.model_root_sha256 != config.model_root_sha256
        ):
            raise ValueError("Core ML resource handle is invalid")
        before = verify_model_integrity(config.model_path, config.integrity_manifest_path)
        if before.get("root_sha256") != config.model_root_sha256:
            raise ValueError("Core ML resource integrity digest mismatch")
        prediction: CoreMLPrediction = run_coreml_prediction(
            config,
            input_values,
            swift_executable=self.swift_executable,
            timeout_seconds=self.timeout_seconds,
            maximum_output_bytes=self.maximum_output_bytes,
        )
        after = verify_model_integrity(config.model_path, config.integrity_manifest_path)
        if after != before:
            raise ValueError("Core ML resource changed during execution")
        return CoreMLFixedGraphResult(
            prediction.values,
            prediction.latency_nanoseconds,
            ExecutionBackend.COREML_DRAFT,
            capability.capability_id,
        )

    def unload(self, resource: CoreMLFixedGraphResource) -> None:
        with self._lock:
            entry = self._resources.get(resource.resource_id)
            if entry is None:
                raise RuntimeError("Core ML resource is not loaded")
            config, capability = entry
            if (
                resource.operator != capability.operators[0]
                or resource.capability_id != capability.capability_id
                or resource.model_root_sha256 != config.model_root_sha256
            ):
                raise ValueError("Core ML resource handle is invalid")
            del self._resources[resource.resource_id]

    @staticmethod
    def require_auxiliary_dispatch(
        registry: DeviceCapabilityRegistry,
        resource: CoreMLFixedGraphResource,
    ) -> None:
        """Validate scheduler placement immediately before auxiliary execution."""
        decision = registry.decide(DeviceEligibilityRequest(
            resource.operator,
            WorkloadPhase.AUXILIARY,
            "fp32",
            (ExecutionBackend.COREML_DRAFT,),
        ))
        if decision.selected is not ExecutionBackend.COREML_DRAFT:
            raise RuntimeError("Core ML auxiliary dispatch was not selected")
        selected_capability = next(
            (
                capability
                for capability in registry.snapshot()
                if capability.backend is ExecutionBackend.COREML_DRAFT
                and capability.operators == (resource.operator,)
            ),
            None,
        )
        if (
            selected_capability is None
            or selected_capability.capability_id != resource.capability_id
        ):
            raise RuntimeError("Core ML resource capability evidence is stale")
