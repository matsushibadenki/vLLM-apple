from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
import subprocess
from pathlib import Path

from .ane_probe import (
    CoreMLANEModelProbe,
    CoreMLANEModelProbeConfig,
    CoreMLANESurfaceProbe,
    CoreMLANESurfaceResult,
)
from .cpu_probe import NativeCPUProbeAdapter
from .device_capability import (
    DeviceCapability,
    DeviceCapabilityRegistry,
    compose_device_capability_registry,
)
from .execution import AppleChipProfile, ExecutionBackend, WorkloadPhase
from .kernel_probe import (
    KernelCapabilityRegistry,
    KernelProbeCache,
    KernelProbeResult,
    build_environment_fingerprint,
)
from .metal_probe import NativeMetalProbeAdapter
from .mlx_probe import NativeMLXProbeAdapter
from .operator_dispatch import OperatorDispatcher
from .service import RuntimeService


@dataclass(frozen=True, slots=True)
class RuntimeProbeReport:
    hardware_fingerprint: str
    environment_fingerprint: str
    results: tuple[KernelProbeResult, ...]
    dispatcher_applied: bool
    cache_status: str
    device_capabilities: tuple[DeviceCapability, ...] = ()
    ane_surface: CoreMLANESurfaceResult | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "hardware_fingerprint": self.hardware_fingerprint,
            "environment_fingerprint": self.environment_fingerprint,
            "results": [result.to_dict() for result in self.results],
            "dispatcher_applied": self.dispatcher_applied,
            "cache_status": self.cache_status,
            "device_capabilities": [
                capability.to_dict() for capability in self.device_capabilities
            ],
            "ane_surface": None if self.ane_surface is None else self.ane_surface.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RuntimeEnvironmentVersions:
    toolchain_version: str
    mlx_version: str
    backend_version: str


def discover_runtime_versions(backend_version: str | None) -> RuntimeEnvironmentVersions:
    toolchain = _bounded_version(
        ("/usr/bin/xcrun", "-sdk", "macosx", "metal", "--version")
    )
    if toolchain == "unavailable":
        toolchain = _bounded_version(("/usr/bin/swift", "--version"))
    try:
        mlx_version = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        mlx_version = "unavailable"
    return RuntimeEnvironmentVersions(
        toolchain_version=toolchain,
        mlx_version=mlx_version[:128],
        backend_version=(backend_version or "unavailable")[:128],
    )


def _bounded_version(command: tuple[str, ...]) -> str:
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    output = " ".join(completed.stdout.split())
    return output[:128] if output else "unavailable"


class RuntimeProbeCoordinator:
    def __init__(
        self,
        chip: AppleChipProfile,
        *,
        toolchain_version: str,
        mlx_version: str,
        backend_version: str,
        mlx_adapter: NativeMLXProbeAdapter | None = None,
        metal_adapter: NativeMetalProbeAdapter | None = None,
        cpu_adapter: NativeCPUProbeAdapter | None = None,
        ane_surface_probe: CoreMLANESurfaceProbe | None = None,
        ane_model_probe: CoreMLANEModelProbe | None = None,
        ane_model_config: CoreMLANEModelProbeConfig | None = None,
        cache_path: Path | None = None,
        cache_root: Path | None = None,
    ) -> None:
        if cache_path is not None and cache_root is not None:
            raise ValueError("cache_path and cache_root are mutually exclusive")
        self.chip = chip
        self.toolchain_version = toolchain_version
        self.mlx_version = mlx_version
        self.backend_version = backend_version
        self.environment_fingerprint = build_environment_fingerprint(
            platform=f"{chip.platform}-{chip.architecture}",
            os_version=chip.os_version,
            toolchain_version=toolchain_version,
            mlx_version=mlx_version,
            backend_version=backend_version,
        )
        self.mlx_adapter = mlx_adapter or NativeMLXProbeAdapter()
        self.metal_adapter = metal_adapter or NativeMetalProbeAdapter()
        self.cpu_adapter = cpu_adapter or NativeCPUProbeAdapter()
        self.ane_surface_probe = ane_surface_probe or CoreMLANESurfaceProbe()
        self.ane_model_probe = ane_model_probe or CoreMLANEModelProbe()
        self.ane_model_config = ane_model_config
        self.cache_path = cache_path or (
            cache_root
            / f"{chip.hardware_fingerprint}-{self.environment_fingerprint}.json"
            if cache_root is not None
            else None
        )

    def probe_and_install(
        self, service: RuntimeService, *, samples: int = 1
    ) -> RuntimeProbeReport:
        cache = (
            KernelProbeCache(
                self.cache_path,
                self.chip.hardware_fingerprint,
                self.environment_fingerprint,
            )
            if self.cache_path is not None
            else None
        )
        cache_status = "disabled"
        registry = None
        if cache is not None and cache.path.exists():
            try:
                registry = cache.load()
                cache_status = "hit"
            except (OSError, ValueError, json.JSONDecodeError):
                cache_status = "rebuilt"
        if registry is None:
            if cache_status != "rebuilt":
                cache_status = "miss" if cache is not None else "disabled"
            registry = KernelCapabilityRegistry(
                self.chip.hardware_fingerprint, self.environment_fingerprint
            )
            results = self._run_probes(samples)
            for result in results:
                registry.record(result)
            if cache is not None:
                cache.save(registry)
            results = list(registry.snapshot())
        else:
            results = list(registry.snapshot())
        ane_surface = self.ane_surface_probe.probe(
            platform_name=self.chip.platform,
            architecture=self.chip.architecture,
        )
        if (
            self.ane_model_config is not None
            and ane_surface.reason == "surface_available_model_probe_required"
        ):
            registry.record(self.ane_model_probe.probe(
                self.ane_model_config,
                hardware_fingerprint=self.chip.hardware_fingerprint,
                environment_fingerprint=self.environment_fingerprint,
                samples=samples,
            ))
            results = list(registry.snapshot())
            if cache is not None:
                cache.save(registry)
        applied = service.install_operator_dispatcher(OperatorDispatcher(registry))
        device_registry = self._compose_device_registry(registry)
        return RuntimeProbeReport(
            hardware_fingerprint=self.chip.hardware_fingerprint,
            environment_fingerprint=self.environment_fingerprint,
            results=tuple(results),
            dispatcher_applied=applied,
            cache_status=cache_status,
            device_capabilities=device_registry.snapshot(),
            ane_surface=ane_surface,
        )

    def _compose_device_registry(
        self, registry: KernelCapabilityRegistry
    ) -> DeviceCapabilityRegistry:
        operators = {result.operator for result in registry.snapshot()}
        phases = {
            operator: (
                (WorkloadPhase.AUXILIARY,)
                if operator == "vector_add" or operator.startswith("coreml_fixed_graph@")
                else (WorkloadPhase.PREFILL, WorkloadPhase.DECODE)
            )
            for operator in operators
        }
        versions = {
            ExecutionBackend.NATIVE_MLX: self.mlx_version,
            ExecutionBackend.NATIVE_METAL: self.toolchain_version,
            ExecutionBackend.VLLM_METAL: self.backend_version,
            ExecutionBackend.COREML_DRAFT: self.chip.os_version,
            ExecutionBackend.CPU: self.chip.os_version,
        }
        return compose_device_capability_registry(
            registry,
            backend_versions=versions,
            phases_by_operator=phases,
            precisions_by_operator={operator: ("fp32",) for operator in operators},
        )

    def _run_probes(self, samples: int) -> list[KernelProbeResult]:
        results: list[KernelProbeResult] = []
        if ExecutionBackend.CPU in self.chip.backends:
            results.extend(
                self.cpu_adapter.probe_suite(
                    hardware_fingerprint=self.chip.hardware_fingerprint,
                    environment_fingerprint=self.environment_fingerprint,
                    samples=samples,
                )
            )
        if ExecutionBackend.NATIVE_MLX in self.chip.backends:
            results.extend(
                self.mlx_adapter.probe_suite(
                    hardware_fingerprint=self.chip.hardware_fingerprint,
                    environment_fingerprint=self.environment_fingerprint,
                    samples=samples,
                )
            )
        if self.chip.platform == "Darwin" and self.chip.architecture == "arm64":
            results.extend(
                self.metal_adapter.probe_suite(
                    hardware_fingerprint=self.chip.hardware_fingerprint,
                    environment_fingerprint=self.environment_fingerprint,
                    samples=samples,
                )
            )
        return results
