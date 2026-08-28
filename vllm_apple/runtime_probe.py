from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
import subprocess
from pathlib import Path

from .execution import AppleChipProfile, ExecutionBackend
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

    def to_dict(self) -> dict[str, object]:
        return {
            "hardware_fingerprint": self.hardware_fingerprint,
            "environment_fingerprint": self.environment_fingerprint,
            "results": [result.to_dict() for result in self.results],
            "dispatcher_applied": self.dispatcher_applied,
            "cache_status": self.cache_status,
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
        cache_path: Path | None = None,
        cache_root: Path | None = None,
    ) -> None:
        if cache_path is not None and cache_root is not None:
            raise ValueError("cache_path and cache_root are mutually exclusive")
        self.chip = chip
        self.environment_fingerprint = build_environment_fingerprint(
            platform=f"{chip.platform}-{chip.architecture}",
            os_version=chip.os_version,
            toolchain_version=toolchain_version,
            mlx_version=mlx_version,
            backend_version=backend_version,
        )
        self.mlx_adapter = mlx_adapter or NativeMLXProbeAdapter()
        self.metal_adapter = metal_adapter or NativeMetalProbeAdapter()
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
        applied = service.install_operator_dispatcher(OperatorDispatcher(registry))
        return RuntimeProbeReport(
            hardware_fingerprint=self.chip.hardware_fingerprint,
            environment_fingerprint=self.environment_fingerprint,
            results=tuple(results),
            dispatcher_applied=applied,
            cache_status=cache_status,
        )

    def _run_probes(self, samples: int) -> list[KernelProbeResult]:
        results: list[KernelProbeResult] = []
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
