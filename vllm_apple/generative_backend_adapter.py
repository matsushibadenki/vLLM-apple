"""Versioned framework-specific adapters for generative qualification workers."""
from __future__ import annotations

import os
import re
import stat
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .generative_subprocess_adapter import SubprocessGenerativeTelemetryAdapter

MAX_BACKEND_VERSIONS = 32


class GenerativeBackendFamily(str, Enum):
    DIFFUSERS = "diffusers"
    MLX_GEN = "mlx-gen"
    MFLUX = "mflux"
    COMFYUI = "comfyui"


_MODULES = {
    GenerativeBackendFamily.DIFFUSERS: "vllm_apple.diffusers_generation_worker",
    GenerativeBackendFamily.MLX_GEN: "vllm_apple.mlx_gen_generation_worker",
    GenerativeBackendFamily.MFLUX: "vllm_apple.mflux_generation_worker",
}


@dataclass(frozen=True, slots=True)
class GenerativeWorkerCapability:
    schema_version: int
    family: GenerativeBackendFamily
    adapter_version: str
    backend_version: str
    executable: bool
    command_kind: str
    issues: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.adapter_version != "1.0.0"
            or not _valid_version(self.backend_version)
            or self.command_kind not in {"python_module", "external_worker"}
            or self.executable != (not self.issues)
            or len(self.issues) > 16
        ):
            raise ValueError("invalid generative worker capability")


class VersionedGenerativeWorkerAdapter:
    """Build a fixed, shell-free worker command after exact version admission."""

    adapter_version = "1.0.0"

    def __init__(
        self,
        family: GenerativeBackendFamily,
        *,
        backend_version: str,
        supported_backend_versions: tuple[str, ...],
        python_executable: Path | None = None,
        external_worker: Path | None = None,
    ) -> None:
        if (
            not isinstance(family, GenerativeBackendFamily)
            or not _valid_version(backend_version)
            or not 1 <= len(supported_backend_versions) <= MAX_BACKEND_VERSIONS
            or len(set(supported_backend_versions)) != len(supported_backend_versions)
            or any(not _valid_version(version) for version in supported_backend_versions)
        ):
            raise ValueError("invalid generative backend version contract")
        if family is GenerativeBackendFamily.COMFYUI:
            if python_executable is not None or external_worker is None:
                raise ValueError("ComfyUI requires one explicit external telemetry worker")
        elif python_executable is None or external_worker is not None:
            raise ValueError("Python generative backend requires one explicit interpreter")
        self.family = family
        self.backend_version = backend_version
        self.supported_backend_versions = frozenset(supported_backend_versions)
        self.python_executable = python_executable
        self.external_worker = external_worker

    def detect(self) -> GenerativeWorkerCapability:
        issues = []
        if self.backend_version not in self.supported_backend_versions:
            issues.append(f"backend_version_unsupported:{self.backend_version}")
        executable = self.external_worker or self.python_executable
        assert executable is not None
        if not _safe_executable(executable):
            issues.append("worker_executable_unsafe_or_missing")
        return GenerativeWorkerCapability(
            schema_version=1,
            family=self.family,
            adapter_version=self.adapter_version,
            backend_version=self.backend_version,
            executable=not issues,
            command_kind=(
                "external_worker"
                if self.family is GenerativeBackendFamily.COMFYUI
                else "python_module"
            ),
            issues=tuple(issues),
        )

    def worker_command(self) -> tuple[str, ...]:
        capability = self.detect()
        if not capability.executable:
            raise RuntimeError(",".join(capability.issues))
        if self.family is GenerativeBackendFamily.COMFYUI:
            assert self.external_worker is not None
            return (
                str(self.external_worker.expanduser().absolute()),
                "--telemetry-contract",
                "vllm-apple-generative-jsonl-v1",
                "--backend-version",
                self.backend_version,
            )
        assert self.python_executable is not None
        return (
            str(self.python_executable.expanduser().absolute()),
            "-m",
            _MODULES[self.family],
        )

    def telemetry_adapter(
        self,
        *,
        request_path: Path,
        workspace_root: Path,
        timeout_seconds: float,
    ) -> SubprocessGenerativeTelemetryAdapter:
        workspace = workspace_root.expanduser().resolve(strict=True)
        request_candidate = request_path.expanduser().absolute()
        if request_candidate.is_symlink():
            raise ValueError("generative worker request must be a real workspace file")
        request = request_candidate.resolve(strict=True)
        if (
            request == workspace
            or not request.is_relative_to(workspace)
            or request.is_symlink()
            or not request.is_file()
        ):
            raise ValueError("generative worker request must be a real workspace file")
        command = (
            *self.worker_command(),
            "--request",
            str(request),
            "--workspace-root",
            str(workspace),
        )
        return SubprocessGenerativeTelemetryAdapter(
            command,
            timeout_seconds=timeout_seconds,
            cwd=workspace,
        )


def _safe_executable(path: Path) -> bool:
    candidate = path.expanduser().absolute()
    try:
        info = candidate.lstat()
        resolved = candidate.resolve(strict=True)
        target = resolved.stat()
    except OSError:
        return False
    current_python = resolved == Path(sys.executable).resolve(strict=True)
    return (
        (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode))
        and (
            info.st_uid in (os.getuid(), 0)
            or (current_python and candidate == Path(sys.executable).absolute())
        )
        and stat.S_ISREG(target.st_mode)
        and (current_python or target.st_uid in (os.getuid(), 0))
        and (current_python or not stat.S_IMODE(target.st_mode) & 0o022)
        and os.access(resolved, os.X_OK)
    )


def _valid_version(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z.+_-]{0,63}", value) is not None
