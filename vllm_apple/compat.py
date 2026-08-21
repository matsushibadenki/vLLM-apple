from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class BackendCompatibility:
    executable: str | None
    python_version: str | None
    vllm_version: str | None
    vllm_metal_version: str | None
    compatible: bool
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["issues"] = list(self.issues)
        return result


def resolve_vllm_executable(explicit: str | None = None) -> Path | None:
    candidate = explicit or os.environ.get("VLLM_APPLE_VLLM_EXECUTABLE")
    if candidate:
        path = Path(candidate).expanduser()
        return path.resolve() if path.is_file() and os.access(path, os.X_OK) else None
    discovered = shutil.which("vllm")
    return Path(discovered).resolve() if discovered else None


def inspect_backend(executable: str | Path | None = None) -> BackendCompatibility:
    resolved = resolve_vllm_executable(str(executable) if executable else None)
    if resolved is None:
        return BackendCompatibility(
            executable=None,
            python_version=None,
            vllm_version=None,
            vllm_metal_version=None,
            compatible=False,
            issues=("vllm_executable_not_found",),
        )

    python = resolved.parent / "python"
    if not python.is_file():
        python = Path(sys.executable)
    script = (
        "import importlib.metadata as m,json,platform;"
        "v=lambda n: (m.version(n) if any(d.metadata.get('Name','').lower()==n "
        "for d in m.distributions()) else None);"
        "print(json.dumps({'python':platform.python_version(),'vllm':v('vllm'),"
        "'vllm_metal':v('vllm-metal')}))"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", script],
            capture_output=True,
            check=True,
            text=True,
            timeout=5.0,
        )
        payload = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return BackendCompatibility(
            executable=str(resolved),
            python_version=None,
            vllm_version=None,
            vllm_metal_version=None,
            compatible=False,
            issues=("backend_environment_inspection_failed",),
        )

    issues: list[str] = []
    if payload.get("vllm") is None:
        issues.append("vllm_package_not_found")
    if payload.get("vllm_metal") is None:
        issues.append("vllm_metal_package_not_found")
    python_version = str(payload.get("python") or "")
    try:
        major, minor, *_ = (int(part) for part in python_version.split("."))
        if major != 3 or minor not in {12, 13}:
            issues.append("vllm_metal_requires_python_3_12_or_3_13")
    except ValueError:
        issues.append("backend_python_version_unknown")
    return BackendCompatibility(
        executable=str(resolved),
        python_version=python_version or None,
        vllm_version=payload.get("vllm"),
        vllm_metal_version=payload.get("vllm_metal"),
        compatible=not issues,
        issues=tuple(issues),
    )

