from __future__ import annotations

import json
import os
import re
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
    transformers_version: str | None = None
    platform_module: str | None = None
    platform_class: str | None = None
    platform_is_cpu: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["issues"] = list(self.issues)
        return result


@dataclass(frozen=True, slots=True)
class MLXBackendCompatibility:
    executable: str | None
    mlx_lm_version: str | None
    compatible: bool
    issues: tuple[str, ...]
    architecture_features: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["issues"] = list(self.issues)
        result["architecture_features"] = list(self.architecture_features)
        return result


MAX_MLX_PROBE_OUTPUT_BYTES = 16 * 1024
MAX_MLX_ARCHITECTURE_FEATURES = 64
_MLX_FEATURE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _decode_mlx_probe(payload: str) -> tuple[str, tuple[str, ...]]:
    if not 1 <= len(payload.encode("utf-8")) <= MAX_MLX_PROBE_OUTPUT_BYTES:
        raise ValueError("mlx_lm probe output is outside the bounded limit")
    decoded = json.loads(payload)
    if not isinstance(decoded, dict) or set(decoded) != {"version", "architecture_features"}:
        raise ValueError("mlx_lm probe output has an invalid schema")
    version = decoded["version"]
    features = decoded["architecture_features"]
    if not isinstance(version, str) or not 1 <= len(version) <= 128:
        raise ValueError("mlx_lm probe version is invalid")
    if (
        not isinstance(features, list)
        or len(features) > MAX_MLX_ARCHITECTURE_FEATURES
        or any(not isinstance(value, str) or not _MLX_FEATURE_PATTERN.fullmatch(value) for value in features)
    ):
        raise ValueError("mlx_lm architecture features are invalid")
    return version, tuple(sorted(set(features)))


def inspect_mlx_lm_backend(executable: str | Path) -> MLXBackendCompatibility:
    path = Path(executable).expanduser()
    if not path.is_file() or not os.access(path, os.X_OK):
        return MLXBackendCompatibility(None, None, False, ("mlx_lm_server_not_executable",))
    python = path.parent / "python"
    if not python.is_file():
        python = Path(sys.executable)
    try:
        result = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import importlib.metadata as m,json;import mlx_lm;"
                    "f=getattr(mlx_lm,'VLLM_APPLE_ARCHITECTURE_FEATURES',());"
                    "f=f if isinstance(f,(list,tuple,frozenset)) else ();"
                    "print(json.dumps({'version':m.version('mlx-lm'),"
                    "'architecture_features':list(f)}))"
                ),
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=5.0,
        )
        version, features = _decode_mlx_probe(result.stdout.strip())
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError, json.JSONDecodeError):
        return MLXBackendCompatibility(
            str(path.resolve()), None, False, ("mlx_lm_version_unavailable",)
        )
    parsed = _release_tuple(version)
    issues = () if parsed is not None and (0, 26, 0) <= parsed < (0, 27, 0) else (
        "mlx_lm_version_outside_verified_matrix",
    )
    return MLXBackendCompatibility(
        str(path.resolve()), version, not issues, issues, features
    )


def resolve_vllm_executable(explicit: str | None = None) -> Path | None:
    candidate = explicit or os.environ.get("VLLM_APPLE_VLLM_EXECUTABLE")
    if candidate:
        path = Path(candidate).expanduser()
        return path.resolve() if path.is_file() and os.access(path, os.X_OK) else None
    discovered = shutil.which("vllm")
    return Path(discovered).resolve() if discovered else None


VERIFIED_VLLM_MIN = (0, 24, 0)
VERIFIED_VLLM_MAX_EXCLUSIVE = (0, 28, 0)
VERIFIED_VLLM_METAL_MIN = (0, 2, 0)
VERIFIED_VLLM_METAL_MAX_EXCLUSIVE = (0, 4, 0)
VERIFIED_TRANSFORMERS_MIN = (5, 5, 3)
VERIFIED_TRANSFORMERS_MAX_EXCLUSIVE = (5, 13, 0)


def _release_tuple(value: str | None) -> tuple[int, int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value or "")
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3) or 0)


def assess_backend_versions(
    *,
    vllm_version: str | None,
    vllm_metal_version: str | None,
    transformers_version: str | None,
) -> tuple[str, ...]:
    issues: list[str] = []
    values = (
        (
            "vllm",
            vllm_version,
            VERIFIED_VLLM_MIN,
            VERIFIED_VLLM_MAX_EXCLUSIVE,
        ),
        (
            "vllm_metal",
            vllm_metal_version,
            VERIFIED_VLLM_METAL_MIN,
            VERIFIED_VLLM_METAL_MAX_EXCLUSIVE,
        ),
        (
            "transformers",
            transformers_version,
            VERIFIED_TRANSFORMERS_MIN,
            VERIFIED_TRANSFORMERS_MAX_EXCLUSIVE,
        ),
    )
    for name, raw, minimum, maximum in values:
        if raw is None:
            continue
        parsed = _release_tuple(raw)
        if parsed is None:
            issues.append(f"{name}_version_unparseable")
        elif not minimum <= parsed < maximum:
            issues.append(f"{name}_version_outside_verified_matrix")
    return tuple(issues)


def assess_platform_selection(
    *,
    platform_module: str | None,
    platform_class: str | None,
    platform_is_cpu: bool | None,
) -> tuple[str, ...]:
    if platform_module is None or platform_class is None or platform_is_cpu is None:
        return ("vllm_platform_selection_unknown",)
    normalized = f"{platform_module}.{platform_class}".lower()
    if platform_is_cpu or "vllm_metal" not in normalized:
        return ("vllm_metal_platform_not_selected",)
    return ()


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
        "from vllm.platforms import current_platform as p;"
        "print(json.dumps({'python':platform.python_version(),'vllm':v('vllm'),"
        "'vllm_metal':v('vllm-metal'),'transformers':v('transformers'),"
        "'platform_module':type(p).__module__,'platform_class':type(p).__name__,"
        "'platform_is_cpu':p.is_cpu()}))"
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
    if payload.get("transformers") is None:
        issues.append("transformers_package_not_found")
    python_version = str(payload.get("python") or "")
    try:
        major, minor, *_ = (int(part) for part in python_version.split("."))
        if major != 3 or minor not in {12, 13}:
            issues.append("vllm_metal_requires_python_3_12_or_3_13")
    except ValueError:
        issues.append("backend_python_version_unknown")
    issues.extend(
        assess_backend_versions(
            vllm_version=payload.get("vllm"),
            vllm_metal_version=payload.get("vllm_metal"),
            transformers_version=payload.get("transformers"),
        )
    )
    issues.extend(
        assess_platform_selection(
            platform_module=payload.get("platform_module"),
            platform_class=payload.get("platform_class"),
            platform_is_cpu=payload.get("platform_is_cpu"),
        )
    )
    return BackendCompatibility(
        executable=str(resolved),
        python_version=python_version or None,
        vllm_version=payload.get("vllm"),
        vllm_metal_version=payload.get("vllm_metal"),
        compatible=not issues,
        issues=tuple(issues),
        transformers_version=payload.get("transformers"),
        platform_module=payload.get("platform_module"),
        platform_class=payload.get("platform_class"),
        platform_is_cpu=payload.get("platform_is_cpu"),
    )
