from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .compat import BackendCompatibility, inspect_backend
from .execution import AppleChipProfile, ExecutionBackend
from .hardware import default_application_support, detect_hardware
from .types import HardwareInfo


CHIP_PROFILE_VERSION = 1


def detect_apple_chip_profile(
    hardware: HardwareInfo | None = None,
    compatibility: BackendCompatibility | None = None,
    *,
    backend_executable: str | Path | None = None,
    mlx_available: bool | None = None,
) -> AppleChipProfile:
    """Detect usable capabilities without loading a model or allocating large buffers."""
    detected = hardware or detect_hardware()
    backend = compatibility or inspect_backend(backend_executable)
    has_mlx = importlib.util.find_spec("mlx") is not None if mlx_available is None else mlx_available

    backends: list[ExecutionBackend] = []
    issues = list(backend.issues)
    if detected.is_apple_silicon and backend.compatible:
        backends.append(ExecutionBackend.VLLM_METAL)
    if detected.is_apple_silicon and has_mlx:
        backends.append(ExecutionBackend.NATIVE_MLX)
    backends.append(ExecutionBackend.CPU)

    precisions = ["fp32"]
    if detected.is_apple_silicon:
        precisions.insert(0, "fp16")
    identity = {
        "profile_version": CHIP_PROFILE_VERSION,
        "platform": detected.platform,
        "architecture": detected.architecture,
        "soc": detected.soc,
        "gpu_core_count": detected.gpu_core_count,
        "total_memory_bytes": detected.memory.total_bytes,
        "os_version": detected.os_version,
        "backends": [item.value for item in backends],
        "precisions": precisions,
        "vllm_version": backend.vllm_version,
        "vllm_metal_version": backend.vllm_metal_version,
        "transformers_version": backend.transformers_version,
        "platform_module": backend.platform_module,
        "platform_class": backend.platform_class,
        "platform_is_cpu": backend.platform_is_cpu,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return AppleChipProfile(
        profile_version=CHIP_PROFILE_VERSION,
        hardware_fingerprint=fingerprint,
        soc=detected.soc,
        total_memory_bytes=detected.memory.total_bytes,
        backends=tuple(backends),
        precisions=tuple(precisions),
        platform=detected.platform,
        architecture=detected.architecture,
        os_version=detected.os_version,
        gpu_core_count=detected.gpu_core_count,
        capability_issues=tuple(issues),
    )


def default_chip_profile_path(profile: AppleChipProfile) -> Path:
    return (
        default_application_support()
        / "profiles"
        / "execution"
        / f"{profile.hardware_fingerprint}.json"
    )


def save_chip_profile(
    profile: AppleChipProfile, path: Path | None = None
) -> Path:
    destination = path or default_chip_profile_path(profile)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(profile.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def load_chip_profile(path: Path) -> AppleChipProfile:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "profile_version",
        "hardware_fingerprint",
        "soc",
        "total_memory_bytes",
        "backends",
        "precisions",
        "platform",
        "architecture",
        "os_version",
        "gpu_core_count",
        "capability_issues",
        "measured_memory_bandwidth_bytes_per_second",
        "metal_launch_latency_nanoseconds",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("invalid AppleChipProfile fields")
    if payload.get("profile_version") != CHIP_PROFILE_VERSION:
        raise ValueError("unsupported AppleChipProfile version")
    string_fields = ("hardware_fingerprint", "soc", "platform", "architecture", "os_version")
    if any(not isinstance(payload[name], str) or not payload[name] for name in string_fields):
        raise ValueError("invalid AppleChipProfile string value")
    if (
        not isinstance(payload["total_memory_bytes"], int)
        or isinstance(payload["total_memory_bytes"], bool)
        or payload["total_memory_bytes"] <= 0
    ):
        raise ValueError("invalid AppleChipProfile memory value")
    for name in ("backends", "precisions", "capability_issues"):
        if not isinstance(payload[name], list) or not all(
            isinstance(value, str) and value for value in payload[name]
        ):
            raise ValueError(f"invalid AppleChipProfile {name}")
    for name in (
        "gpu_core_count",
        "measured_memory_bandwidth_bytes_per_second",
        "metal_launch_latency_nanoseconds",
    ):
        value = payload[name]
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ):
            raise ValueError(f"invalid AppleChipProfile {name}")
    try:
        return AppleChipProfile(
            profile_version=payload["profile_version"],
            hardware_fingerprint=payload["hardware_fingerprint"],
            soc=payload["soc"],
            total_memory_bytes=payload["total_memory_bytes"],
            backends=tuple(ExecutionBackend(value) for value in payload["backends"]),
            precisions=tuple(payload["precisions"]),
            platform=payload["platform"],
            architecture=payload["architecture"],
            os_version=payload["os_version"],
            gpu_core_count=payload["gpu_core_count"],
            capability_issues=tuple(payload["capability_issues"]),
            measured_memory_bandwidth_bytes_per_second=payload[
                "measured_memory_bandwidth_bytes_per_second"
            ],
            metal_launch_latency_nanoseconds=payload["metal_launch_latency_nanoseconds"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid AppleChipProfile values") from error
